"""SAE feature distillation into Boolean logic circuits — end-to-end.

Pipeline
--------
Phase 1 – Pretrain (one circuit per SAE feature):
    hidden [N, top_neurons]  →  LearnedThreshold  →  binary
        →  feature_circuit_i  →  predicts SAE feature i fires  (0/1)

Phase 2 – Joint fine-tune (all circuits + toxicity head, end-to-end):
    hidden  →  [8 thresholds + 8 feature circuits]  →  8 soft bits
        →  toxicity_circuit  →  toxic / clean

The result is a single Boolean circuit:
    raw LLM neurons  →  toxic / clean
with no SAE needed at inference time.

Checkpointing
-------------
After Phase 1: pretrain_checkpoint.pt
After Phase 2: finetuned_model.pt     ← full end-to-end model
Results JSON:  gemma_sae_distill_results.json

Usage
-----
# First run (full pipeline):
qwen/.venv/bin/python qwen/gemma_sae_distill.py

# Resume from saved pretrain checkpoint (skip Phase 1):
qwen/.venv/bin/python qwen/gemma_sae_distill.py --resume-pretrain

# Skip fine-tuning (Phase 1 only):
qwen/.venv/bin/python qwen/gemma_sae_distill.py --skip-finetune
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_scope_toxicity_classifier import (  # noqa: E402
    Metrics,
    binary_metrics,
    export_hard_logic_circuit,
)

DEFAULT_OUT_DIR   = HERE / "out"
DEFAULT_CACHE_DIR = HERE / "out" / "firing_cache"
DEFAULT_DIFFLOGIC = HERE.parent / "difflogic"
DEFAULT_JSON      = DEFAULT_OUT_DIR / "gemma_scope_layer20_en_top8_difflogic.json"
DEFAULT_SAVE_DIR  = DEFAULT_OUT_DIR / "distill_checkpoints"


# ── Straight-Through Threshold ─────────────────────────────────────────────

class LearnedThreshold(nn.Module):
    """Per-neuron learned threshold with temperature-annealed sigmoid.

    train: sigmoid((x - τ) / temperature)   — differentiable
    eval:  (x > τ).float()                  — hard binary
    """
    def __init__(self, n_neurons: int, init: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.full((n_neurons,), float(init)))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.sigmoid((x - self.tau) / max(self.temperature, 1e-6))
        return (x > self.tau).float()


# ── Logic circuit builder ─────────────────────────────────────────────────

def build_logic_circuit(
    in_dim: int,
    dims: list[int],
    *,
    device: torch.device,
    implementation: str,
    connections: str,
    difflogic_path: Path,
) -> nn.Sequential:
    sys.path.insert(0, str(difflogic_path.resolve()))
    from difflogic import LogicLayer  # type: ignore
    layers: list[nn.Module] = [nn.Flatten()]
    prev = in_dim
    for width in dims:
        layers.append(LogicLayer(prev, width, device=str(device),
                                 implementation=implementation,
                                 connections=connections))
        prev = width
    return nn.Sequential(*layers).to(device)


# ── End-to-end distilled model ─────────────────────────────────────────────

class DistilledToxicityModel(nn.Module):
    """8 feature circuits + toxicity head, jointly trainable."""

    def __init__(
        self,
        thresholds: list[LearnedThreshold],
        feature_circuits: list[nn.Sequential],
        toxicity_circuit: nn.Sequential,
    ):
        super().__init__()
        self.thresholds        = nn.ModuleList(thresholds)
        self.feature_circuits  = nn.ModuleList(feature_circuits)
        self.toxicity_circuit  = toxicity_circuit

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [N, top_neurons] float  →  [N] float (soft toxic score)."""
        soft_bits = []
        for thresh, circ in zip(self.thresholds, self.feature_circuits):
            x   = thresh(hidden)
            bit = circ(x).reshape(-1, 1)
            soft_bits.append(bit)
        x = torch.cat(soft_bits, dim=1)          # [N, n_features]
        return self.toxicity_circuit(x).reshape(-1)


# ── Checkpointing ─────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    *,
    model: DistilledToxicityModel,
    neuron_idx: torch.Tensor,
    feature_ids: list[int],
    dims_feature: list[int],
    dims_toxicity: list[int],
    phase: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "phase": phase,
        "neuron_idx": neuron_idx.cpu(),
        "feature_ids": feature_ids,
        "dims_feature": dims_feature,
        "dims_toxicity": dims_toxicity,
        "thresholds": [t.state_dict() for t in model.thresholds],
        "feature_circuits": [c.state_dict() for c in model.feature_circuits],
        "toxicity_circuit": model.toxicity_circuit.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"  saved checkpoint ({phase}): {path}")


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    implementation: str,
    connections: str,
    difflogic_path: Path,
) -> tuple[DistilledToxicityModel, torch.Tensor, list[int], list[int], list[int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron_idx    = ckpt["neuron_idx"]
    feature_ids   = ckpt["feature_ids"]
    dims_feature  = ckpt["dims_feature"]
    dims_toxicity = ckpt["dims_toxicity"]
    top_n         = len(neuron_idx)
    n_features    = len(feature_ids)

    thresholds       = [LearnedThreshold(top_n) for _ in range(n_features)]
    feature_circuits = [build_logic_circuit(top_n, dims_feature, device=device,
                                            implementation=implementation,
                                            connections=connections,
                                            difflogic_path=difflogic_path)
                        for _ in range(n_features)]
    toxicity_circuit = build_logic_circuit(n_features, dims_toxicity, device=device,
                                           implementation=implementation,
                                           connections=connections,
                                           difflogic_path=difflogic_path)

    for t, sd in zip(thresholds, ckpt["thresholds"]):
        t.load_state_dict(sd)
    for c, sd in zip(feature_circuits, ckpt["feature_circuits"]):
        c.load_state_dict(sd)
    toxicity_circuit.load_state_dict(ckpt["toxicity_circuit"])

    model = DistilledToxicityModel(thresholds, feature_circuits, toxicity_circuit).to(device)
    print(f"  loaded checkpoint ({ckpt['phase']}): {path}")
    return model, neuron_idx, feature_ids, dims_feature, dims_toxicity


# ── Neuron pre-selection ──────────────────────────────────────────────────

def select_top_neurons(
    x: torch.Tensor,     # [N, hidden_dim]
    y: torch.Tensor,     # [N] bool
    top_k: int,
) -> torch.Tensor:
    toxic_mean = x[y.bool()].mean(0)
    clean_mean = x[~y.bool()].mean(0)
    return (toxic_mean - clean_mean).abs().topk(top_k).indices


# ── Eval helper ───────────────────────────────────────────────────────────

def eval_model(
    model: DistilledToxicityModel,
    x: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
) -> Metrics:
    model.eval()
    with torch.no_grad():
        out  = model(x).cpu()
        pred = (out >= threshold).bool()
    return binary_metrics(y.bool().cpu(), pred)


# ── Phase 1: pretrain feature circuits ────────────────────────────────────

def pretrain_feature_circuits(
    model: DistilledToxicityModel,
    x_disc: torch.Tensor,
    firing_disc: torch.Tensor,     # [N_disc, 16k] bool
    x_eval: torch.Tensor,
    firing_eval: torch.Tensor,     # [N_eval, 16k] bool
    feature_ids: list[int],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold_lr: float,
    temperature: float,
    temperature_final: float,
    print_every: int,
) -> list[dict[str, Any]]:
    """Train each feature circuit + threshold independently."""
    results = []
    loss_fn = nn.BCELoss()
    x_disc_dev = x_disc.to(device)
    x_eval_dev = x_eval.to(device)

    for i, feat_id in enumerate(feature_ids):
        feat_name = f"i{i+1}"
        y_disc = firing_disc[:, feat_id].float().to(device)
        y_eval = firing_eval[:, feat_id].float()

        thresh  = model.thresholds[i]
        circuit = model.feature_circuits[i]
        thresh.temperature = temperature

        opt = torch.optim.Adam([
            {"params": circuit.parameters(), "lr": lr},
            {"params": thresh.parameters(),  "lr": threshold_lr},
        ])

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_disc_dev, y_disc),
            batch_size=batch_size, shuffle=True,
        )

        print(f"\n  [{feat_name}] fid={feat_id}  fire_disc={int(y_disc.sum())}/{len(y_disc)}")
        losses = []
        for epoch in range(1, epochs + 1):
            t = temperature + (temperature_final - temperature) * (epoch / epochs)
            thresh.temperature = t
            circuit.train(); thresh.train()
            epoch_loss = 0.0
            for bx, by in loader:
                bx = thresh(bx)
                out = circuit(bx).reshape(-1).clamp(1e-6, 1 - 1e-6)
                loss = loss_fn(out, by)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += float(loss) * bx.shape[0]
            losses.append(epoch_loss / len(y_disc))
            if print_every and (epoch == 1 or epoch % print_every == 0):
                print(f"    epoch={epoch:4d}  loss={losses[-1]:.5f}  "
                      f"τ=[{thresh.tau.min():.2f}, {thresh.tau.max():.2f}]")

        # eval
        circuit.eval(); thresh.eval()
        with torch.no_grad():
            out  = circuit(thresh(x_eval_dev)).reshape(-1).cpu()
            pred = (out >= 0.5).bool()
        m = binary_metrics(y_eval.bool(), pred)
        print(f"    eval hard → acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
              f"TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}")
        results.append({"feature": feat_name, "feature_id": feat_id,
                         "eval": asdict(m), "final_loss": losses[-1]})

    return results


# ── Phase 2: joint fine-tuning ────────────────────────────────────────────

def finetune_end_to_end(
    model: DistilledToxicityModel,
    x_disc: torch.Tensor,
    y_disc: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold_lr: float,
    temperature: float,
    temperature_final: float,
    print_every: int,
) -> list[dict[str, Any]]:
    """Fine-tune the entire stack end-to-end on toxicity labels. Restores best F1 weights."""
    loss_fn = nn.BCELoss()
    x_disc_dev = x_disc.to(device)
    y_disc_dev = y_disc.float().to(device)
    x_eval_dev = x_eval.to(device)

    for t in model.thresholds:
        t.temperature = temperature

    opt = torch.optim.Adam([
        {"params": model.toxicity_circuit.parameters(), "lr": lr},
        *[{"params": c.parameters(), "lr": lr} for c in model.feature_circuits],
        *[{"params": t.parameters(), "lr": threshold_lr} for t in model.thresholds],
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_disc_dev, y_disc_dev),
        batch_size=batch_size, shuffle=True,
    )

    print(f"\n  Joint fine-tuning  (epochs={epochs}  lr={lr}  thr_lr={threshold_lr})")
    losses = []
    snapshots = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t = temperature + (temperature_final - temperature) * (epoch / epochs)
        for thresh in model.thresholds:
            thresh.temperature = t
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            out  = model(bx).clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(out, by)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss) * bx.shape[0]
        scheduler.step()
        losses.append(epoch_loss / len(y_disc))

        # track best hard-circuit F1
        m = eval_model(model, x_eval_dev, y_eval)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_state = copy.deepcopy(model.state_dict())

        if print_every and (epoch == 1 or epoch % print_every == 0):
            print(f"    epoch={epoch:4d}  loss={losses[-1]:.5f}  "
                  f"eval → acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
                  f"TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}"
                  f"  [best={best_f1:.3f}]")
            snapshots.append({"epoch": epoch, "loss": losses[-1], "eval": asdict(m)})

    # restore best weights
    model.load_state_dict(best_state)
    final = eval_model(model, x_eval_dev, y_eval)
    print(f"\n  Restored best weights → acc={final.accuracy:.3f}  prec={final.precision:.3f}  "
          f"rec={final.recall:.3f}  f1={final.f1:.3f}  "
          f"TP={final.tp}  FP={final.fp}  TN={final.tn}  FN={final.fn}")
    return snapshots


# ── Phase 3: evolutionary hill-climbing on gate types ────────────────────

def _get_logic_layers(model: DistilledToxicityModel):
    """Yield all LogicLayer modules in the model."""
    for mod in model.modules():
        if type(mod).__name__ == "LogicLayer":
            yield mod


def _all_gates(model: DistilledToxicityModel) -> list[tuple[nn.Module, int]]:
    """Return list of (layer, gate_index) for every gate in the model."""
    gates = []
    for layer in _get_logic_layers(model):
        for g in range(layer.weights.shape[0]):
            gates.append((layer, g))
    return gates


def _current_gate_type(layer: nn.Module, g: int) -> int:
    return int(layer.weights.data[g].argmax().item())


def _set_gate_type(layer: nn.Module, g: int, gate_type: int) -> None:
    w = torch.zeros(16, device=layer.weights.device)
    w[gate_type] = 10.0
    layer.weights.data[g] = w


def evolve_circuits(
    model: DistilledToxicityModel,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    device: torch.device,
    n_random_iters: int = 2000,
    mutation_size_min: int = 2,
    mutation_size_max: int = 8,
    n_greedy_passes: int = 3,
    seed: int = 0,
) -> Metrics:
    """Phase 3: two-stage evolution.

    Stage A — Random multi-gate mutation (GPU-friendly, exploratory):
      Each iteration: pick k random gates (k in [min,max]), assign each a random
      new gate type, do ONE forward pass, keep if F1 improves. Fast because it's
      1 eval/iter regardless of k, and GPU processes all 1000 examples at once.

    Stage B — Greedy polish (thorough, convergence):
      For each gate try all 16 types, keep the best. One pass over all gates.
      Repeat until no gate improves. Used after random search to squeeze out
      any remaining single-gate improvements.
    """
    import time
    rng = torch.Generator()
    rng.manual_seed(seed)

    x_eval_dev = x_eval.to(device)
    model.eval()

    current = eval_model(model, x_eval_dev, y_eval)
    gates = _all_gates(model)
    n_gates = len(gates)

    print(f"\n── Phase 3: evolutionary search ──")
    print(f"  {n_gates} gates total")
    print(f"  Stage A: {n_random_iters} random mutations "
          f"(k={mutation_size_min}–{mutation_size_max} gates/iter, 1 eval/iter)")
    print(f"  Stage B: greedy polish (up to {n_greedy_passes} passes, "
          f"{n_gates * 16} evals/pass)")
    print(f"  start → f1={current.f1:.3f}  acc={current.accuracy:.3f}  "
          f"TP={current.tp}  FP={current.fp}  TN={current.tn}  FN={current.fn}\n")

    # ── Stage A: random multi-gate mutations ──────────────────────────────
    best_f1 = current.f1
    n_accepted = 0
    t0 = time.time()
    print("  [Stage A] random mutations")

    for it in range(1, n_random_iters + 1):
        k = int(torch.randint(mutation_size_min, mutation_size_max + 1, (1,), generator=rng).item())
        chosen_idx = torch.randperm(n_gates, generator=rng)[:k].tolist()

        # save current types and apply random mutations
        old_types = []
        for gi in chosen_idx:
            layer, g = gates[gi]
            old_types.append(_current_gate_type(layer, g))
            new_type = int(torch.randint(16, (1,), generator=rng).item())
            _set_gate_type(layer, g, new_type)

        m = eval_model(model, x_eval_dev, y_eval)
        if m.f1 > best_f1:
            best_f1 = m.f1
            current = m
            n_accepted += 1
        else:
            # revert
            for gi, old_t in zip(chosen_idx, old_types):
                layer, g = gates[gi]
                _set_gate_type(layer, g, old_t)

        if it % 200 == 0:
            elapsed = time.time() - t0
            rate = it / elapsed
            print(f"    iter {it:5d}/{n_random_iters}  accepted={n_accepted}"
                  f"  f1={current.f1:.3f}  acc={current.accuracy:.3f}"
                  f"  {elapsed:.0f}s  ({rate:.0f} iter/s)"
                  f"  ~{(n_random_iters - it) / rate:.0f}s left")

    print(f"  Stage A done → accepted {n_accepted}/{n_random_iters} mutations  "
          f"f1={current.f1:.3f}  ({time.time()-t0:.1f}s)\n")

    # ── Stage B: greedy polish ────────────────────────────────────────────
    print("  [Stage B] greedy polish")
    t0 = time.time()
    for pass_idx in range(1, n_greedy_passes + 1):
        n_improved = 0
        pass_t = time.time()
        for gi, (layer, g) in enumerate(gates):
            best_w = layer.weights.data[g].clone()
            best_f1_gate = current.f1
            for gate_type in range(16):
                _set_gate_type(layer, g, gate_type)
                m = eval_model(model, x_eval_dev, y_eval)
                if m.f1 > best_f1_gate:
                    best_f1_gate = m.f1
                    best_w = layer.weights.data[g].clone()
            layer.weights.data[g] = best_w
            if best_f1_gate > current.f1:
                current = eval_model(model, x_eval_dev, y_eval)
                n_improved += 1

            if (gi + 1) % 300 == 0:
                elapsed = time.time() - pass_t
                rate = (gi + 1) / elapsed if elapsed > 0 else 1
                print(f"    pass {pass_idx}  gate {gi+1:4d}/{n_gates}"
                      f"  improved={n_improved}"
                      f"  f1={current.f1:.3f}"
                      f"  {elapsed:.0f}s  ~{(n_gates - gi - 1) / rate:.0f}s left")

        print(f"  pass={pass_idx}  gates_improved={n_improved}  "
              f"f1={current.f1:.3f}  acc={current.accuracy:.3f}  "
              f"TP={current.tp}  FP={current.fp}  TN={current.tn}  FN={current.fn}  "
              f"({time.time()-pass_t:.1f}s)")
        if n_improved == 0:
            print(f"  converged after {pass_idx} passes")
            break

    print(f"\n  Final after evolution → acc={current.accuracy:.3f}  prec={current.precision:.3f}  "
          f"rec={current.recall:.3f}  f1={current.f1:.3f}  "
          f"TP={current.tp}  FP={current.fp}  TN={current.tn}  FN={current.fn}")
    return current


# ── Hard circuit export ────────────────────────────────────────────────────

def export_circuits(
    model: DistilledToxicityModel,
    neuron_idx: torch.Tensor,      # [top_n] — actual hidden-state neuron positions
    feature_ids: list[int],
    feature_meanings: dict[str, str],
) -> dict[str, Any]:
    """Export all circuits as hard Boolean expressions."""
    neuron_ids = neuron_idx.tolist()

    # one circuit per SAE feature: neurons → SAE feature bit
    feature_circuit_exprs = []
    for i, (feat_id, circ) in enumerate(zip(feature_ids, model.feature_circuits)):
        circ.eval()
        hard = export_hard_logic_circuit(circ, neuron_ids, output_mode="binary")
        tau  = model.thresholds[i].tau.detach().cpu().tolist()
        expr = hard.get("binary_output", {}).get("expression", "?")
        feature_circuit_exprs.append({
            "input_name":  f"i{i+1}",
            "feature_id":  feat_id,
            "meaning":     feature_meanings.get(f"i{i+1}", "?"),
            "expression":  expr,
            "tau_mean":    round(sum(tau) / len(tau), 4),
            "tau_std":     round((sum((v - sum(tau)/len(tau))**2 for v in tau) / len(tau))**0.5, 4),
            "hard_circuit": hard,
        })

    # toxicity circuit: 8 SAE feature bits → toxic/clean
    model.toxicity_circuit.eval()
    tox_hard = export_hard_logic_circuit(
        model.toxicity_circuit,
        list(range(len(feature_ids))),   # pseudo-ids 0..7; names will be i1..i8
        output_mode="binary",
    )
    tox_expr = tox_hard.get("binary_output", {}).get("expression", "?")

    return {
        "feature_circuits": feature_circuit_exprs,
        "toxicity_circuit": {
            "expression": tox_expr,
            "hard_circuit": tox_hard,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json",          default=str(DEFAULT_JSON))
    ap.add_argument("--cache-dir",     default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir",       default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--out-json",      default=None,
                    help="Output results JSON path. Default: <out-dir>/gemma_sae_distill_results.json")
    ap.add_argument("--save-dir",      default=str(DEFAULT_SAVE_DIR),
                    help="Directory for checkpoints (pretrain + finetuned model)")
    ap.add_argument("--difflogic-path", default=str(DEFAULT_DIFFLOGIC))

    # explicit cache file overrides (optional; if not set, uses glob in --cache-dir)
    ap.add_argument("--discovery-firing", default=None, help="Explicit path to discovery firing cache .pt")
    ap.add_argument("--eval-firing",      default=None, help="Explicit path to eval firing cache .pt")
    ap.add_argument("--discovery-hidden", default=None, help="Explicit path to discovery hidden cache .pt")
    ap.add_argument("--eval-hidden",      default=None, help="Explicit path to eval hidden cache .pt")

    # resume flags
    ap.add_argument("--resume-pretrain", action="store_true",
                    help="Load pretrain_checkpoint.pt from --save-dir and skip Phase 1")
    ap.add_argument("--resume-finetuned", action="store_true",
                    help="Load finetuned_model.pt from --save-dir and skip Phases 1+2 (go straight to evolution)")
    ap.add_argument("--skip-finetune",   action="store_true",
                    help="Skip Phase 2 joint fine-tuning")

    # neuron pre-selection
    ap.add_argument("--top-neurons",  type=int, default=256)

    # architecture
    ap.add_argument("--dims-feature",  default=None,
                    help="Feature circuit dims, e.g. '128,64,32,16,8,4,2,1'. Default: auto.")
    ap.add_argument("--dims-toxicity", default="4,2,1",
                    help="Toxicity circuit dims after 8 feature bits. Default: 4,2,1")

    # Phase 1 hyper-params
    ap.add_argument("--pretrain-epochs",      type=int,   default=200)
    ap.add_argument("--pretrain-batch",       type=int,   default=256)
    ap.add_argument("--pretrain-lr",          type=float, default=0.01)
    ap.add_argument("--pretrain-threshold-lr",type=float, default=0.005)

    # Phase 2 hyper-params
    ap.add_argument("--finetune-epochs",      type=int,   default=300)
    ap.add_argument("--finetune-batch",       type=int,   default=256)
    ap.add_argument("--finetune-lr",          type=float, default=0.005)
    ap.add_argument("--finetune-threshold-lr",type=float, default=0.002)

    # shared
    ap.add_argument("--temperature",          type=float, default=2.0)
    ap.add_argument("--temperature-final",    type=float, default=0.1)
    ap.add_argument("--connections",  default="random", choices=("random", "unique"))
    ap.add_argument("--implementation", default="python", choices=("python", "cuda", "auto"))
    ap.add_argument("--device",   default=None)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--print-every", type=int, default=50)

    # Phase 3 — evolutionary hill-climbing
    ap.add_argument("--skip-evolution", action="store_true",
                    help="Skip Phase 3 evolutionary gate-type hill-climbing")
    ap.add_argument("--evolution-random-iters", type=int, default=2000,
                    help="Stage A: number of random multi-gate mutation iters (default: 2000)")
    ap.add_argument("--evolution-mutation-min", type=int, default=2,
                    help="Stage A: minimum gates mutated per iteration (default: 2)")
    ap.add_argument("--evolution-mutation-max", type=int, default=8,
                    help="Stage A: maximum gates mutated per iteration (default: 8)")
    ap.add_argument("--evolution-greedy-passes", type=int, default=3,
                    help="Stage B: max greedy-polish passes over all gates (default: 3)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ckpt = save_dir / "pretrain_checkpoint.pt"
    finetune_ckpt = save_dir / "finetuned_model.pt"

    # ── device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    impl = args.implementation
    if impl == "auto":
        impl = "cuda" if device.type == "cuda" else "python"

    difflogic_path = Path(args.difflogic_path)

    # ── load experiment JSON ───────────────────────────────────────────────
    result      = json.loads(Path(args.json).read_text())
    features    = result["features"]
    feature_ids = [f["feature_id"] for f in features]
    fm          = result["logic_classifier"]["hard_circuit"]["feature_meanings"]
    meanings    = {
        f"i{i+1}": (fm[f"i{i+1}"].get("top_explanation") or f"feature_{feature_ids[i]}")
        for i in range(len(features))
    }
    n_features  = len(feature_ids)

    # ── resolve cache paths ────────────────────────────────────────────────
    def _resolve_cache(explicit: str | None, glob_pattern: str, label: str) -> Path:
        if explicit:
            p = Path(explicit)
            if not p.exists():
                print(f"ERROR: {label} cache not found: {p}"); sys.exit(1)
            return p
        hits = sorted(Path(args.cache_dir).glob(glob_pattern))
        if not hits:
            print(f"ERROR: {label} cache not found via glob '{glob_pattern}' in {args.cache_dir}")
            sys.exit(1)
        if len(hits) > 1:
            print(f"WARNING: multiple {label} caches found, using first: {hits[0].name}")
            print(f"  (use --{label.replace(' ', '-')} to specify explicitly)")
        return hits[0]

    disc_fire_p = _resolve_cache(args.discovery_firing, "gemma_scope_layer20_en_discovery_firing_*.pt", "discovery firing")
    eval_fire_p = _resolve_cache(args.eval_firing,      "gemma_scope_layer20_en_eval_firing_*.pt",      "eval firing")
    disc_hidd_p = _resolve_cache(args.discovery_hidden, "gemma_scope_layer20_en_discovery_hidden_*.pt", "discovery hidden")
    eval_hidd_p = _resolve_cache(args.eval_hidden,      "gemma_scope_layer20_en_eval_hidden_*.pt",      "eval hidden")

    firing_disc = torch.load(disc_fire_p, map_location="cpu", weights_only=False)["firing"].bool()
    firing_eval = torch.load(eval_fire_p, map_location="cpu", weights_only=False)["firing"].bool()
    labels_disc = torch.load(disc_fire_p, map_location="cpu", weights_only=False)["labels"].bool()
    labels_eval = torch.load(eval_fire_p, map_location="cpu", weights_only=False)["labels"].bool()

    hidden_disc = torch.load(disc_hidd_p, map_location="cpu", weights_only=False)["hidden"].float()
    hidden_eval = torch.load(eval_hidd_p, map_location="cpu", weights_only=False)["hidden"].float()

    print(f"Firing:  disc={tuple(firing_disc.shape)}  eval={tuple(firing_eval.shape)}")
    print(f"Hidden:  disc={tuple(hidden_disc.shape)}  eval={tuple(hidden_eval.shape)}")

    hidden_dim = hidden_disc.shape[1]

    # ── dims ──────────────────────────────────────────────────────────────
    dims_toxicity = [int(d) for d in args.dims_toxicity.split(",")]

    if args.resume_finetuned and finetune_ckpt.exists():
        # ── load finetuned model (skip Phase 1 + 2) ───────────────────────
        print(f"\n── Phases 1+2: loading finetuned model from {finetune_ckpt} ──")
        model, neuron_idx, feature_ids, dims_feature, dims_toxicity = load_checkpoint(
            finetune_ckpt, device=device, implementation=impl,
            connections=args.connections, difflogic_path=difflogic_path,
        )
        x_disc = hidden_disc.index_select(1, neuron_idx.cpu())
        x_eval = hidden_eval.index_select(1, neuron_idx.cpu())
        top_n  = len(neuron_idx)
        args.skip_finetune = True   # don't re-run Phase 2
    elif args.resume_pretrain and pretrain_ckpt.exists():
        # ── load pretrained model ──────────────────────────────────────────
        print(f"\n── Phase 1: loading from {pretrain_ckpt} ──")
        model, neuron_idx, feature_ids, dims_feature, dims_toxicity = load_checkpoint(
            pretrain_ckpt, device=device, implementation=impl,
            connections=args.connections, difflogic_path=difflogic_path,
        )
        x_disc = hidden_disc.index_select(1, neuron_idx.cpu())
        x_eval = hidden_eval.index_select(1, neuron_idx.cpu())
        top_n  = len(neuron_idx)
    else:
        # ── select neurons ─────────────────────────────────────────────────
        top_n = min(args.top_neurons, hidden_dim)
        print(f"\n── Neuron selection: top-{top_n} by |toxic_mean - clean_mean| ──")
        neuron_idx = select_top_neurons(hidden_disc, labels_disc, top_n)
        x_disc = hidden_disc.index_select(1, neuron_idx.cpu())
        x_eval = hidden_eval.index_select(1, neuron_idx.cpu())

        if args.dims_feature:
            dims_feature = [int(d) for d in args.dims_feature.split(",")]
        else:
            d, dims_feature = top_n, []
            while d > 1:
                d = max(1, d // 2)
                dims_feature.append(d)

        print(f"Feature circuit arch: {top_n} → {' → '.join(str(d) for d in dims_feature)}")
        print(f"Toxicity circuit arch: {n_features} → {' → '.join(str(d) for d in dims_toxicity)}")

        # ── build model ────────────────────────────────────────────────────
        thresholds = [LearnedThreshold(top_n, temperature=args.temperature)
                      for _ in range(n_features)]
        feature_circuits = [build_logic_circuit(top_n, dims_feature, device=device,
                                                 implementation=impl,
                                                 connections=args.connections,
                                                 difflogic_path=difflogic_path)
                             for _ in range(n_features)]
        toxicity_circuit = build_logic_circuit(n_features, dims_toxicity, device=device,
                                                implementation=impl,
                                                connections=args.connections,
                                                difflogic_path=difflogic_path)
        model = DistilledToxicityModel(thresholds, feature_circuits, toxicity_circuit).to(device)

        # ── Phase 1: pretrain ──────────────────────────────────────────────
        print(f"\n── Phase 1: pretraining {n_features} feature circuits ──")
        pretrain_results = pretrain_feature_circuits(
            model, x_disc, firing_disc, x_eval, firing_eval, feature_ids,
            device=device,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch,
            lr=args.pretrain_lr,
            threshold_lr=args.pretrain_threshold_lr,
            temperature=args.temperature,
            temperature_final=args.temperature_final,
            print_every=args.print_every,
        )
        save_checkpoint(pretrain_ckpt, model=model, neuron_idx=neuron_idx,
                        feature_ids=feature_ids, dims_feature=dims_feature,
                        dims_toxicity=dims_toxicity, phase="pretrain",
                        metrics={"pretrain": pretrain_results})

    # ── Phase 2: joint fine-tuning ─────────────────────────────────────────
    finetune_snapshots: list[dict] = []
    if not args.skip_finetune:
        print(f"\n── Phase 2: joint end-to-end fine-tuning ──")
        finetune_snapshots = finetune_end_to_end(
            model, x_disc, labels_disc, x_eval, labels_eval,
            device=device,
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch,
            lr=args.finetune_lr,
            threshold_lr=args.finetune_threshold_lr,
            temperature=args.temperature,
            temperature_final=args.temperature_final,
            print_every=args.print_every,
        )
        final_metrics = eval_model(model, x_eval.to(device), labels_eval)
        save_checkpoint(finetune_ckpt, model=model, neuron_idx=neuron_idx,
                        feature_ids=feature_ids, dims_feature=dims_feature,
                        dims_toxicity=dims_toxicity, phase="finetuned",
                        metrics={"finetune_snapshots": finetune_snapshots,
                                 "final": asdict(final_metrics)})
    else:
        final_metrics = eval_model(model, x_eval.to(device), labels_eval)

    # ── Phase 3: evolutionary hill-climbing ───────────────────────────────
    if not args.skip_evolution:
        final_metrics = evolve_circuits(
            model, x_eval.to(device), labels_eval,
            device=device,
            n_random_iters=args.evolution_random_iters,
            mutation_size_min=args.evolution_mutation_min,
            mutation_size_max=args.evolution_mutation_max,
            n_greedy_passes=args.evolution_greedy_passes,
            seed=args.seed,
        )
        evolution_ckpt = Path(args.save_dir) / "evolved_model.pt"
        save_checkpoint(evolution_ckpt, model=model, neuron_idx=neuron_idx,
                        feature_ids=feature_ids, dims_feature=dims_feature,
                        dims_toxicity=dims_toxicity, phase="evolved",
                        metrics={"final": asdict(final_metrics)})

    # ── export hard circuits ───────────────────────────────────────────────
    print(f"\n── Exporting hard Boolean circuits ──")
    hard_circuits = export_circuits(model, neuron_idx, feature_ids, meanings)

    print("\nFeature circuits:")
    for fc in hard_circuits["feature_circuits"]:
        print(f"  {fc['input_name']} (feat {fc['feature_id']}): {fc['expression'][:120]}")
    print(f"\nToxicity circuit: {hard_circuits['toxicity_circuit']['expression']}")

    # ── save results JSON ──────────────────────────────────────────────────
    out = {
        "top_neurons": top_n,
        "neuron_idx": neuron_idx.tolist(),
        "dims_feature": dims_feature,
        "dims_toxicity": dims_toxicity,
        "feature_ids": feature_ids,
        "feature_meanings": meanings,
        "final_eval_metrics": asdict(final_metrics),
        "finetune_snapshots": finetune_snapshots,
        "hard_circuits": hard_circuits,
        "checkpoints": {
            "pretrain":  str(pretrain_ckpt),
            "finetuned": str(finetune_ckpt) if not args.skip_finetune else None,
        },
    }
    out_path = Path(args.out_json) if args.out_json else Path(args.out_dir) / "gemma_sae_distill_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")

    print(f"\n{'='*60}")
    print(f"Final eval  acc={final_metrics.accuracy:.3f}  "
          f"prec={final_metrics.precision:.3f}  "
          f"rec={final_metrics.recall:.3f}  "
          f"f1={final_metrics.f1:.3f}")
    print(f"TP={final_metrics.tp}  FP={final_metrics.fp}  "
          f"TN={final_metrics.tn}  FN={final_metrics.fn}")
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"  pretrain:  {pretrain_ckpt.name}")
    if not args.skip_finetune:
        print(f"  finetuned: {finetune_ckpt.name}")


if __name__ == "__main__":
    main()
