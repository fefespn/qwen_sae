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
    """Fine-tune the entire stack end-to-end on toxicity labels."""
    loss_fn = nn.BCELoss()
    x_disc_dev = x_disc.to(device)
    y_disc_dev = y_disc.float().to(device)
    x_eval_dev = x_eval.to(device)

    # reset temperature on all thresholds
    for t in model.thresholds:
        t.temperature = temperature

    opt = torch.optim.Adam([
        {"params": model.toxicity_circuit.parameters(), "lr": lr},
        *[{"params": c.parameters(),                    "lr": lr}
          for c in model.feature_circuits],
        *[{"params": t.parameters(),                    "lr": threshold_lr}
          for t in model.thresholds],
    ])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_disc_dev, y_disc_dev),
        batch_size=batch_size, shuffle=True,
    )

    print(f"\n  Joint fine-tuning  (epochs={epochs}  lr={lr}  thr_lr={threshold_lr})")
    losses = []
    snapshots = []
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
        losses.append(epoch_loss / len(y_disc))
        if print_every and (epoch == 1 or epoch % print_every == 0):
            m = eval_model(model, x_eval_dev, y_eval)
            print(f"    epoch={epoch:4d}  loss={losses[-1]:.5f}  "
                  f"eval → acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
                  f"TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}")
            snapshots.append({"epoch": epoch, "loss": losses[-1], "eval": asdict(m)})

    final = eval_model(model, x_eval_dev, y_eval)
    print(f"\n  Final eval → acc={final.accuracy:.3f}  prec={final.precision:.3f}  "
          f"rec={final.recall:.3f}  f1={final.f1:.3f}  "
          f"TP={final.tp}  FP={final.fp}  TN={final.tn}  FN={final.fn}")
    return snapshots


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
    ap.add_argument("--save-dir",      default=str(DEFAULT_SAVE_DIR),
                    help="Directory for checkpoints (pretrain + finetuned model)")
    ap.add_argument("--difflogic-path", default=str(DEFAULT_DIFFLOGIC))

    # resume flags
    ap.add_argument("--resume-pretrain", action="store_true",
                    help="Load pretrain_checkpoint.pt from --save-dir and skip Phase 1")
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
    result     = json.loads(Path(args.json).read_text())
    features   = result["features"]
    feature_ids = [f["feature_id"] for f in features]
    fm          = result["logic_classifier"]["hard_circuit"]["feature_meanings"]
    meanings    = {f"i{i+1}": fm[f"i{i+1}"]["top_explanation"] for i in range(len(features))}
    n_features  = len(feature_ids)

    # ── load caches ────────────────────────────────────────────────────────
    disc_fire_f = sorted(Path(args.cache_dir).glob("gemma_scope_layer20_en_discovery_firing_*.pt"))
    eval_fire_f = sorted(Path(args.cache_dir).glob("gemma_scope_layer20_en_eval_firing_*.pt"))
    disc_hidd_f = sorted(Path(args.cache_dir).glob("gemma_scope_layer20_en_discovery_hidden_*.pt"))
    eval_hidd_f = sorted(Path(args.cache_dir).glob("gemma_scope_layer20_en_eval_hidden_*.pt"))

    for label, files in [("discovery firing", disc_fire_f), ("eval firing", eval_fire_f),
                          ("discovery hidden", disc_hidd_f), ("eval hidden", eval_hidd_f)]:
        if not files:
            print(f"ERROR: {label} cache not found in {args.cache_dir}")
            sys.exit(1)

    firing_disc = torch.load(disc_fire_f[0], map_location="cpu", weights_only=False)["firing"].bool()
    firing_eval = torch.load(eval_fire_f[0], map_location="cpu", weights_only=False)["firing"].bool()
    labels_disc = torch.load(disc_fire_f[0], map_location="cpu", weights_only=False)["labels"].bool()
    labels_eval = torch.load(eval_fire_f[0], map_location="cpu", weights_only=False)["labels"].bool()

    hidden_disc = torch.load(disc_hidd_f[0], map_location="cpu", weights_only=False)["hidden"].float()
    hidden_eval = torch.load(eval_hidd_f[0], map_location="cpu", weights_only=False)["hidden"].float()

    print(f"Firing:  disc={tuple(firing_disc.shape)}  eval={tuple(firing_eval.shape)}")
    print(f"Hidden:  disc={tuple(hidden_disc.shape)}  eval={tuple(hidden_eval.shape)}")

    hidden_dim = hidden_disc.shape[1]

    # ── dims ──────────────────────────────────────────────────────────────
    dims_toxicity = [int(d) for d in args.dims_toxicity.split(",")]

    if args.resume_pretrain and pretrain_ckpt.exists():
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
    out_path = Path(args.out_dir) / "gemma_sae_distill_results.json"
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
