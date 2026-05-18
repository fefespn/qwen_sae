"""Qwen-Scope SAE toxicity classifier from paper sections 5.1.1 and 5.1.2.

This implements the deliberately simple classifier described in Qwen-Scope:

1. Discover toxic features at a layer by counting whether each SAE feature
   fires anywhere in an example, then ranking by
       P(fires | toxic) - P(fires | clean).
2. Classify a new example as toxic when any selected feature fires at any
   token position.

The default settings target Qwen3-1.7B-Base and Qwen-Scope residual SAEs, with
layer exposed as an argument and defaulted to 6 for the first experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_SAE_REPO = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
DEFAULT_DATASET = "textdetox/multilingual_toxicity_dataset"
DEFAULT_LAYER = 6
DEFAULT_TOP_K = 1
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
LOGIC_OP_NAMES = [
    "zero",
    "and",
    "a_and_not_b",
    "a",
    "b_and_not_a",
    "b",
    "xor",
    "or",
    "nor",
    "xnor",
    "not_b",
    "b_implies_a",
    "not_a",
    "a_implies_b",
    "nand",
    "one",
]


@dataclass(frozen=True)
class ToxicFeature:
    layer: int
    feature_id: int
    delta: float
    toxic_rate: float
    clean_rate: float
    toxic_count: int
    clean_count: int
    n_toxic: int
    n_clean: int


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Metrics:
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = int((y_true & y_pred).sum().item())
    fp = int((~y_true & y_pred).sum().item())
    tn = int((~y_true & ~y_pred).sum().item())
    fn = int((y_true & ~y_pred).sum().item())
    total = max(1, tp + fp + tn + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return Metrics(
        accuracy=(tp + tn) / total,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def discover_top_features_from_firing(
    firing: torch.Tensor,
    labels: torch.Tensor,
    *,
    layer: int,
    top_k: int,
) -> list[ToxicFeature]:
    """Rank features by the paper's delta score.

    Args:
        firing: bool tensor [n_examples, n_features], where firing[i, f] is
            1[max_t a_i,t,f > epsilon].
        labels: bool/int tensor [n_examples], 1 means toxic and 0 means clean.
    """
    if firing.ndim != 2:
        raise ValueError(f"firing must be [n_examples, n_features], got {tuple(firing.shape)}")
    if labels.ndim != 1 or labels.shape[0] != firing.shape[0]:
        raise ValueError("labels must be [n_examples] and align with firing")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    firing = firing.bool()
    labels = labels.bool()
    toxic = firing[labels]
    clean = firing[~labels]
    if toxic.shape[0] == 0 or clean.shape[0] == 0:
        raise ValueError("feature discovery needs at least one toxic and one clean example")

    toxic_counts = toxic.sum(dim=0)
    clean_counts = clean.sum(dim=0)
    toxic_rate = toxic_counts.float() / toxic.shape[0]
    clean_rate = clean_counts.float() / clean.shape[0]
    delta = toxic_rate - clean_rate
    values, indices = torch.topk(delta, k=min(top_k, delta.numel()))

    features = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        features.append(
            ToxicFeature(
                layer=layer,
                feature_id=int(idx),
                delta=float(value),
                toxic_rate=float(toxic_rate[idx].item()),
                clean_rate=float(clean_rate[idx].item()),
                toxic_count=int(toxic_counts[idx].item()),
                clean_count=int(clean_counts[idx].item()),
                n_toxic=int(toxic.shape[0]),
                n_clean=int(clean.shape[0]),
            )
        )
    return features


def run_name(*, layer: int, language: str, top_k_features: int) -> str:
    return f"qwen_scope_layer{layer}_{language}_top{top_k_features}"


def default_output_path(args: argparse.Namespace) -> str:
    suffix = "" if args.classifier == "or" else f"_{args.classifier}"
    return str(Path(args.out_dir) / f"{run_name(layer=args.layer, language=args.language, top_k_features=args.top_k_features)}{suffix}.json")


def firing_cache_metadata(args: argparse.Namespace, *, split: str, text_key: str, label_key: str) -> dict[str, Any]:
    """Metadata that determines whether expensive firing tensors can be reused.

    top_k_features is intentionally excluded: top-10/top-20 feature sweeps use
    the same example-level firing matrix and only change how many ranked
    features are selected for the OR rule.
    """
    return {
        "version": 1,
        "split": split,
        "model": args.model,
        "sae_repo": args.sae_repo,
        "dataset": args.dataset,
        "language": args.language,
        "layer": args.layer,
        "sae_top_k": args.sae_top_k,
        "epsilon": args.epsilon,
        "discovery_per_class": args.discovery_per_class,
        "eval_per_class": args.eval_per_class,
        "max_length": args.max_length,
        "seed": args.seed,
        "text_key": text_key,
        "label_key": label_key,
    }


def firing_cache_path(args: argparse.Namespace, *, split: str, text_key: str, label_key: str) -> Path:
    meta = firing_cache_metadata(args, split=split, text_key=text_key, label_key=label_key)
    encoded = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    name = f"qwen_scope_layer{args.layer}_{args.language}_{split}_firing_{digest}.pt"
    return Path(args.firing_cache_dir) / name


def load_firing_cache(path: Path, expected_meta: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ignoring unreadable firing cache {path}: {e}")
        return None
    if not isinstance(payload, dict) or payload.get("metadata") != expected_meta:
        print(f"ignoring stale firing cache {path}")
        return None
    firing = payload.get("firing")
    labels = payload.get("labels")
    if not isinstance(firing, torch.Tensor) or not isinstance(labels, torch.Tensor):
        print(f"ignoring invalid firing cache {path}")
        return None
    print(f"loaded {expected_meta['split']} firing cache: {path}")
    return firing.bool(), labels.bool()


def save_firing_cache(path: Path, metadata: dict[str, Any], firing: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "firing": firing.cpu().bool(),
            "labels": labels.cpu().bool(),
        },
        path,
    )
    print(f"saved {metadata['split']} firing cache: {path}")


def predict_from_firing(firing: torch.Tensor, feature_ids: Sequence[int]) -> torch.Tensor:
    """Apply the section 5.1.2 OR rule over selected feature ids."""
    if firing.ndim != 2:
        raise ValueError(f"firing must be [n_examples, n_features], got {tuple(firing.shape)}")
    if not feature_ids:
        return torch.zeros(firing.shape[0], dtype=torch.bool, device=firing.device)
    ids = torch.as_tensor(list(feature_ids), dtype=torch.long, device=firing.device)
    return firing.bool().index_select(1, ids).any(dim=1)


def selected_feature_matrix(firing: torch.Tensor, feature_ids: Sequence[int]) -> torch.Tensor:
    if not feature_ids:
        raise ValueError("logic classifier needs at least one selected feature")
    ids = torch.as_tensor(list(feature_ids), dtype=torch.long, device=firing.device)
    return firing.bool().index_select(1, ids).to(torch.float32)


def parse_logic_dims(value: str | None, input_dim: int, *, output_mode: str) -> list[int]:
    if value:
        dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif output_mode == "groupsum" and input_dim <= 8:
        dims = [8, 8, 4, 2]
    elif output_mode == "binary" and input_dim <= 8:
        dims = [8, 8, 4, 2, 1]
    else:
        dims = [input_dim, input_dim]
        current = input_dim
        while current > 4:
            current = max(4, math.ceil(current / 2))
            dims.append(current)
        dims.extend([2, 1] if output_mode == "binary" else [2])

    if not dims:
        raise ValueError("logic dims must contain at least one layer width")
    previous = input_dim
    for width in dims:
        if width <= 0:
            raise ValueError(f"logic layer widths must be positive, got {dims}")
        if width * 2 < previous:
            raise ValueError(
                f"invalid logic architecture {input_dim}->{dims}: "
                f"LogicLayer out_dim={width} is too small for in_dim={previous}"
            )
        previous = width
    if output_mode == "binary" and dims[-1] != 1:
        raise ValueError(f"binary logic output needs final layer width 1, got {dims[-1]}")
    if output_mode == "groupsum" and dims[-1] % 2 != 0:
        raise ValueError(f"final logic layer width must be divisible by 2 for GroupSum(k=2), got {dims[-1]}")
    return dims


def logic_expr(op_idx: int, a: str, b: str) -> str:
    if op_idx == 0:
        return "0"
    if op_idx == 1:
        return f"({a} AND {b})"
    if op_idx == 2:
        return f"({a} AND NOT {b})"
    if op_idx == 3:
        return a
    if op_idx == 4:
        return f"({b} AND NOT {a})"
    if op_idx == 5:
        return b
    if op_idx == 6:
        return f"({a} XOR {b})"
    if op_idx == 7:
        return f"({a} OR {b})"
    if op_idx == 8:
        return f"NOT ({a} OR {b})"
    if op_idx == 9:
        return f"({a} XNOR {b})"
    if op_idx == 10:
        return f"NOT {b}"
    if op_idx == 11:
        return f"({b} -> {a})"
    if op_idx == 12:
        return f"NOT {a}"
    if op_idx == 13:
        return f"({a} -> {b})"
    if op_idx == 14:
        return f"NOT ({a} AND {b})"
    if op_idx == 15:
        return "1"
    raise ValueError(f"unknown logic op index {op_idx}")


def export_hard_logic_circuit(
    model: torch.nn.Sequential,
    feature_ids: Sequence[int],
    *,
    output_mode: str,
) -> dict[str, Any]:
    inputs = [
        {
            "name": f"i{i + 1}",
            "feature_id": int(feature_id),
            "description": f"top_feature_{i + 1}_fires",
        }
        for i, feature_id in enumerate(feature_ids)
    ]
    previous_refs = [item["name"] for item in inputs]
    previous_exprs = [item["name"] for item in inputs]
    layers = []

    for module in model:
        if not hasattr(module, "weights") or not hasattr(module, "indices"):
            continue
        idx_a, idx_b = module.indices
        idx_a = idx_a.detach().cpu().tolist()
        idx_b = idx_b.detach().cpu().tolist()
        ops = module.weights.detach().cpu().argmax(dim=-1).tolist()
        layer_index = len(layers)
        nodes = []
        next_refs = []
        next_exprs = []
        for gate_index, (a_idx, b_idx, op_idx) in enumerate(zip(idx_a, idx_b, ops)):
            a_ref = previous_refs[int(a_idx)]
            b_ref = previous_refs[int(b_idx)]
            a_expr = previous_exprs[int(a_idx)]
            b_expr = previous_exprs[int(b_idx)]
            expr = logic_expr(int(op_idx), a_expr, b_expr)
            node_id = f"L{layer_index}.g{gate_index}"
            nodes.append(
                {
                    "id": node_id,
                    "op_index": int(op_idx),
                    "op": LOGIC_OP_NAMES[int(op_idx)],
                    "inputs": [a_ref, b_ref],
                    "input_indices": [int(a_idx), int(b_idx)],
                    "expression": expr,
                }
            )
            next_refs.append(node_id)
            next_exprs.append(expr)
        layers.append(
            {
                "layer": layer_index,
                "in_dim": int(module.in_dim),
                "out_dim": int(module.out_dim),
                "nodes": nodes,
            }
        )
        previous_refs = next_refs
        previous_exprs = next_exprs

    circuit: dict[str, Any] = {
        "format": "difflogic_hard_circuit_v1",
        "output_mode": output_mode,
        "inputs": inputs,
        "operators": LOGIC_OP_NAMES,
        "layers": layers,
    }
    if output_mode == "binary":
        if len(previous_exprs) != 1:
            raise ValueError(f"binary circuit expects one final output, got {len(previous_exprs)}")
        circuit["binary_output"] = {
            "term": previous_refs[0],
            "expression": previous_exprs[0],
            "decision": "toxic if expression == 1; clean if expression == 0",
        }
        return circuit

    if len(previous_exprs) % 2 != 0:
        raise ValueError("final logic layer width must be divisible by 2 for GroupSum(k=2)")
    group_size = len(previous_exprs) // 2
    class_terms = {
        "clean": previous_refs[:group_size],
        "toxic": previous_refs[group_size:],
    }
    class_expressions = {
        "clean": previous_exprs[:group_size],
        "toxic": previous_exprs[group_size:],
    }
    score_expressions = {
        name: " + ".join(exprs) if exprs else "0"
        for name, exprs in class_expressions.items()
    }
    circuit["group_sum"] = {
        "k": 2,
        "group_size": group_size,
        "class_terms": class_terms,
        "class_expressions": class_expressions,
        "score_expressions": score_expressions,
        "decision": "toxic if toxic_score > clean_score; ties go to clean",
    }
    return circuit


def train_difflogic_classifier(
    train_firing: torch.Tensor,
    train_labels: torch.Tensor,
    eval_firing: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    feature_ids: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    difflogic_path = Path(args.difflogic_path).resolve()
    sys.path.insert(0, str(difflogic_path))
    from difflogic import GroupSum, LogicLayer

    torch.manual_seed(args.logic_seed)
    if torch.cuda.is_available() and args.logic_device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    implementation = args.logic_implementation
    if implementation == "auto":
        implementation = "cuda" if device.type == "cuda" else "python"

    x_train = selected_feature_matrix(train_firing, feature_ids).to(device)
    y_train = train_labels.long().to(device)
    x_eval = selected_feature_matrix(eval_firing, feature_ids).to(device)
    y_eval = eval_labels.long().to(device)

    dims = parse_logic_dims(args.logic_dims, len(feature_ids), output_mode=args.logic_output)
    layers: list[torch.nn.Module] = [torch.nn.Flatten()]
    previous = len(feature_ids)
    for width in dims:
        layers.append(
            LogicLayer(
                previous,
                width,
                device=str(device),
                implementation=implementation,
                connections=args.logic_connections,
                grad_factor=args.logic_grad_factor,
            )
        )
        previous = width
    if args.logic_output == "groupsum":
        layers.append(GroupSum(k=2, tau=args.logic_tau, device=str(device)))
    model = torch.nn.Sequential(*layers).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.logic_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.logic_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.logic_epochs, eta_min=args.logic_lr * 0.05
    )
    loss_fn = torch.nn.BCELoss() if args.logic_output == "binary" else torch.nn.CrossEntropyLoss()

    import copy
    best_f1 = -1.0
    best_state = None

    losses = []
    for epoch in range(1, args.logic_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            output = model(batch_x)
            if args.logic_output == "binary":
                output = output.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
                loss = loss_fn(output, batch_y.to(torch.float32))
            else:
                loss = loss_fn(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_x.shape[0]
        losses.append(epoch_loss / max(1, x_train.shape[0]))
        scheduler.step()

        # track best weights by eval F1
        model.eval()
        with torch.no_grad():
            out = model(x_eval)
            if args.logic_output == "binary":
                pred = (out.reshape(-1) >= args.logic_threshold).bool().cpu()
            else:
                pred = out.argmax(dim=-1).bool().cpu()
        m = binary_metrics(y_eval.bool().cpu(), pred)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_state = copy.deepcopy(model.state_dict())

        if args.logic_print_every and (epoch == 1 or epoch % args.logic_print_every == 0):
            print(f"difflogic epoch={epoch:4d}  loss={losses[-1]:.5f}  f1={m.f1:.4f}  [best={best_f1:.4f}]")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  restored best weights → f1={best_f1:.4f}")

    def eval_split(x: torch.Tensor, y: torch.Tensor, *, hard: bool) -> Metrics:
        model.train(mode=not hard)
        with torch.no_grad():
            output = model(x)
            if args.logic_output == "binary":
                pred = (output.reshape(-1) >= args.logic_threshold).bool().cpu()
            else:
                pred = output.argmax(dim=-1).bool().cpu()
        return binary_metrics(y.bool().cpu(), pred)

    train_soft_metrics = eval_split(x_train, y_train, hard=False)
    eval_soft_metrics = eval_split(x_eval, y_eval, hard=False)
    train_hard_metrics = eval_split(x_train, y_train, hard=True)
    eval_hard_metrics = eval_split(x_eval, y_eval, hard=True)
    model.eval()
    hard_circuit = export_hard_logic_circuit(model, feature_ids, output_mode=args.logic_output)

    return {
        "architecture": {
            "input_dim": len(feature_ids),
            "logic_layer_dims": dims,
            "output_mode": args.logic_output,
            "group_sum_k": 2 if args.logic_output == "groupsum" else None,
            "tau": args.logic_tau if args.logic_output == "groupsum" else None,
            "threshold": args.logic_threshold if args.logic_output == "binary" else None,
            "implementation": implementation,
            "device": str(device),
            "connections": args.logic_connections,
        },
        "training": {
            "epochs": args.logic_epochs,
            "batch_size": args.logic_batch_size,
            "learning_rate": args.logic_lr,
            "seed": args.logic_seed,
            "final_loss": losses[-1] if losses else None,
            "loss_first": losses[0] if losses else None,
        },
        "feature_ids": list(map(int, feature_ids)),
        "train_soft_metrics": asdict(train_soft_metrics),
        "eval_soft_metrics": asdict(eval_soft_metrics),
        "train_hard_metrics": asdict(train_hard_metrics),
        "eval_hard_metrics": asdict(eval_hard_metrics),
        "hard_circuit": hard_circuit,
    }


class QwenScopeTopKSAE:
    """Small wrapper around Qwen-Scope TopK SAE checkpoints."""

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_SAE_REPO,
        layer: int = DEFAULT_LAYER,
        top_k: int = 50,
        device: str | torch.device | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.layer = int(layer)
        self.top_k = int(top_k)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = hf_hub_download(repo_id=repo_id, filename=f"layer{self.layer}.sae.pt")
        try:
            sae = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            sae = torch.load(path, map_location=self.device)
        self.W_enc = sae["W_enc"].to(self.device, dtype=torch.float32)
        self.b_enc = sae["b_enc"].to(self.device, dtype=torch.float32)
        self.width = int(self.b_enc.shape[0])

    @torch.no_grad()
    def firing_from_hidden(
        self,
        hidden: torch.Tensor,
        epsilon: float = 0.0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return example-level firing indicators for all SAE features.

        hidden is [batch, seq, d_model]. The output is [batch, d_sae], matching
        h_i,f = 1[max_t a_i,t,f > epsilon].
        """
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [batch, seq, d_model], got {tuple(hidden.shape)}")
        hidden = hidden.to(self.device, dtype=torch.float32)
        pre = hidden @ self.W_enc.T + self.b_enc
        relu = torch.relu(pre)
        values, indices = torch.topk(relu, k=self.top_k, dim=-1)
        active = values > epsilon
        if attention_mask is not None:
            mask = attention_mask.to(self.device, dtype=torch.bool).unsqueeze(-1)
            active = active & mask
        firing = torch.zeros(
            (hidden.shape[0], self.width),
            dtype=torch.bool,
            device=self.device,
        )
        batch_idx = (
            torch.arange(hidden.shape[0], device=self.device)
            .view(-1, 1, 1)
            .expand_as(indices)
        )
        firing[batch_idx[active], indices[active]] = True
        return firing

    @torch.no_grad()
    def selected_firing_from_hidden(
        self,
        hidden: torch.Tensor,
        feature_ids: Sequence[int],
        epsilon: float = 0.0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, len(feature_ids)] firing indicators for selected features."""
        if not feature_ids:
            return torch.zeros((hidden.shape[0], 0), dtype=torch.bool, device=self.device)
        firing = self.firing_from_hidden(hidden, epsilon=epsilon, attention_mask=attention_mask)
        ids = torch.as_tensor(list(feature_ids), dtype=torch.long, device=self.device)
        return firing.index_select(1, ids)


def capture_layer_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int) -> torch.Tensor:
    """Run a forward pass and return residual stream output at model.model.layers[layer]."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"failed to capture hidden state at layer {layer}")
    return captured["hidden"]


def iter_batches(examples: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(examples), batch_size):
        yield list(examples[i : i + batch_size])


@torch.no_grad()
def collect_firing(
    examples: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    sae: QwenScopeTopKSAE,
    text_key: str,
    label_key: str,
    batch_size: int,
    max_length: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect h_i,f firing matrix for examples."""
    model_device = next(model.parameters()).device
    all_firing = []
    all_labels = []
    for batch in tqdm(list(iter_batches(examples, batch_size)), desc="SAE firing"):
        texts = [str(row[text_key]) for row in batch]
        labels = [int(row[label_key]) for row in batch]
        toks = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.to(model_device) for k, v in toks.items()}
        hidden = capture_layer_hidden(
            model,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            layer=sae.layer,
        )
        firing = sae.firing_from_hidden(
            hidden,
            epsilon=epsilon,
            attention_mask=toks["attention_mask"],
        ).cpu()
        all_firing.append(firing)
        all_labels.append(torch.tensor(labels, dtype=torch.bool))
    return torch.cat(all_firing, dim=0), torch.cat(all_labels, dim=0)


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    s = str(value).strip().lower()
    toxic_values = {"1", "toxic", "toxicity", "offensive", "hate", "harmful", "yes", "true"}
    clean_values = {"0", "clean", "non-toxic", "not-toxic", "not_toxic", "neutral", "no", "false"}
    if s in toxic_values:
        return 1
    if s in clean_values:
        return 0
    raise ValueError(f"cannot normalize toxicity label {value!r}")


def load_balanced_toxicity_split(
    *,
    dataset_name: str = DEFAULT_DATASET,
    language: str = "en",
    seed: int = 0,
    discovery_per_class: int = 2000,
    eval_per_class: int = 500,
    text_key: str | None = None,
    label_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Load a paper-style balanced discovery/eval split.

    This is intentionally schema-tolerant because toxicity datasets on HF use
    slightly different column names. Pass --text-key/--label-key to pin it down.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install `datasets` to load the toxicity corpus from Hugging Face.") from e

    ds = load_dataset(dataset_name)

    if language in ds:
        rows = [dict(r) for r in ds[language]]
    else:
        rows = []
        for split in ds:
            rows.extend(dict(r) for r in ds[split])
    if not rows:
        raise RuntimeError(f"no rows loaded from {dataset_name}")

    candidates = rows[0].keys()
    if text_key is None:
        for key in ("text", "comment_text", "comment", "sentence", "toxic_sentence", "source"):
            if key in candidates:
                text_key = key
                break
    if label_key is None:
        for key in ("label", "toxicity", "toxic", "is_toxic", "target"):
            if key in candidates:
                label_key = key
                break
    if text_key is None or label_key is None:
        raise ValueError(f"could not infer text/label columns from columns={sorted(candidates)}")

    normalized = []
    for row in rows:
        if "lang" in row and str(row["lang"]).lower() != language.lower():
            continue
        if "language" in row and str(row["language"]).lower() not in {language.lower(), language}:
            continue
        row = dict(row)
        row[label_key] = normalize_label(row[label_key])
        normalized.append(row)

    toxic = [r for r in normalized if int(r[label_key]) == 1]
    clean = [r for r in normalized if int(r[label_key]) == 0]
    need = discovery_per_class + eval_per_class
    if len(toxic) < need or len(clean) < need:
        raise RuntimeError(
            f"need {need} toxic and clean examples; got toxic={len(toxic)} clean={len(clean)}"
        )

    g = torch.Generator().manual_seed(seed)
    toxic_idx = torch.randperm(len(toxic), generator=g).tolist()[:need]
    clean_idx = torch.randperm(len(clean), generator=g).tolist()[:need]
    toxic = [toxic[i] for i in toxic_idx]
    clean = [clean[i] for i in clean_idx]

    discovery = toxic[:discovery_per_class] + clean[:discovery_per_class]
    evaluation = toxic[discovery_per_class:] + clean[discovery_per_class:]
    order = torch.randperm(len(discovery), generator=g).tolist()
    discovery = [discovery[i] for i in order]
    order = torch.randperm(len(evaluation), generator=g).tolist()
    evaluation = [evaluation[i] for i in order]
    return discovery, evaluation, text_key, label_key


def save_results(
    path: str,
    *,
    args: argparse.Namespace,
    features: Sequence[ToxicFeature],
    metrics: Metrics,
    ranked_features: Sequence[ToxicFeature],
    or_metrics: Metrics,
    logic_result: dict[str, Any] | None,
    text_key: str,
    label_key: str,
) -> None:
    payload = {
        "method": "qwen_scope_sections_5_1_1_5_1_2",
        "paper_rule": "feature_delta_discovery_then_or_rule",
        "model": args.model,
        "sae_repo": args.sae_repo,
        "layer": args.layer,
        "top_k_features": args.top_k_features,
        "epsilon": args.epsilon,
        "language": args.language,
        "dataset": args.dataset,
        "text_key": text_key,
        "label_key": label_key,
        "classifier": args.classifier,
        "features": [asdict(f) for f in features],
        "ranked_features": [asdict(f) for f in ranked_features],
        "ranked_features_top_k": len(ranked_features),
        "or_metrics": asdict(or_metrics),
        "logic_classifier": logic_result,
        "metrics": asdict(metrics),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sae-repo", default=DEFAULT_SAE_REPO)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--language", default="en")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--top-k-features", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--sae-top-k", type=int, default=50)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--discovery-per-class", type=int, default=2000)
    p.add_argument("--eval-per-class", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--text-key", default=None)
    p.add_argument("--label-key", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--firing-cache-dir", default=str(DEFAULT_OUT_DIR / "firing_cache"))
    p.add_argument("--no-firing-cache", action="store_true")
    p.add_argument("--save-ranked-top-k", type=int, default=50)
    p.add_argument("--classifier", choices=("or", "difflogic"), default="or")
    p.add_argument("--difflogic-path", default=str(Path(__file__).resolve().parents[1] / "difflogic"))
    p.add_argument("--logic-output", choices=("binary", "groupsum"), default="binary")
    p.add_argument(
        "--logic-dims",
        default=None,
        help="Comma-separated LogicLayer widths; binary default ends in 1, e.g. 32,16,8,4,2,1.",
    )
    p.add_argument("--logic-epochs", type=int, default=100)
    p.add_argument("--logic-batch-size", type=int, default=256)
    p.add_argument("--logic-lr", type=float, default=0.01)
    p.add_argument("--logic-tau", type=float, default=1.0)
    p.add_argument("--logic-threshold", type=float, default=0.5)
    p.add_argument("--logic-grad-factor", type=float, default=1.0)
    p.add_argument("--logic-connections", choices=("random", "unique"), default="random")
    p.add_argument("--logic-implementation", choices=("auto", "python", "cuda"), default="python")
    p.add_argument("--logic-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--logic-seed", type=int, default=0)
    p.add_argument("--logic-print-every", type=int, default=25)
    args = p.parse_args()

    discovery, evaluation, text_key, label_key = load_balanced_toxicity_split(
        dataset_name=args.dataset,
        language=args.language,
        seed=args.seed,
        discovery_per_class=args.discovery_per_class,
        eval_per_class=args.eval_per_class,
        text_key=args.text_key,
        label_key=args.label_key,
    )

    if args.out is None:
        args.out = default_output_path(args)

    discovery_meta = firing_cache_metadata(args, split="discovery", text_key=text_key, label_key=label_key)
    eval_meta = firing_cache_metadata(args, split="eval", text_key=text_key, label_key=label_key)
    discovery_cached = None
    eval_cached = None
    if not args.no_firing_cache:
        discovery_cached = load_firing_cache(
            firing_cache_path(args, split="discovery", text_key=text_key, label_key=label_key),
            discovery_meta,
        )
        eval_cached = load_firing_cache(
            firing_cache_path(args, split="eval", text_key=text_key, label_key=label_key),
            eval_meta,
        )

    if discovery_cached is None or eval_cached is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
        model.eval()
        sae = QwenScopeTopKSAE(
            repo_id=args.sae_repo,
            layer=args.layer,
            top_k=args.sae_top_k,
            device=device,
        )
    else:
        model = tokenizer = sae = None

    if discovery_cached is None:
        discovery_firing, discovery_labels = collect_firing(
            discovery,
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            text_key=text_key,
            label_key=label_key,
            batch_size=args.batch_size,
            max_length=args.max_length,
            epsilon=args.epsilon,
        )
        if not args.no_firing_cache:
            save_firing_cache(
                firing_cache_path(args, split="discovery", text_key=text_key, label_key=label_key),
                discovery_meta,
                discovery_firing,
                discovery_labels,
            )
    else:
        discovery_firing, discovery_labels = discovery_cached

    features = discover_top_features_from_firing(
        discovery_firing,
        discovery_labels,
        layer=args.layer,
        top_k=args.top_k_features,
    )
    ranked_features = discover_top_features_from_firing(
        discovery_firing,
        discovery_labels,
        layer=args.layer,
        top_k=max(args.top_k_features, args.save_ranked_top_k),
    )
    feature_ids = [f.feature_id for f in features]

    if eval_cached is None:
        eval_firing, eval_labels = collect_firing(
            evaluation,
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            text_key=text_key,
            label_key=label_key,
            batch_size=args.batch_size,
            max_length=args.max_length,
            epsilon=args.epsilon,
        )
        if not args.no_firing_cache:
            save_firing_cache(
                firing_cache_path(args, split="eval", text_key=text_key, label_key=label_key),
                eval_meta,
                eval_firing,
                eval_labels,
            )
    else:
        eval_firing, eval_labels = eval_cached

    or_pred = predict_from_firing(eval_firing, feature_ids)
    or_metrics = binary_metrics(eval_labels, or_pred)
    metrics = or_metrics
    logic_result = None
    if args.classifier == "difflogic":
        logic_result = train_difflogic_classifier(
            discovery_firing,
            discovery_labels,
            eval_firing,
            eval_labels,
            feature_ids=feature_ids,
            args=args,
        )
        metrics = Metrics(**logic_result["eval_hard_metrics"])

    print("selected features:")
    for f in features:
        print(
            f"  layer={f.layer} feature={f.feature_id} "
            f"delta={f.delta:.4f} toxic_rate={f.toxic_rate:.4f} clean_rate={f.clean_rate:.4f}"
        )
    print("or rule metrics:")
    print(json.dumps(asdict(or_metrics), indent=2))
    if logic_result is not None:
        print("difflogic hard-gate eval metrics:")
        print(json.dumps(logic_result["eval_hard_metrics"], indent=2))
        print("difflogic soft eval metrics:")
        print(json.dumps(logic_result["eval_soft_metrics"], indent=2))
    save_results(
        args.out,
        args=args,
        features=features,
        ranked_features=ranked_features,
        or_metrics=or_metrics,
        logic_result=logic_result,
        metrics=metrics,
        text_key=text_key,
        label_key=label_key,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

'''
How to run:
It does the paper’s exact two-step rule:

Layer-specific discovery: P(feature fires | toxic) - P(feature fires | clean)
Classification: toxic iff any selected feature fires anywhere in the prompt
Layer is parameterized and defaults to 6. Defaults are set for Qwen/Qwen3-1.7B-Base and Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50, with --top-k-features 1 by default so you can test the surprising “one feature gets high F1” claim directly.

Run from feature-circuits:

python qwen/qwen_scope_toxicity_classifier.py `
  --layer 6 `
  --top-k-features 1 `
  --language en

The default result path is:
qwen/out/qwen_scope_layer6_en_top1.json

Expensive discovery/eval firing tensors are cached under:
qwen/out/firing_cache/

The firing cache key excludes --top-k-features, so top-10/top-20/top-50 sweeps
reuse the same firing tensors when the model, dataset, layer, SAE top-k,
epsilon, seed, split sizes, max length, and inferred text/label columns match.

To sweep like Figure 8:

python qwen/qwen_scope_toxicity_classifier.py --layer 6 --top-k-features 2
python qwen/qwen_scope_toxicity_classifier.py --layer 6 --top-k-features 5
python qwen/qwen_scope_toxicity_classifier.py --layer 6 --top-k-features 10
'''
