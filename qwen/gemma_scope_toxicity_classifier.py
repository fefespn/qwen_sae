"""Gemma Scope SAE toxicity classifier with Neuronpedia feature meanings.

This mirrors qwen_scope_toxicity_classifier.py, but uses Gemma Scope residual
SAEs loaded through SAELens and enriches selected features with Neuronpedia
links/explanations when available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

# Avoid optional vision backends during text-only Gemma loading. In this venv,
# torchvision is present but ABI-mismatched with torch, and Transformers imports
# it through image_utils unless these flags are set before importing transformers.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")

from qwen_scope_toxicity_classifier import (
    Metrics,
    ToxicFeature,
    binary_metrics,
    discover_top_features_from_firing,
    load_balanced_toxicity_split,
    load_firing_cache,
    predict_from_firing,
    save_firing_cache,
    train_difflogic_classifier,
)


DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
DEFAULT_SAE_ID = "layer_20/width_16k/canonical"
DEFAULT_NEURONPEDIA_MODEL = "gemma-2-2b"
DEFAULT_NEURONPEDIA_SOURCE = "20-gemmascope-res-16k"
DEFAULT_DATASET = "textdetox/multilingual_toxicity_dataset"
DEFAULT_LAYER = 20
DEFAULT_TOP_K = 8
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
SUPPORTED_CUDA_CAPABILITIES = {
    (7, 0),
    (7, 5),
    (8, 0),
    (8, 6),
    (9, 0),
    (10, 0),
    (12, 0),
}


def run_name(*, layer: int, language: str, top_k_features: int) -> str:
    return f"gemma_scope_layer{layer}_{language}_top{top_k_features}"


def default_output_path(args: argparse.Namespace) -> str:
    suffix = "" if args.classifier == "or" else f"_{args.classifier}"
    return str(Path(args.out_dir) / f"{run_name(layer=args.layer, language=args.language, top_k_features=args.top_k_features)}{suffix}.json")


def firing_cache_metadata(args: argparse.Namespace, *, split: str, text_key: str, label_key: str) -> dict[str, Any]:
    return {
        "version": 1,
        "split": split,
        "model": args.model,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "dataset": args.dataset,
        "language": args.language,
        "layer": args.layer,
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
    name = f"gemma_scope_layer{args.layer}_{args.language}_{split}_firing_{digest}.pt"
    return Path(args.firing_cache_dir) / name


class GemmaScopeSAE:
    """Small SAELens wrapper for Gemma Scope residual SAEs."""

    def __init__(
        self,
        *,
        release: str = DEFAULT_SAE_RELEASE,
        sae_id: str = DEFAULT_SAE_ID,
        layer: int = DEFAULT_LAYER,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            from sae_lens import SAE
        except ImportError as e:
            raise ImportError(
                "Gemma Scope SAEs are loaded through SAELens. Install it with `pip install sae-lens` "
                "inside the environment used to run this script."
            ) from e

        self.release = release
        self.sae_id = sae_id
        self.layer = int(layer)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=str(self.device),
        )
        self.sae.eval()
        cfg = getattr(self.sae, "cfg", None)
        self.width = int(
            getattr(cfg, "d_sae", 0)
            or self.cfg_dict.get("d_sae", 0)
            or getattr(self.sae, "W_enc").shape[-1]
        )

    @torch.no_grad()
    def firing_from_hidden(
        self,
        hidden: torch.Tensor,
        epsilon: float = 0.0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=torch.float32)
        acts = self.sae.encode(hidden)
        if isinstance(acts, tuple):
            acts = acts[0]
        active = acts > epsilon
        if attention_mask is not None:
            mask = attention_mask.to(self.device, dtype=torch.bool).unsqueeze(-1)
            active = active & mask
        return active.any(dim=1).to(torch.bool)


def capture_layer_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int) -> torch.Tensor:
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
    sae: GemmaScopeSAE,
    text_key: str,
    label_key: str,
    batch_size: int,
    max_length: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_device = next(model.parameters()).device
    all_firing = []
    all_labels = []
    for batch in tqdm(list(iter_batches(examples, batch_size)), desc="Gemma Scope firing"):
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


def fetch_neuronpedia_feature(model_id: str, source: str, feature_id: int, *, timeout: float = 10.0) -> dict[str, Any]:
    import requests

    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source}/{feature_id}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {
            "feature_id": int(feature_id),
            "neuronpedia_url": f"https://www.neuronpedia.org/{model_id}/{source}/{feature_id}",
            "error": str(e),
        }

    explanations = data.get("explanations") or []
    parsed_explanations = []
    for item in explanations:
        if not isinstance(item, dict):
            continue
        text = item.get("description") or item.get("explanation") or item.get("text") or item.get("title")
        if text:
            parsed_explanations.append(
                {
                    "text": str(text),
                    "type": item.get("typeName") or item.get("type") or item.get("model"),
                    "score": item.get("score"),
                }
            )
    return {
        "feature_id": int(feature_id),
        "neuronpedia_url": f"https://www.neuronpedia.org/{model_id}/{source}/{feature_id}",
        "explanations": parsed_explanations,
    }


def default_neuronpedia_cache_path(model_id: str, source: str) -> Path:
    safe_model = model_id.replace("/", "__")
    safe_source = source.replace("/", "__")
    return DEFAULT_OUT_DIR / "neuronpedia_cache" / f"{safe_model}__{safe_source}.json"


def load_neuronpedia_cache(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    features = payload.get("features", payload)
    return {int(k): v for k, v in features.items()}


def top_neuronpedia_explanation(neuronpedia: dict[str, Any] | None) -> str | None:
    if not neuronpedia:
        return None
    explanations = neuronpedia.get("explanations") or []
    for explanation in explanations:
        if isinstance(explanation, dict) and explanation.get("text"):
            return str(explanation["text"])
    return None


def fallback_neuronpedia_entry(model_id: str, source: str, feature_id: int) -> dict[str, Any]:
    return {
        "feature_id": int(feature_id),
        "neuronpedia_url": f"https://www.neuronpedia.org/{model_id}/{source}/{feature_id}",
    }


def enrich_features_with_neuronpedia(
    features: Sequence[ToxicFeature],
    *,
    model_id: str,
    source: str,
    enabled: bool,
    cache: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    enriched = []
    cache = cache or {}
    for feature in features:
        item = asdict(feature)
        cached = cache.get(feature.feature_id)
        item["neuronpedia"] = (
            cached
            if cached is not None
            else fetch_neuronpedia_feature(model_id, source, feature.feature_id)
            if enabled
            else fallback_neuronpedia_entry(model_id, source, feature.feature_id)
        )
        enriched.append(item)
    return enriched


def feature_meaning_lookup(
    features: Sequence[ToxicFeature],
    *,
    model_id: str,
    source: str,
    cache: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    lookup = {}
    for feature in features:
        entry = cache.get(feature.feature_id) or fallback_neuronpedia_entry(model_id, source, feature.feature_id)
        lookup[feature.feature_id] = {
            "feature_id": feature.feature_id,
            "neuronpedia_url": entry.get("neuronpedia_url")
            or f"https://www.neuronpedia.org/{model_id}/{source}/{feature.feature_id}",
            "explanations": entry.get("explanations", []),
            "top_explanation": top_neuronpedia_explanation(entry),
        }
    return lookup


def attach_feature_meanings_to_logic_result(
    logic_result: dict[str, Any] | None,
    *,
    features: Sequence[ToxicFeature],
    model_id: str,
    source: str,
    cache: dict[int, dict[str, Any]],
) -> None:
    if logic_result is None:
        return
    circuit = logic_result.get("hard_circuit")
    if not isinstance(circuit, dict):
        return
    meanings = feature_meaning_lookup(features, model_id=model_id, source=source, cache=cache)
    by_input_name = {}
    for index, feature in enumerate(features, start=1):
        by_input_name[f"i{index}"] = meanings[feature.feature_id]
    for input_item in circuit.get("inputs", []):
        if isinstance(input_item, dict):
            meaning = by_input_name.get(input_item.get("name"))
            if meaning:
                input_item["meaning"] = meaning
    circuit["feature_meanings"] = by_input_name


def save_results(
    path: str,
    *,
    args: argparse.Namespace,
    features: Sequence[ToxicFeature],
    ranked_features: Sequence[ToxicFeature],
    metrics: Metrics,
    or_metrics: Metrics,
    logic_result: dict[str, Any] | None,
    text_key: str,
    label_key: str,
) -> None:
    neuronpedia_cache = load_neuronpedia_cache(args.neuronpedia_cache)
    payload = {
        "method": "gemma_scope_sae_toxicity_classifier",
        "model": args.model,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "neuronpedia_model": args.neuronpedia_model,
        "neuronpedia_source": args.neuronpedia_source,
        "neuronpedia_cache": args.neuronpedia_cache,
        "layer": args.layer,
        "top_k_features": args.top_k_features,
        "epsilon": args.epsilon,
        "language": args.language,
        "dataset": args.dataset,
        "text_key": text_key,
        "label_key": label_key,
        "classifier": args.classifier,
        "features": enrich_features_with_neuronpedia(
            features,
            model_id=args.neuronpedia_model,
            source=args.neuronpedia_source,
            enabled=not args.no_neuronpedia,
            cache=neuronpedia_cache,
        ),
        "ranked_features": enrich_features_with_neuronpedia(
            ranked_features,
            model_id=args.neuronpedia_model,
            source=args.neuronpedia_source,
            enabled=False,
            cache=neuronpedia_cache,
        ),
        "ranked_features_top_k": len(ranked_features),
        "or_metrics": asdict(or_metrics),
        "logic_classifier": logic_result,
        "metrics": asdict(metrics),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def disable_broken_torchvision_import() -> None:
    """Tell Transformers not to import torchvision for this text-only script.

    Some environments have a torchvision wheel that is ABI-incompatible with
    torch. Transformers only needs torchvision for optional vision utilities,
    but Gemma2 model import can still pass through image_utils. Marking
    torchvision unavailable avoids failing on an optional dependency.
    """
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    import_utils.is_torchvision_available = lambda: False
    import_utils.is_torchvision_v2_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_v2_available = lambda: False


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_CUDA_CAPABILITIES:
        name = torch.cuda.get_device_name()
        supported = ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_CUDA_CAPABILITIES))
        print(
            f"CUDA device {name!r} has compute capability {capability[0]}.{capability[1]}, "
            f"but this PyTorch build supports only {supported}. Falling back to CPU. "
            "Pass --device cuda to force CUDA after installing a compatible torch build."
        )
        return torch.device("cpu")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument("--neuronpedia-model", default=DEFAULT_NEURONPEDIA_MODEL)
    parser.add_argument("--neuronpedia-source", default=DEFAULT_NEURONPEDIA_SOURCE)
    parser.add_argument("--neuronpedia-cache", default=None)
    parser.add_argument("--no-neuronpedia", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--language", default="en")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--top-k-features", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--discovery-per-class", type=int, default=2000)
    parser.add_argument("--eval-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--firing-cache-dir", default=str(DEFAULT_OUT_DIR / "firing_cache"))
    parser.add_argument("--no-firing-cache", action="store_true")
    parser.add_argument("--save-ranked-top-k", type=int, default=50)
    parser.add_argument("--classifier", choices=("or", "difflogic"), default="or")
    parser.add_argument("--difflogic-path", default=str(Path(__file__).resolve().parents[1] / "difflogic"))
    parser.add_argument("--logic-output", choices=("binary", "groupsum"), default="binary")
    parser.add_argument("--logic-dims", default=None)
    parser.add_argument("--logic-epochs", type=int, default=100)
    parser.add_argument("--logic-batch-size", type=int, default=256)
    parser.add_argument("--logic-lr", type=float, default=0.01)
    parser.add_argument("--logic-tau", type=float, default=1.0)
    parser.add_argument("--logic-threshold", type=float, default=0.5)
    parser.add_argument("--logic-grad-factor", type=float, default=1.0)
    parser.add_argument("--logic-connections", choices=("random", "unique"), default="random")
    parser.add_argument("--logic-implementation", choices=("auto", "python", "cuda"), default="python")
    parser.add_argument("--logic-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--logic-seed", type=int, default=0)
    parser.add_argument("--logic-print-every", type=int, default=25)
    args = parser.parse_args()

    if args.neuronpedia_cache is None:
        args.neuronpedia_cache = str(default_neuronpedia_cache_path(args.neuronpedia_model, args.neuronpedia_source))
    if args.out is None:
        args.out = default_output_path(args)

    discovery, evaluation, text_key, label_key = load_balanced_toxicity_split(
        dataset_name=args.dataset,
        language=args.language,
        seed=args.seed,
        discovery_per_class=args.discovery_per_class,
        eval_per_class=args.eval_per_class,
        text_key=args.text_key,
        label_key=args.label_key,
    )

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
        disable_broken_torchvision_import()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = choose_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
        model.eval()
        sae = GemmaScopeSAE(
            release=args.sae_release,
            sae_id=args.sae_id,
            layer=args.layer,
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
    neuronpedia_cache = load_neuronpedia_cache(args.neuronpedia_cache)
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
        attach_feature_meanings_to_logic_result(
            logic_result,
            features=features,
            model_id=args.neuronpedia_model,
            source=args.neuronpedia_source,
            cache=neuronpedia_cache,
        )

    print("selected features:")
    for f in features:
        meaning = feature_meaning_lookup(
            [f],
            model_id=args.neuronpedia_model,
            source=args.neuronpedia_source,
            cache=neuronpedia_cache,
        )[f.feature_id]
        explanation = meaning.get("top_explanation") or "(no cached explanation)"
        print(
            f"  layer={f.layer} feature={f.feature_id} "
            f"delta={f.delta:.4f} toxic_rate={f.toxic_rate:.4f} clean_rate={f.clean_rate:.4f} "
            f"meaning={explanation!r} {meaning['neuronpedia_url']}"
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
        metrics=metrics,
        or_metrics=or_metrics,
        logic_result=logic_result,
        text_key=text_key,
        label_key=label_key,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
