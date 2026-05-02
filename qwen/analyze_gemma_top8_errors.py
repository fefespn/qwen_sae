"""Interpretability error analysis for the Gemma top-8 toxic circuit.

Circuit under test (from gemma_scope_layer20_en_top8_difflogic.json):
    toxic = i1 AND ((i4 OR i2) OR i5)

For 5 FN + 5 FP eval examples, print:
  - the raw text and gold label
  - which of i1..i8 fired
  - which top-50 ranked features fired
  - a short diagnosis of why the circuit got it wrong
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import torch

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_scope_toxicity_classifier import load_balanced_toxicity_split  # noqa: E402

DEFAULT_JSON = HERE / "out" / "gemma_scope_layer20_en_top8_difflogic.json"
DEFAULT_FIRING = HERE / "out" / "firing_cache" / "gemma_scope_layer20_en_eval_firing_3a4272ece97d9021.pt"


def short_meaning(meaning: dict | None) -> str:
    if not meaning:
        return "(no Neuronpedia meaning)"
    top = meaning.get("top_explanation")
    if isinstance(top, str) and top:
        return top
    if isinstance(top, dict):
        text = top.get("text")
        if text:
            return text
    explanations = meaning.get("explanations") or []
    if explanations:
        return explanations[0].get("text", "(empty)")
    return "(no explanation)"


def wrap(s: str, width: int = 100) -> str:
    return textwrap.fill(s, width=width, replace_whitespace=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--firing", default=str(DEFAULT_FIRING))
    ap.add_argument("--n-fn", type=int, default=5)
    ap.add_argument("--n-fp", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--discovery-per-class", type=int, default=2000)
    ap.add_argument("--eval-per-class", type=int, default=500)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    result = json.loads(Path(args.json).read_text())

    top_features = result["features"]
    feature_ids = [f["feature_id"] for f in top_features]
    n_inputs = len(feature_ids)

    feature_meanings = result["logic_classifier"]["hard_circuit"]["feature_meanings"]
    binary_expr = result["logic_classifier"]["hard_circuit"]["binary_output"]["expression"]
    ranked = result["ranked_features"]

    print(f"Circuit JSON: {args.json}")
    print(f"Hard circuit: {binary_expr}")
    print(f"Inputs (feature_ids): {feature_ids}")
    print()
    print("Input -> SAE feature -> meaning")
    for i, f in enumerate(top_features, start=1):
        m = feature_meanings.get(f"i{i}")
        print(f"  i{i:<2} = {f['feature_id']:>5}  | delta={f['delta']:+.3f}  toxic_rate={f['toxic_rate']:.3f}  clean_rate={f['clean_rate']:.3f}")
        print(f"          meaning: {short_meaning(m)}")
    print()

    payload = torch.load(args.firing, map_location="cpu", weights_only=False)
    firing_full = payload["firing"].bool()  # [N, n_sae_features]
    labels = payload["labels"].bool()       # [N]
    meta = payload["metadata"]
    print(f"Firing cache: {args.firing}")
    print(f"  N={firing_full.shape[0]}  sae_features={firing_full.shape[1]}  toxic={int(labels.sum())}")
    print()

    # Reload eval texts in the same deterministic order used during training/eval.
    discovery, evaluation, text_key, label_key = load_balanced_toxicity_split(
        dataset_name=meta["dataset"],
        language=args.language,
        seed=args.seed,
        discovery_per_class=args.discovery_per_class,
        eval_per_class=args.eval_per_class,
        text_key=meta["text_key"],
        label_key=meta["label_key"],
    )
    assert len(evaluation) == firing_full.shape[0], \
        f"eval split size mismatch: dataset={len(evaluation)} firing={firing_full.shape[0]}"

    eval_labels_from_text = torch.tensor(
        [int(r[meta["label_key"]]) for r in evaluation], dtype=torch.bool
    )
    if not torch.equal(eval_labels_from_text, labels):
        # Should not happen if seed/per-class match, but guard against silent drift.
        n_match = int((eval_labels_from_text == labels).sum())
        print(f"WARNING: dataset labels differ from cached labels in {len(labels) - n_match} positions")
    eval_texts = [r[meta["text_key"]] for r in evaluation]

    # Slice firing -> [N, k] columns matching i1..ik
    ids = torch.as_tensor(feature_ids, dtype=torch.long)
    firing_topk = firing_full.index_select(1, ids)  # [N, k]

    # Slice firing -> [N, 50] columns matching ranked top-50
    ranked_ids = [f["feature_id"] for f in ranked]
    ranked_idx = torch.as_tensor(ranked_ids, dtype=torch.long)
    firing_ranked = firing_full.index_select(1, ranked_idx)  # [N, 50]

    # Predict: i1 AND (i2 OR i4 OR i5).  Inputs are 0-indexed: i1->0, i2->1, ...
    i1, i2, i4, i5 = firing_topk[:, 0], firing_topk[:, 1], firing_topk[:, 3], firing_topk[:, 4]
    pred = i1 & (i2 | i4 | i5)

    tp = (pred & labels).sum().item()
    fp = (pred & ~labels).sum().item()
    tn = (~pred & ~labels).sum().item()
    fn = (~pred & labels).sum().item()
    print(f"Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print()

    fn_idx = torch.nonzero(labels & ~pred, as_tuple=False).flatten().tolist()
    fp_idx = torch.nonzero(~labels & pred, as_tuple=False).flatten().tolist()
    print(f"FN pool: {len(fn_idx)}  FP pool: {len(fp_idx)}")

    # Take the first n of each (deterministic since the eval split is seeded).
    fn_pick = fn_idx[: args.n_fn]
    fp_pick = fp_idx[: args.n_fp]

    def render(idx_list: list[int], tag: str) -> None:
        print()
        print("=" * 100)
        print(f"{tag}  (n={len(idx_list)})")
        print("=" * 100)
        for n, idx in enumerate(idx_list, start=1):
            text = eval_texts[idx]
            gold = "toxic" if labels[idx].item() else "clean"
            p = "toxic" if pred[idx].item() else "clean"
            row = firing_topk[idx].tolist()  # [k] bools
            on = [f"i{i+1}" for i, v in enumerate(row) if v]
            off = [f"i{i+1}" for i, v in enumerate(row) if not v]

            print(f"\n[{tag} #{n}]  eval_idx={idx}  gold={gold}  pred={p}")
            print("text:")
            print(wrap(text, 100))
            print(f"top-8 ON : {on}")
            print(f"top-8 OFF: {off}")

            # Diagnose
            if tag == "FN":
                # gold toxic, pred clean -> AND broke. Which side?
                if not row[0]:
                    side = "i1 (expletive/strong-emotion gate) is OFF -> AND short-circuits to 0"
                elif not (row[1] or row[3] or row[4]):
                    side = "i1 is ON but none of i2/i4/i5 (negativity/fraud/sexual) fired"
                else:
                    side = "(unexpected: should not be FN given inputs)"
                print(f"why FN  : {side}")
            elif tag == "FP":
                # gold clean, pred toxic -> all required gates fired on a clean text.
                fired_required = [name for name, v in zip(["i2", "i4", "i5"], [row[1], row[3], row[4]]) if v]
                print(f"why FP  : i1=ON AND OR-side fired via {fired_required} despite clean gold label")

            # Also list which top-50 ranked features fired (skipping i1..i8 themselves to keep it readable)
            ranked_row = firing_ranked[idx].tolist()
            extras = []
            for j, on_flag in enumerate(ranked_row, start=1):
                if not on_flag:
                    continue
                fid = ranked[j - 1]["feature_id"]
                exp = (ranked[j - 1].get("neuronpedia") or {}).get("explanations") or []
                meaning = exp[0]["text"] if exp else "(no meaning)"
                marker = "*" if fid in feature_ids else " "
                extras.append(f"  {marker} r{j:<2} fid={fid:>5}  delta={ranked[j-1]['delta']:+.3f}  {meaning}")
            print(f"top-50 ranked features that fired ({len(extras)}; '*' = also in top-8 inputs):")
            for line in extras:
                print(line)

    render(fn_pick, "FN")
    render(fp_pick, "FP")


if __name__ == "__main__":
    main()
