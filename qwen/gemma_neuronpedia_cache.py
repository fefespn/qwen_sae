"""Build a local Neuronpedia feature-meaning cache for Gemma Scope features.

Examples:
    python qwen/gemma_neuronpedia_cache.py --features 13324,2746,7579
    python qwen/gemma_neuronpedia_cache.py --from-result qwen/out/gemma_scope_layer20_en_top8_difflogic.json
    python qwen/gemma_neuronpedia_cache.py --max-feature 16383
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from gemma_scope_toxicity_classifier import (
    DEFAULT_NEURONPEDIA_MODEL,
    DEFAULT_NEURONPEDIA_SOURCE,
    default_neuronpedia_cache_path,
    fetch_neuronpedia_feature,
)


def parse_feature_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def feature_ids_from_result(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    for section in ("features", "ranked_features"):
        for item in payload.get(section, []):
            if "feature_id" in item:
                ids.append(int(item["feature_id"]))
    return ids


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"features": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("features", {})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_NEURONPEDIA_MODEL)
    parser.add_argument("--source", default=DEFAULT_NEURONPEDIA_SOURCE)
    parser.add_argument("--features", default=None, help="Comma-separated feature ids.")
    parser.add_argument("--from-result", type=Path, action="append", default=[])
    parser.add_argument("--max-feature", type=int, default=None, help="Fetch feature ids 0..max-feature inclusive.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out_path = args.out or default_neuronpedia_cache_path(args.model, args.source)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_ids = set(parse_feature_ids(args.features))
    for result_path in args.from_result:
        feature_ids.update(feature_ids_from_result(result_path))
    if args.max_feature is not None:
        feature_ids.update(range(args.max_feature + 1))
    if not feature_ids:
        raise SystemExit("No features requested. Use --features, --from-result, or --max-feature.")

    payload = load_existing(out_path)
    payload["model"] = args.model
    payload["source"] = args.source
    features = payload.setdefault("features", {})

    ordered_ids = sorted(feature_ids)
    for n, feature_id in enumerate(ordered_ids, start=1):
        key = str(feature_id)
        if key in features and not args.refresh:
            print(f"[{n}/{len(ordered_ids)}] cached feature {feature_id}")
            continue
        print(f"[{n}/{len(ordered_ids)}] fetching feature {feature_id}")
        features[key] = fetch_neuronpedia_feature(args.model, args.source, feature_id)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.sleep > 0:
            time.sleep(args.sleep)

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
