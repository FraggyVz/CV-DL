"""Shuffled-label refits for train-tactics (TCN and time-blind control), same protocol.

Labels are shuffled within each training match with numpy default_rng(permutation), then
tactic_tcn.cross_validate runs exactly as train-tactics does (same TacticConfig as the CLI
with --clip-data, --seeds 3) and scores against the shuffled labels. Uses no test clip.

One JSON file per permutation in --output; existing ones are skipped, so a later run with
--start 5 --count 15 adds permutations 5-19 to make 20.

  python scripts/shuffle_tcn.py --dataset data/processed/datasets/2026-11-16-train-only \
      --clip-data data/processed/clip-data-2026-11-16-train-only \
      --output runs/tactic-tcn-clipdata-shuffled-20261119 --start 0 --count 5

Add --models tcn to refit only the TCN (default: tcn and control). Use the same --models
for every batch of one output folder, so all permutations cover the same models.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Run from anywhere in the repo.

from stratscout.config import TACTIC_CLASSES  # noqa: E402
from stratscout.tactic_features import load_dataset  # noqa: E402
from stratscout.tactic_tcn import TacticConfig, _load_clip_inputs, cross_validate  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--clip-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tcn", "control"],
        choices=["tcn", "control"],
        help="Models to refit on each shuffle, e.g. --models tcn",
    )
    args = parser.parse_args(argv)

    # The same settings train-tactics builds for --clip-data --seeds 3 (all other options default).
    cfg = TacticConfig(seeds=(0, 1, 2), models=tuple(args.models), input="clip-data", resnet_pcs=32)
    rows = [r for r in load_dataset(args.dataset) if not r["uncertain"]]
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Use the train-only dataset copy: this script must not see test clips")
    dets, info = _load_clip_inputs(args.clip_data, rows, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    matches = sorted({r["match"] for r in rows})
    print(f"{len(rows)} training clips, matches {matches}", flush=True)

    for permutation in range(args.start, args.start + args.count):
        path = args.output / f"permutation-{permutation:02d}.json"
        if path.exists():
            print(f"permutation {permutation}: already saved, skipped", flush=True)
            continue
        started = time.time()
        generator = np.random.default_rng(permutation)
        y = np.array([r["y"] for r in rows])
        shuffled = y.copy()
        for match in matches:  # Every match keeps its class counts.
            index = np.array([i for i, r in enumerate(rows) if r["match"] == match])
            shuffled[index] = y[generator.permutation(index)]
        shuffled_rows = [
            {**r, "y": int(s), "label": TACTIC_CLASSES[int(s)]} for r, s in zip(rows, shuffled)
        ]
        result = {
            "permutation": permutation,
            "method": "labels permuted within each training match with numpy default_rng(permutation); "
            "tactic_tcn.cross_validate as in train-tactics; scored against the shuffled labels",
            "config": asdict(cfg),
            "clip_data": info,
            "clips": len(rows),
            "changed_labels": int((shuffled != y).sum()),
            "models": {},
        }
        for kind in cfg.models:
            cv = cross_validate(kind, shuffled_rows, dets, cfg)
            result["models"][kind] = {
                "cross_fold_macro_f1": cv["cross_fold_macro_f1"],
                "ensemble_macro_f1": cv["pooled"]["ensemble"]["macro_f1"],
                "per_fold_macro_f1": cv["per_fold_macro_f1"],
                "cross_fold_epochs": cv["cross_fold_epochs"],
            }
        result["seconds"] = round(time.time() - started, 1)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(
            f"permutation {permutation}: "
            + ", ".join(
                f"{k} {v['cross_fold_macro_f1']:.3f} (ensemble {v['ensemble_macro_f1']:.3f})"
                for k, v in result["models"].items()
            )
            + f" in {result['seconds']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
