"""Nested epoch selection for train-tactics on the shared clip data (TCN and pooled control).

For each training match as the OUTER held-out match:
  1. tactic_tcn.cross_validate on the other three matches only (leave-one-match-out over
     those three, seeds 0-2); its "final_epoch" (best mean inner curve) is the epoch count;
  2. for each seed, tactic_tcn.train_model on those three matches for exactly that many
     epochs (same TacticConfig as train-tactics --clip-data --seeds 3);
  3. predict the outer match and score it.
The outer match never enters the inner folds, the ResNet PCA, the scaling or the class
weights; this is asserted for every fitted model. Uses no test clip.

One JSON per (model, outer match) in --output, saved as it finishes; existing ones are
skipped, so a stopped run can be restarted with the same command. summary.json and
predictions.csv are written once every part exists.

  python scripts/nested_tcn.py --dataset data/processed/datasets/2026-11-16-train-only \
      --clip-data data/processed/clip-data-2026-11-16-train-only \
      --output runs/tactic-tcn-clipdata-nested-20261120 --models tcn control
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Run from anywhere in the repo.

from stratscout.clip_data import ResnetCompressor  # noqa: E402
from stratscout.config import TACTIC_CLASSES  # noqa: E402
from stratscout.tactic_features import load_dataset  # noqa: E402
from stratscout.tactic_tcn import (  # noqa: E402
    Normaliser,
    TacticConfig,
    _eval_features,
    _load_clip_inputs,
    _raw_features,
    class_weights,
    classification_metrics,
    cross_validate,
    predict,
    train_model,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--clip-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["tcn", "control"], choices=["tcn", "control"]
    )
    args = parser.parse_args(argv)

    # The same settings train-tactics builds for --clip-data --seeds 3 (all other options default).
    cfg = TacticConfig(seeds=(0, 1, 2), models=tuple(args.models), input="clip-data", resnet_pcs=32)
    rows = [r for r in load_dataset(args.dataset) if not r["uncertain"]]
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Use the train-only dataset copy: this script must not see test clips")
    dets, info = _load_clip_inputs(args.clip_data, rows, cfg)
    matches = sorted({r["match"] for r in rows})
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} training clips, matches {matches}", flush=True)

    def check_no_leak(norm, inner, outer, ys_inner):
        """The fitted PCA, scaling and class weights equal ones refitted on the inner clips
        alone, and differ from ones refitted with the outer clips added."""
        inner_dets = [dets[r["clip_id"]] for r in inner]
        both_dets = inner_dets + [dets[r["clip_id"]] for r in outer]

        def refit(ds):
            comp = ResnetCompressor(cfg.resnet_pcs).fit([d["resnet"] for d in ds])
            return comp, Normaliser.fit([_raw_features(d, cfg, comp) for d in ds], comp)

        comp_in, norm_in = refit(inner_dets)
        comp_both, norm_both = refit(both_dets)
        assert np.allclose(norm.compressor.mean, comp_in.mean), "PCA mean is not inner-only"
        assert np.allclose(
            np.abs(norm.compressor.components), np.abs(comp_in.components), atol=1e-4
        )
        assert np.allclose(norm.mean, norm_in.mean) and np.allclose(norm.std, norm_in.std), (
            "scaling"
        )
        assert not np.allclose(norm.mean, norm_both.mean), (
            "check has no power: outer clips change nothing"
        )
        weights = class_weights(ys_inner).tolist()
        return {
            "pca_inner_only": True,
            "scaling_inner_only": True,
            "class_weights_from_inner": dict(zip(TACTIC_CLASSES, [round(w, 4) for w in weights])),
        }

    for kind in cfg.models:
        for outer_match in matches:
            path = args.output / f"{kind}-{outer_match}.json"
            if path.exists():
                print(f"{kind} / {outer_match}: already saved, skipped", flush=True)
                continue
            started = time.time()
            inner = [r for r in rows if r["match"] != outer_match]
            outer = [r for r in rows if r["match"] == outer_match]
            assert outer_match not in {r["match"] for r in inner}
            assert not {r["clip_id"] for r in inner} & {r["clip_id"] for r in outer}
            print(f"{kind} / outer {outer_match}: inner CV on {len(inner)} clips", flush=True)
            cv = cross_validate(kind, inner, dets, cfg)
            epochs = cv["final_epoch"]
            inner_matches = sorted({r["held_out"] for r in cv["runs"]})
            assert outer_match not in inner_matches
            pairs = [(dets[r["clip_id"]], r["y"]) for r in inner]
            seeds, leak = [], None
            for seed in cfg.seeds:
                model, norm, _ = train_model(kind, pairs, cfg, seed, epochs=epochs)
                if leak is None:  # PCA, scaling and weights do not depend on the seed.
                    leak = check_no_leak(norm, inner, outer, [r["y"] for r in inner])
                probs = predict(
                    model, _eval_features([dets[r["clip_id"]] for r in outer], norm, cfg)
                )
                metrics = classification_metrics([r["y"] for r in outer], probs.argmax(1))
                seeds.append(
                    {
                        "seed": seed,
                        "macro_f1": metrics["macro_f1"],
                        "accuracy": metrics["accuracy"],
                        "probs": probs.tolist(),
                    }
                )
                print(
                    f"  seed {seed}: {epochs} epochs, macro-F1 {metrics['macro_f1']:.3f}",
                    flush=True,
                )
            result = {
                "model": kind,
                "outer_match": outer_match,
                "inner_matches": inner_matches,
                "inner_curve": cv["curve"],
                "inner_optimistic_macro_f1": cv["optimistic_macro_f1"],
                "chosen_epochs": epochs,
                "clip_ids": [r["clip_id"] for r in outer],
                "y": [r["y"] for r in outer],
                "seeds": seeds,
                "leak_check": leak,
                "config": asdict(cfg),
                "clip_data": info,
                "seconds": round(time.time() - started, 1),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            tmp.replace(path)
            print(f"{kind} / {outer_match}: saved in {result['seconds']:.0f}s", flush=True)

    # Summary once every part exists.
    summary, prediction_rows = {"method": __doc__.strip().splitlines()[0], "models": {}}, []
    for kind in cfg.models:
        parts = [
            json.loads((args.output / f"{kind}-{m}.json").read_text(encoding="utf-8"))
            for m in matches
        ]
        per_seed = [
            np.mean([p["seeds"][k]["macro_f1"] for p in parts]) for k in range(len(cfg.seeds))
        ]
        y = np.concatenate([p["y"] for p in parts])
        mean_probs = np.concatenate([np.mean([s["probs"] for s in p["seeds"]], 0) for p in parts])
        ensemble = classification_metrics(y, mean_probs.argmax(1))
        summary["models"][kind] = {
            "nested_macro_f1_mean_over_seeds": float(np.mean(per_seed)),
            "nested_macro_f1_seed_sd": float(np.std(per_seed)),
            "per_seed": [float(v) for v in per_seed],
            "ensemble_macro_f1": ensemble["macro_f1"],
            "ensemble_accuracy": ensemble["accuracy"],
            "ensemble_per_class_f1": {c: v["f1"] for c, v in ensemble["per_class"].items()},
            "ensemble_confusion": ensemble["confusion"],
            "per_match_macro_f1": {
                p["outer_match"]: float(np.mean([s["macro_f1"] for s in p["seeds"]])) for p in parts
            },
            "chosen_epochs": {p["outer_match"]: p["chosen_epochs"] for p in parts},
            "clips": int(len(y)),
            "seconds": sum(p["seconds"] for p in parts),
        }
        for p in parts:
            probs = np.mean([s["probs"] for s in p["seeds"]], 0)
            for clip_id, truth, pr in zip(p["clip_ids"], p["y"], probs):
                prediction_rows.append(
                    {
                        "model": kind,
                        "clip_id": clip_id,
                        "match": p["outer_match"],
                        "true": TACTIC_CLASSES[truth],
                        "predicted": TACTIC_CLASSES[int(pr.argmax())],
                        **{f"p_{c}": f"{v:.6f}" for c, v in zip(TACTIC_CLASSES, pr)},
                    }
                )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    for kind, s in summary["models"].items():
        print(
            f"{kind}: nested {s['nested_macro_f1_mean_over_seeds']:.3f} +/- {s['nested_macro_f1_seed_sd']:.3f}, "
            f"ensemble {s['ensemble_macro_f1']:.3f} (accuracy {s['ensemble_accuracy']:.3f}), epochs {s['chosen_epochs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
