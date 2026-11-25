"""The final models on the test split, as the web app runs them (run once, after selection).

Trains, tunes and changes nothing:
  1. reads the test manifest (rows of split "test" only) and its clips;
  2. computes detections and clip features once, exactly as predict_clips does without a
     saved background: one empty-map background per match, from that match's clips. These
     are all of the match's test clips, the grouping the detection cache used for the TCN's
     saved test run;
  3. TCN check: predict_clips(models=("tcn",)) on those features must reproduce the saved
     test predictions of runs/tactic-tcn-clipdata-final (same predicted class, probabilities
     within one 4-decimal rounding step), or the script stops and writes nothing;
  4. GRU: predict_clips(models=("gru",)) on the same features, and each seed's probabilities
     on their own (their mean must equal predict_clips's ensemble);
  5. writes a new folder: predictions.csv (each GRU seed and the ensemble), app-results.json
     (predict_clips output for both models, as the app shows it), report.json, report.md.

Scores use tactic_tcn.classification_metrics, the function behind the TCN's saved test
report: macro-F1 pooled over all test clips per seed (mean and SD over seeds), the seed
ensemble, and, for comparison with cross-validation only, the ensemble per test match.

  python scripts/final_test_gru.py --dataset data/processed/datasets/2026-11-16 \
      --output runs/tactic-gru-clipdata-test-20261121
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Run from anywhere in the repo.

from stratscout import gru_predict  # noqa: E402
from stratscout.config import TACTIC_CLASSES  # noqa: E402
from stratscout.storage import write_new_text  # noqa: E402
from stratscout.tactic_tcn import _write_csv, classification_metrics  # noqa: E402

ROUNDING = 1e-4 + 1e-9  # one step of the 4-decimal rounding both prediction files use


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_rows(dataset: Path):
    manifest = dataset / "test" / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError(f"{manifest}: expected test rows only")
    paths = [dataset / "test" / r["clip"] for r in rows]
    for row, path in zip(rows, paths):
        if not path.is_file() or path.stem != row["clip_id"]:
            raise ValueError(f"{row['clip_id']}: clip file {path} missing or misnamed")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("clip names are not unique")
    return manifest, rows, paths


def tcn_check(tcn_results, saved_csv: Path):
    """The fresh TCN predictions against the saved test predictions, clip by clip."""
    with saved_csv.open(encoding="utf-8", newline="") as stream:
        saved = {r["clip_id"]: r for r in csv.DictReader(stream) if r["model"] == "tcn"}
    fresh = {r["clip"]: r["predictions"]["tcn"] for r in tcn_results}
    if set(fresh) != set(saved):
        raise SystemExit("TCN check failed: the test clips differ from the saved predictions")
    same_class = sum(fresh[c]["tactic"] == saved[c]["predicted"] for c in saved)
    diff = max(
        abs(fresh[c]["probabilities"][k] - float(saved[c][f"p_{k}"]))
        for c in saved
        for k in TACTIC_CLASSES
    )
    check = {
        "saved_predictions": str(saved_csv),
        "clips": len(saved),
        "same_predicted_class": same_class,
        "max_abs_probability_difference": diff,
        "tolerance": ROUNDING,
    }
    if same_class != len(saved) or diff > ROUNDING:
        raise SystemExit(f"TCN check failed, nothing written: {json.dumps(check)}")
    check["passed"] = True
    return check


def scores(y, seed_probs, ensemble, matches):
    """Per seed (pooled over all clips), the seed ensemble, and the ensemble per match."""
    seed_metrics = {s: classification_metrics(y, p.argmax(1)) for s, p in seed_probs.items()}
    seed_f1 = [seed_metrics[s]["macro_f1"] for s in seed_probs]
    y, matches = np.asarray(y), np.asarray(matches)
    per_match = {}
    for m in sorted(set(matches)):
        keep = matches == m
        per_match[m] = {
            "ensemble": classification_metrics(y[keep], ensemble[keep].argmax(1))["macro_f1"],
            "per_seed": {
                str(s): classification_metrics(y[keep], p[keep].argmax(1))["macro_f1"]
                for s, p in seed_probs.items()
            },
        }
    return {
        "seed_macro_f1": seed_f1,
        "seed_macro_f1_mean": float(np.mean(seed_f1)),
        "seed_macro_f1_std": float(np.std(seed_f1)),
        "seed_metrics": {str(s): m for s, m in seed_metrics.items()},
        "ensemble": classification_metrics(y, ensemble.argmax(1)),
        "per_match": per_match,
        "per_match_mean": {
            "ensemble": float(np.mean([v["ensemble"] for v in per_match.values()])),
            "per_seed_mean": float(
                np.mean(
                    [
                        np.mean([v["per_seed"][str(s)] for v in per_match.values()])
                        for s in seed_probs
                    ]
                )
            ),
        },
    }


def render(report) -> str:
    g, t = report["gru"]["test"], report["tcn_check"]
    lines = [
        "# Final Compact pooled GRU on the test split",
        "",
        f"Test clips: {report['clips']} from {len(report['matches'])} matches "
        f"({', '.join(f'{m}: {n}' for m, n in report['matches'].items())}). The GRU was run once, "
        "after model selection; nothing was trained or tuned. Backgrounds: one per match, from "
        "that match's test clips, as predict_clips builds them.",
        "",
        f"TCN check: predict_clips reproduced the saved TCN test predictions of "
        f"`{report['tcn_check']['saved_predictions']}` ({t['same_predicted_class']}/{t['clips']} "
        f"same class, largest probability difference {t['max_abs_probability_difference']:.4f}).",
        "",
        "| GRU | Macro-F1 |",
        "|---|---|",
        f"| Pooled over all clips, mean ± SD over seeds | {g['seed_macro_f1_mean']:.3f} ± "
        f"{g['seed_macro_f1_std']:.3f} |",
        f"| Seed ensemble (accuracy {g['ensemble']['accuracy']:.3f}) | {g['ensemble']['macro_f1']:.3f} |",
        f"| Seed ensemble, per match then averaged (for comparison with cross-validation) | "
        f"{g['per_match_mean']['ensemble']:.3f} |",
        f"| Majority class ({report['majority']['class']}) | {report['majority']['macro_f1']:.3f} |",
        "",
        "Per-seed and ensemble predictions: `predictions.csv`; the app's output for both models: "
        "`app-results.json`. Analysis: `notebooks/tactic-final-test.ipynb`.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/datasets/2026-11-16"))
    ap.add_argument("--tcn-run", type=Path, default=Path("runs/tactic-tcn-clipdata-final"))
    ap.add_argument("--gru-models", type=Path, default=gru_predict.GRU_MODELS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; choose a new --output folder")
    started = time.perf_counter()
    manifest, rows, paths = test_rows(args.dataset)
    print(f"{len(paths)} test clips from {manifest}")

    tcn_models = args.tcn_run / "models"
    tcn_members = gru_predict.load_models("tcn", tcn_models)
    gru_members = gru_predict.load_models("gru", args.gru_models)
    step = tcn_members[0][2].step
    if gru_members[0][1]["settings"]["step"] != step:
        raise SystemExit("the TCN and GRU read rows at different steps")

    # Detections and clip features once, as predict_clips computes them (no saved background).
    dets, warnings = gru_predict.detect(paths, log=print)
    features = gru_predict.clip_features(paths, dets, step=step, log=print)
    names = [p.stem for p in paths]
    if list(features) != names:
        raise SystemExit("clip features are not in manifest order")
    seconds = {"features": time.perf_counter() - started}

    tcn_results = gru_predict.predict_clips(
        paths, models=("tcn",), tcn_models=tcn_models, features=features, log=print
    )
    check = tcn_check(tcn_results, args.tcn_run / "predictions.csv")
    print(f"TCN check passed: {json.dumps(check)}")

    started = time.perf_counter()
    gru_results = gru_predict.predict_clips(
        paths, models=("gru",), gru_models=args.gru_models, features=features, log=print
    )
    feats = [features[n] for n in names]
    seed_probs = {
        info["seed"]: gru_predict.gru_probs([(h, info)], feats) for h, info in gru_members
    }
    ensemble = np.mean(list(seed_probs.values()), 0)
    for result, p in zip(gru_results, ensemble):
        saved = result["predictions"]["gru"]["probabilities"]
        if any(saved[c] != round(float(v), 4) for c, v in zip(TACTIC_CLASSES, p)):
            raise SystemExit(f"{result['clip']}: per-seed mean differs from predict_clips")
    seconds["gru"] = time.perf_counter() - started

    by_name = {r["clip_id"]: r for r in rows}
    y = [TACTIC_CLASSES.index(by_name[n]["label"]) for n in names]
    matches = [by_name[n]["match_folder"] for n in names]
    gru_scores = scores(y, seed_probs, ensemble, matches)
    tcn_report = json.loads((args.tcn_run / "report.json").read_text(encoding="utf-8"))
    majority = tcn_report["majority"]["class"]
    counts = {
        m: {
            c: sum(1 for n, mm in zip(names, matches) if mm == m and by_name[n]["label"] == c)
            for c in TACTIC_CLASSES
        }
        for m in sorted(set(matches))
    }

    app = []
    for path, t, g in zip(paths, tcn_results, gru_results):
        if t["clip"] != g["clip"] or t["clip"] != path.stem:
            raise SystemExit("TCN and GRU results are not in the same clip order")
        app.append(
            {
                "clip": path.stem,
                "file": str(path),
                "warnings": warnings[path.stem],
                "predictions": {**t["predictions"], **g["predictions"]},
            }
        )

    csv_rows = []
    for label, probs in [*((str(s), p) for s, p in seed_probs.items()), ("ensemble", ensemble)]:
        for n, m, t, p in zip(names, matches, y, probs):
            csv_rows.append(
                {
                    "model": "gru",
                    "seed": label,
                    "clip_id": n,
                    "match": m,
                    "true": TACTIC_CLASSES[t],
                    "predicted": TACTIC_CLASSES[int(p.argmax())],
                    **{f"p_{c}": float(v) for c, v in zip(TACTIC_CLASSES, p)},
                }
            )

    import torch

    report = {
        "purpose": "Final test of the selected TCN (reproduction check) and the final Compact "
        "pooled GRU, run once after model selection; no training, tuning or model change.",
        "test_manifest": str(manifest),
        "test_manifest_sha256": sha256(manifest),
        "clips": len(names),
        "matches": {m: sum(1 for x in matches if x == m) for m in sorted(set(matches))},
        "clips_per_match_and_class": counts,
        "background": "one per match, from that match's test clips (predict_clips default)",
        "warnings": {n: w for n, w in warnings.items() if w},
        "tcn_check": {**check, "models": str(tcn_models)},
        "gru": {
            "models": str(args.gru_models),
            "checkpoint_sha256": {Path(i["path"]).name: sha256(i["path"]) for _, i in gru_members},
            "seeds": list(seed_probs),
            "epochs": gru_members[0][1]["epochs"],
            "settings": gru_predict._plain(gru_members[0][1]["settings"]),
            "regions_version": gru_members[0][1]["regions_version"],
            "test": gru_scores,
        },
        "majority": {
            "class": majority,
            "macro_f1": classification_metrics(y, [TACTIC_CLASSES.index(majority)] * len(y))[
                "macro_f1"
            ],
        },
        "seconds": seconds,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.output.mkdir(parents=True)
    _write_csv(args.output / "predictions.csv", csv_rows)
    write_new_text(args.output / "app-results.json", json.dumps(app, indent=2) + "\n")
    write_new_text(args.output / "report.json", json.dumps(report, indent=2) + "\n")
    write_new_text(args.output / "report.md", render(report))
    print(render(report))


if __name__ == "__main__":
    main()
