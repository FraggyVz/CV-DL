"""Per-epoch learning curves for one TCN (or pooled control) fit on the shared clip data.

Trains on every training match except --held-out, with the settings saved by the nested
run (scripts/nested_tcn.py) for that match, and records after every epoch:
  - training loss: the training objective (class-weighted, label-smoothed cross-entropy)
    on all training clips, measured in evaluation mode (no dropout);
  - validation loss: plain cross-entropy on the held-out match (no weights, no smoothing);
  - validation macro-F1 and class probabilities on the held-out match.
Only how often tactic_tcn.train_model calls its validate hook changes (every epoch instead
of every eval_every): the hook draws no random numbers and the schedule does not depend
on the epoch count, so training is unchanged. This is checked: at the epoch the nested run
chose, the held-out probabilities must equal the ones that run saved for the same seed.
The curves are a diagnostic; they choose nothing. Uses no test clip: only the dataset's
train/manifest.csv is read, and a clip-data folder whose index lists any clip that is not a
training clip is refused before any clip array is opened.

  python scripts/tcn_curves.py --dataset data/processed/datasets/2026-11-16-train-only \
      --clip-data data/processed/clip-data-2026-11-16-train-only \
      --nested runs/tactic-tcn-clipdata-nested-20261120 \
      --output runs/tactic-tcn-clipdata-curves-20261121 \
      --held-out spirit-falcons-mirage-cologne-major --seed 0 --model tcn
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Run from anywhere in the repo.

from stratscout.config import TACTIC_CLASSES  # noqa: E402
from stratscout.tactic_tcn import (  # noqa: E402
    TacticConfig,
    _eval_features,
    _load_clip_inputs,
    class_weights,
    classification_metrics,
    pad_batch,
    train_model,
)

TOLERANCE = 1e-5  # probabilities saved by the nested run are float32 softmax outputs


def training_rows(dataset: Path) -> list:
    """Rows of <dataset>/train/manifest.csv only, built as tactic_features.load_dataset builds
    them (same fields and order); the test manifest is never opened."""
    manifest = dataset / "train" / "manifest.csv"
    rows = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "train") != "train":
                raise ValueError(f"{manifest} lists a non-training clip: {row['clip_id']}")
            if row["label"] not in TACTIC_CLASSES:
                raise ValueError(f"{manifest}: unknown class {row['label']!r}")
            flag = row.get("boundary_uncertain", "False").strip().lower()
            if flag not in ("true", "false"):
                raise ValueError(f"{manifest}: boundary_uncertain must be True or False")
            rows.append(
                {
                    "clip_id": row["clip_id"],
                    "split": "train",
                    "label": row["label"],
                    "y": TACTIC_CLASSES.index(row["label"]),
                    "match": row["match_folder"],
                    "duration_s": float(row["duration_s"]),
                    "uncertain": flag == "true",
                    "path": dataset / "train" / row["clip"],
                }
            )
    if not rows:
        raise ValueError(f"{manifest} has no clips")
    return rows


def check_training_index(folder: Path, rows: list) -> None:
    """Refuse a clip-data folder whose index.csv lists any non-training clip, before any
    clip array is opened."""
    with (folder / "index.csv").open(encoding="utf-8", newline="") as handle:
        index = list(csv.DictReader(handle))
    training = {r["clip_id"] for r in rows}
    bad = [r["clip_id"] for r in index if r.get("split") != "train" or r["clip_id"] not in training]
    if bad:
        raise ValueError(
            f"{folder} lists {len(bad)} clips that are not training clips (e.g. {bad[0]}); "
            "use a train-only clip-data copy"
        )


def logits_of(model, feats):
    """One clip at a time, no padding, as tactic_tcn.predict does; returns logits."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(*pad_batch([f])) for f in feats])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--clip-data", type=Path, required=True)
    parser.add_argument("--nested", type=Path, required=True, help="nested_tcn.py output folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out", required=True, help="training match left out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="tcn", choices=["tcn", "control"])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new output folder")

    part = json.loads(
        (args.nested / f"{args.model}-{args.held_out}.json").read_text(encoding="utf-8")
    )
    saved = part["config"]
    cfg = TacticConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})
    if cfg.input != "clip-data":
        raise ValueError("The nested run did not use the shared clip data")
    chosen = part["chosen_epochs"]
    saved_probs = next(s["probs"] for s in part["seeds"] if s["seed"] == args.seed)
    curve_cfg = replace(cfg, eval_every=1)  # Only the validate hook's frequency changes.

    training = training_rows(args.dataset)
    check_training_index(args.clip_data, training)  # Uncertain clips are still training clips.
    rows = [r for r in training if not r["uncertain"]]
    dets, info = _load_clip_inputs(args.clip_data, rows, cfg)
    train = [r for r in rows if r["match"] != args.held_out]  # Same order as nested_tcn.py.
    held = [r for r in rows if r["match"] == args.held_out]
    if not held:
        raise ValueError(f"{args.held_out} is not a training match")
    if [r["clip_id"] for r in held] != part["clip_ids"]:
        raise ValueError("Held-out clips differ from the nested run's")
    pairs = [(dets[r["clip_id"]], r["y"]) for r in train]
    train_y = torch.tensor([r["y"] for r in train])
    held_y = torch.tensor([r["y"] for r in held])
    objective = nn.CrossEntropyLoss(
        weight=class_weights([r["y"] for r in train]), label_smoothing=cfg.label_smoothing
    )
    history = []

    def validate(model, norm):
        train_logits = logits_of(
            model, _eval_features([dets[r["clip_id"]] for r in train], norm, cfg)
        )
        held_logits = logits_of(
            model, _eval_features([dets[r["clip_id"]] for r in held], norm, cfg)
        )
        probs = held_logits.softmax(1).numpy()
        f1 = classification_metrics(held_y.tolist(), probs.argmax(1))["macro_f1"]
        history.append(
            {
                "epoch": len(history) + 1,
                "train_loss": float(objective(train_logits, train_y)),
                "validation_loss": float(nn.functional.cross_entropy(held_logits, held_y)),
                "validation_macro_f1": float(f1),
                "probs": probs.round(7).tolist(),
            }
        )
        return f1

    started = time.time()
    print(f"{args.model}: train on {len(train)} clips, hold out {args.held_out} ({len(held)})")
    train_model(args.model, pairs, curve_cfg, args.seed, validate=validate)
    if [h["epoch"] for h in history] != list(range(1, cfg.epochs + 1)):
        raise RuntimeError("Expected one validation per epoch")
    difference = float(np.abs(np.array(history[chosen - 1]["probs"]) - np.array(saved_probs)).max())
    if difference > TOLERANCE:
        raise RuntimeError(
            f"At epoch {chosen} the probabilities differ from the nested run by {difference:.2e}: "
            "training is not the same fit"
        )
    args.output.mkdir(parents=True)
    result = {
        "method": __doc__.strip().splitlines()[0],
        "model": args.model,
        "held_out": args.held_out,
        "seed": args.seed,
        "train_clips": len(train),
        "held_out_clips": [r["clip_id"] for r in held],
        "nested_run": str(args.nested),
        "chosen_epoch": chosen,
        "reproduces_nested_run": {"epoch": chosen, "max_probability_difference": difference},
        "train_loss": "class-weighted, label-smoothed cross-entropy on all training clips, evaluation mode",
        "validation_loss": "plain cross-entropy on the held-out match",
        "config": asdict(cfg),
        "clip_data": info,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    (args.output / "curves.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        f"saved {args.output / 'curves.json'}; epoch {chosen} matches the nested run "
        f"(max difference {difference:.1e}); {result['seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
