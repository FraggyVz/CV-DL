"""Whole-match development runs for Plan A. This workflow never opens the test split."""

from __future__ import annotations

import csv
import io
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .context import RunContext
from .storage import save_checkpoint, write_new_text
from .tactic_data import TacticClipDataset, whole_match_folds
from .tactic_model import RGB_MEAN, RGB_STD, WEIGHTS, build_baseline, file_sha256, normalise_frames


@dataclass(frozen=True)
class BaselineConfig:
    size: int = 224
    n_frames: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 200
    patience: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    fixed_epochs: int | None = None

    def __post_init__(self):
        for name in ("size", "n_frames", "epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.size < 32:
            raise ValueError("ResNet-18 requires size >= 32 for this workflow")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct seeds, at least one")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("Seeds must be integers in [0, 2**32)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.fixed_epochs is not None and (
            type(self.fixed_epochs) is not int or self.fixed_epochs < 1
        ):
            raise ValueError("fixed_epochs must be a positive integer")

    @property
    def epoch_mode(self):
        return "fixed" if self.fixed_epochs is not None else "cross-fold"


def class_counts(targets, classes) -> dict[str, int]:
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=len(classes))
    return dict(zip(classes, map(int, counts), strict=True))


def training_class_weights(targets: torch.Tensor, classes) -> torch.Tensor:
    """Balanced weights N/(K*n_class), calculated from this fold's training targets only."""
    counts = torch.tensor(list(class_counts(targets.cpu().numpy(), classes).values()))
    if (counts == 0).any():
        missing = [name for name, count in zip(classes, counts, strict=True) if count == 0]
        raise ValueError(
            f"Training fold has no examples for {missing}; cannot fit all five classes"
        )
    return (len(targets) / (len(classes) * counts.float())).to(targets.device)


def classification_metrics(targets, predictions, classes) -> dict:
    """Clip classification metrics; undefined precision/recall/F1 are reported as zero."""
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or predictions.shape != targets.shape or not len(targets):
        raise ValueError("Targets and predictions must be non-empty matching vectors")
    for values in (targets, predictions):
        if not np.issubdtype(values.dtype, np.integer) or np.any(
            (values < 0) | (values >= len(classes))
        ):
            raise ValueError("Class indices are outside the class order")
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    per_class = {}
    for i, name in enumerate(classes):
        true_positive = int(matrix[i, i])
        support, predicted = int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (support + predicted) if support + predicted else 0.0
        per_class[name] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {
        "accuracy": float(np.trace(matrix) / len(targets)),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),  # Rows: truth; columns: prediction; classes order.
    }


def fit_classifier(features, targets, fold, *, classes, config, seed, context):
    """Fit one head, either for exactly N epochs or with optimistic early stopping.

    Fixed fitting scores validation only after the final update. Otherwise highest
    validation macro-F1 wins, with unweighted loss breaking ties. ``fit_folds`` uses
    these optimistic fits to choose the other fold's epoch for each seed.
    """
    train = torch.tensor(fold.train_indices, device=context.device)
    validation = torch.tensor(fold.validation_indices, device=context.device)
    x, y = features.to(context.device), targets.to(context.device)
    weights = training_class_weights(y[train], classes)
    # Initialisation is the only random operation: full-batch fitting has no shuffle.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Linear(x.shape[1], len(classes)).to(context.device)
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best_state, best_epoch = [], None, 0
    best_score, best_loss = -1.0, float("inf")
    epoch_limit = config.fixed_epochs if config.fixed_epochs is not None else config.epochs
    for epoch in range(1, epoch_limit + 1):
        head.train()
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(head(x[train]), y[train])
        loss.backward()
        optimiser.step()
        if not math.isfinite(loss.item()):
            raise ValueError("Non-finite loss; check inputs and learning rate")
        history.append(dict(epoch=epoch, train_loss=loss.item()))
        if config.fixed_epochs is not None:
            continue  # No validation-dependent decisions during fixed-epoch fitting.
        head.eval()
        with torch.no_grad():
            logits = head(x[validation])
            val_loss = nn.functional.cross_entropy(logits, y[validation]).item()
            metrics = classification_metrics(
                y[validation].cpu().numpy(), logits.argmax(1).cpu().numpy(), classes
            )
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite loss; check inputs and learning rate")
        history[-1].update(
            validation_loss=val_loss,
            validation_macro_f1=metrics["macro_f1"],
        )
        score = metrics["macro_f1"]
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        if epoch - best_epoch >= config.patience:
            break
    if config.fixed_epochs is None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        logits = head(x[validation])
        val_loss = nn.functional.cross_entropy(logits, y[validation]).item()
        probabilities = logits.softmax(1).cpu().numpy()
    if not math.isfinite(val_loss):
        raise ValueError("Non-finite loss; check inputs and learning rate")
    metrics = classification_metrics(y[validation].cpu().numpy(), probabilities.argmax(1), classes)
    if config.fixed_epochs is not None:
        history[-1].update(validation_loss=val_loss, validation_macro_f1=metrics["macro_f1"])
    return (
        head,
        dict(
            seed=seed,
            best_epoch=best_epoch if config.fixed_epochs is None else None,
            evaluation_epoch=best_epoch if config.fixed_epochs is None else config.fixed_epochs,
            epoch_selection="early-stopping" if config.fixed_epochs is None else "fixed",
            epochs_run=len(history),
            class_weights=dict(zip(classes, weights.cpu().tolist(), strict=True)),
            metrics=metrics,
            history=history,
        ),
        probabilities,
    )


def fit_folds(features, targets, folds, *, classes, config, context):
    """Return one list of (head, run, probabilities) per fold, in seed order.

    Cross-fold selection is defined for the two reciprocal match folds. It avoids
    selecting a fold's epoch on its own validation score, but reuses both matches:
    the recipient validation match trained the donor model. It is not nested CV.
    """
    if config.fixed_epochs is None and len(folds) != 2:
        raise ValueError("Cross-fold epoch selection needs exactly two folds; use --fixed-epochs")
    initial = [
        [
            fit_classifier(
                features, targets, fold, classes=classes, config=config, seed=seed, context=context
            )
            for seed in config.seeds
        ]
        for fold in folds
    ]
    if config.fixed_epochs is not None:
        for runs in initial:
            for _, run, _ in runs:
                run.update(epoch_source_fold=None, optimistic=None)
        return initial
    results = []
    for index, fold in enumerate(folds):
        other = 1 - index
        runs = []
        for position, seed in enumerate(config.seeds):
            donor = initial[other][position][1]
            head, run, probabilities = fit_classifier(
                features,
                targets,
                fold,
                classes=classes,
                config=replace(config, fixed_epochs=donor["best_epoch"]),
                seed=seed,
                context=context,
            )
            run.update(
                epoch_selection="cross-fold",
                epoch_source_fold=other + 1,
                epoch_source_validation_match_id=folds[other].validation_match_id,
                optimistic=initial[index][position][1],
            )
            runs.append((head, run, probabilities))
        results.append(runs)
    return results


def _write_json(path, value):
    write_new_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _spread(values):
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
    )


def summarise_runs(fold_results, seeds):
    """Average folds equally per seed, then report variation across seeds (population SD)."""
    per_seed = []
    for seed in seeds:
        metrics = [next(r for r in f["runs"] if r["seed"] == seed)["metrics"] for f in fold_results]
        per_seed.append(
            dict(
                seed=seed,
                **{
                    key: float(np.mean([m[key] for m in metrics]))
                    for key in ("macro_f1", "accuracy")
                },
            )
        )
    return {
        "per_seed": per_seed,
        "model": {key: _spread([r[key] for r in per_seed]) for key in ("macro_f1", "accuracy")},
        "majority": {
            key: float(np.mean([f["majority"][key] for f in fold_results]))
            for key in ("macro_f1", "accuracy")
        },
    }


def run_baseline(manifest, output, *, context: RunContext, config=None, exclude_uncertain=False):
    """Extract fixed features once, run match folds, and save to a fresh directory.

    No statistics are fitted during feature extraction, so all development clips
    can share that pass. Fitting and class weighting use only the training indices.
    """
    config = config or BaselineConfig()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    dataset = TacticClipDataset(
        manifest,
        context=context,
        n_frames=config.n_frames,
        size=(config.size, config.size),
        split="train",
        exclude_uncertain=exclude_uncertain,
    )
    classes = dataset.classes
    targets = torch.tensor([classes.index(r.label) for r in dataset.records])
    folds = whole_match_folds(dataset)
    if config.fixed_epochs is None and len(folds) != 2:
        raise ValueError("Cross-fold epoch selection needs exactly two folds; use --fixed-epochs")
    for fold in folds:
        training_class_weights(targets[list(fold.train_indices)], classes)
    output.mkdir(parents=True, exist_ok=False)
    run_config = {
        "model": "resnet18-mean-linear",
        "settings": asdict(config),
        "classes": list(classes),
        "weights": WEIGHTS,
        "normalisation": {"mean": RGB_MEAN, "std": RGB_STD},
        "spatial_preprocessing": "Whole-map OpenCV INTER_AREA resize; no crop or augmentation",
        "optimiser": "AdamW, full-batch",
        "epoch_mode": config.epoch_mode,
        "selection": (
            "Other fold's early-stopping epoch, matched by seed; refit to that exact epoch"
            if config.fixed_epochs is None
            else "Exactly fixed_epochs updates; no early stopping"
        ),
        "optimistic_selection": (
            "Own validation macro-F1; loss breaks ties" if config.fixed_epochs is None else None
        ),
        "device": context.device,
        "torch_threads": torch.get_num_threads(),
        "versions": {
            name: version(name) for name in ("torch", "torchvision", "numpy", "opencv-python")
        },
        "python": platform.python_version(),
        "manifest": str(dataset.manifest),
        "manifest_sha256": file_sha256(dataset.manifest),
        "split": dataset.split,
        "exclude_uncertain": exclude_uncertain,
        "excluded_clip_ids": list(dataset.excluded_clip_ids),
        "class_counts": class_counts(targets.numpy(), classes),
        "definition_version": dataset.records[0].definition_version,
    }
    _write_json(output / "config.json", run_config)
    write_new_text(output / "input-manifest.csv", dataset.manifest.read_text(encoding="utf-8-sig"))
    print(
        f"Using {len(dataset)} development clips; exclusions: {list(dataset.excluded_clip_ids)}",
        flush=True,
    )
    print(f"Class counts: {run_config['class_counts']}", flush=True)
    model = build_baseline(context=context).eval()
    save_checkpoint(
        {k: v.cpu() for k, v in model.encoder.state_dict().items()}, output / "encoder.pt"
    )
    encoder_hash = file_sha256(output / "encoder.pt")
    features, samples = [], []
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        tensor = normalise_frames(sample["frames"], context=context)
        features.append(model.encode(tensor.unsqueeze(0)).cpu())
        samples.append(
            {
                **sample["metadata"],
                "label": record.label,
                "target": sample["target"],
                "clip_sha256": file_sha256(record.path),
                "fps": sample["fps"],
                "duration_s": sample["duration_s"],
                "frame_indices": sample["frame_indices"].tolist(),
                "timestamps_s": sample["timestamps_s"].tolist(),
                "repeat_mask": sample["repeat_mask"].tolist(),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"Extracted features: {index + 1}/{len(dataset)}", flush=True)
    features = torch.cat(features)
    _write_json(output / "samples.json", samples)
    save_checkpoint(
        {
            "features": features,
            "targets": targets,
            "clip_ids": [r.clip_id for r in dataset.records],
        },
        output / "features.pt",
    )
    print(f"Fitting classifiers with {config.epoch_mode} epochs", flush=True)
    fitted = fit_folds(features, targets, folds, classes=classes, config=config, context=context)
    fold_results, predictions = [], []
    for number, fold in enumerate(folds, start=1):
        train_targets = targets[list(fold.train_indices)].numpy()
        val_targets = targets[list(fold.validation_indices)].numpy()
        counts = class_counts(train_targets, classes)
        majority = int(np.argmax(list(counts.values())))  # Ties follow the saved class order.
        result = {
            "fold": number,
            "train_match_ids": fold.train_match_ids,
            "validation_match_id": fold.validation_match_id,
            "train_clip_ids": [dataset.records[i].clip_id for i in fold.train_indices],
            "validation_clip_ids": [dataset.records[i].clip_id for i in fold.validation_indices],
            "train_counts": counts,
            "validation_counts": class_counts(val_targets, classes),
            "majority_class": classes[majority],
            "majority": classification_metrics(
                val_targets, np.full_like(val_targets, majority), classes
            ),
            "runs": [],
        }
        print(f"Fold {number}: train {counts}; validate {result['validation_counts']}", flush=True)
        for head, run, probabilities in fitted[number - 1]:
            seed = run["seed"]
            checkpoint = f"fold-{number}-seed-{seed}.pt"
            save_checkpoint(
                {
                    "model": run_config["model"],
                    "classes": list(classes),
                    "settings": asdict(config),
                    "weights": WEIGHTS,
                    "encoder_sha256": encoder_hash,
                    "definition_version": run_config["definition_version"],
                    "manifest_sha256": run_config["manifest_sha256"],
                    "train_match_ids": fold.train_match_ids,
                    "validation_match_id": fold.validation_match_id,
                    "seed": seed,
                    "best_epoch": run["best_epoch"],
                    "evaluation_epoch": run["evaluation_epoch"],
                    "epoch_selection": run["epoch_selection"],
                    "epoch_source_fold": run["epoch_source_fold"],
                    "classifier": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                },
                output / checkpoint,
            )
            _write_json(output / f"fold-{number}-seed-{seed}-history.json", run.pop("history"))
            if run["optimistic"] is not None:
                _write_json(
                    output / f"fold-{number}-seed-{seed}-optimistic-history.json",
                    run["optimistic"].pop("history"),
                )
            result["runs"].append({**run, "checkpoint": checkpoint})
            for index, probs in zip(fold.validation_indices, probabilities, strict=True):
                record = dataset.records[index]
                predictions.append(
                    dict(
                        fold=number,
                        seed=seed,
                        epoch_selection=run["epoch_selection"],
                        evaluation_epoch=run["evaluation_epoch"],
                        clip_id=record.clip_id,
                        match_id=record.match_id,
                        truth=record.label,
                        predicted=classes[int(probs.argmax())],
                        majority_predicted=classes[majority],
                        **{f"p_{name}": float(p) for name, p in zip(classes, probs, strict=True)},
                    )
                )
            print(
                f"  seed {seed}: macro-F1 {run['metrics']['macro_f1']:.3f}, "
                f"accuracy {run['metrics']['accuracy']:.3f}, epoch {run['evaluation_epoch']}",
                flush=True,
            )
        result["seed_spread"] = {
            key: _spread([run["metrics"][key] for run in result["runs"]])
            for key in ("macro_f1", "accuracy")
        }
        result["optimistic_seed_spread"] = (
            {
                key: _spread([run["optimistic"]["metrics"][key] for run in result["runs"]])
                for key in ("macro_f1", "accuracy")
            }
            if config.fixed_epochs is None
            else None
        )
        fold_results.append(result)
        _write_json(output / f"fold-{number}.json", result)
    summary = summarise_runs(fold_results, config.seeds)
    summary["epoch_mode"] = config.epoch_mode
    summary["optimistic"] = (
        summarise_runs(
            [
                {**fold, "runs": [run["optimistic"] for run in fold["runs"]]}
                for fold in fold_results
            ],
            config.seeds,
        )["model"]
        if config.fixed_epochs is None
        else None
    )
    summary["limitations"] = [
        (
            "Cross-fold still reuses both matches: each validation match trained the other "
            "fold's model. This is not nested CV or an independent test score."
            if config.fixed_epochs is None
            else "Fixed epochs avoid validation-based stopping only if N is chosen before seeing "
            "these scores. Repeated tuning still uses the development matches."
        ),
        "The optimistic column selects and scores the epoch on the same validation match."
        if config.fixed_epochs is None
        else "Optimistic selection is not run in fixed-epoch mode.",
        "Seed spread describes initialisation variability, not uncertainty across unseen matches.",
        "Temporal mean pooling ignores movement order and elapsed time.",
        "No synthetic pretraining, detection, temporal localisation or final test evaluation.",
    ]
    _write_json(output / "summary.json", summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(predictions[0]))
    writer.writeheader()
    writer.writerows(predictions)
    write_new_text(output / "predictions.csv", buffer.getvalue())
    write_new_text(output / "report.md", baseline_report(fold_results, summary, dataset))
    print(f"Saved development results: {output / 'report.md'}", flush=True)
    return summary


def baseline_report(fold_results, summary, dataset):
    """Build a plain Markdown report; all spread values describe variation across seeds."""
    mode = summary["epoch_mode"]
    lines = [
        "# Plan A development results",
        "",
        f"Main result: {mode}. Macro-F1 includes all five classes; values are mean +/- SD across seeds.",
        "Cross-fold uses the other fold's selected epoch for the same seed."
        if mode == "cross-fold"
        else "Fixed mode trains exactly N epochs and scores only epoch N.",
        "",
        f"| Validation match | {mode} macro-F1 | optimistic (selected on validation) macro-F1 | {mode} accuracy | Majority macro-F1 | Majority accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def spread_text(spread):
        return f"{spread['mean']:.3f} +/- {spread['std']:.3f}"

    for fold in fold_results:
        name = next(
            r.match_folder for r in dataset.records if r.match_id == fold["validation_match_id"]
        )
        optimistic = fold["optimistic_seed_spread"]
        lines.append(
            f"| {name} | {spread_text(fold['seed_spread']['macro_f1'])} | "
            f"{spread_text(optimistic['macro_f1']) if optimistic else 'not run'} | "
            f"{spread_text(fold['seed_spread']['accuracy'])} | "
            f"{fold['majority']['macro_f1']:.3f} | {fold['majority']['accuracy']:.3f} |"
        )
    optimistic = summary["optimistic"]
    lines += [
        f"| Equal fold average | {spread_text(summary['model']['macro_f1'])} | "
        f"{spread_text(optimistic['macro_f1']) if optimistic else 'not run'} | "
        f"{spread_text(summary['model']['accuracy'])} | "
        f"{summary['majority']['macro_f1']:.3f} | {summary['majority']['accuracy']:.3f} |",
        "",
        "The overall row averages folds per seed before calculating the spread.",
        "",
        "## Epochs used",
        "",
        "| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold in fold_results:
        name = next(
            r.match_folder for r in dataset.records if r.match_id == fold["validation_match_id"]
        )
        for run in fold["runs"]:
            own = run["optimistic"]["best_epoch"] if run["optimistic"] else "not run"
            lines.append(
                f"| {name} | {run['seed']} | {run['evaluation_epoch']} | "
                f"{run['epoch_source_fold'] or 'fixed N'} | {own} |"
            )
    for fold in fold_results:
        name = next(
            r.match_folder for r in dataset.records if r.match_id == fold["validation_match_id"]
        )
        lines += [
            "",
            f"## Per-class F1: {name}",
            "",
            f"{mode} scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.",
            "",
            "| Class | Clips | Mean F1 |",
            "|---|---:|---:|",
        ]
        for label in dataset.classes:
            f1 = np.mean([run["metrics"]["per_class"][label]["f1"] for run in fold["runs"]])
            lines.append(f"| {label} | {fold['validation_counts'][label]} | {f1:.3f} |")
    lines += [
        "",
        f"Excluded clips: {', '.join(dataset.excluded_clip_ids) or 'none'}.",
        "",
        "See fold JSON files for class counts, weights, per-class scores and confusion matrices",
        "(rows are true labels; columns are predictions, in config.json class order).",
        "",
        *summary["limitations"],
        "",
    ]
    return "\n".join(lines)
