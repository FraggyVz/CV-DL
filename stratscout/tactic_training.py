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
from .tactic_data import MatchFold, TacticClipDataset, whole_match_folds
from .tactic_features import extract_frame_features
from .tactic_model import RGB_MEAN, RGB_STD, WEIGHTS, file_sha256


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


EPOCH_MODES = ("nested", "other-folds")


def selection_mode(config, n_folds, epoch_mode="nested") -> str:
    """Name the epoch choice: fixed N, the two-match swap, nested for three or more, or
    ``other-folds``, the TCN's rule (``tactic_tcn.cross_fold_choice``), on request."""
    if n_folds < 2:
        raise ValueError("Whole-match validation needs at least two folds")
    if epoch_mode not in EPOCH_MODES:
        raise ValueError(f"epoch_mode must be one of {', '.join(EPOCH_MODES)}")
    if config.fixed_epochs is not None:
        if epoch_mode != "nested":
            raise ValueError("fixed_epochs and the other-folds epoch mode cannot be combined")
        return "fixed"
    if epoch_mode == "other-folds":
        return "other-folds"
    return "cross-fold" if n_folds == 2 else "nested"


def other_folds_epochs(curves) -> list[int]:
    """For each fold, the 1-based epoch that is best on the mean validation curve of the
    OTHER folds, earliest on ties; a fold's own curve is never used for its epoch.

    ``curves`` holds one curve per fold, already averaged over seeds. This copies
    ``tactic_tcn.cross_fold_choice``: no loss tie-break and no patience.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or len(curves) < 2:
        raise ValueError("Need one full validation curve per fold, at least two folds")
    return [
        int(np.argmax(np.mean(np.delete(curves, index, axis=0), axis=0))) + 1
        for index in range(len(curves))
    ]


def inner_match_folds(fold, folds) -> tuple[MatchFold, ...]:
    """Leave one of ``fold``'s training matches out at a time; its validation match stays out.

    ``folds`` must be the complete leave-one-match-out set, which supplies each match's clips.
    """
    by_match = {f.validation_match_id: f.validation_indices for f in folds}
    if set(fold.train_match_ids) - set(by_match) or len(fold.train_match_ids) < 2:
        raise ValueError("Nested selection needs every training match as a fold, at least two")
    inner = []
    for match in fold.train_match_ids:
        held_out = set(by_match[match])
        if not held_out <= set(fold.train_indices):
            raise ValueError("Folds disagree about which clips belong to a match")
        inner.append(
            MatchFold(
                tuple(i for i in fold.train_indices if i not in held_out),
                tuple(by_match[match]),
                tuple(m for m in fold.train_match_ids if m != match),
                match,
            )
        )
    return tuple(inner)


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


def fit_classifier(features, targets, fold, *, classes, config, seed, context, build_head=None):
    """Fit one head, either for exactly N epochs or with optimistic early stopping.

    The head is a linear layer on (clips, features) unless ``build_head`` returns
    another module for the same ``features`` tensor, whatever its shape.

    Fixed fitting scores validation only after the final update. Otherwise highest
    validation macro-F1 wins, with unweighted loss breaking ties. ``fit_folds`` uses
    these optimistic fits to choose the other fold's epoch for each seed.
    """
    train = torch.tensor(fold.train_indices, device=context.device)
    validation = torch.tensor(fold.validation_indices, device=context.device)
    x, y = features.to(context.device), targets.to(context.device)
    weights = training_class_weights(y[train], classes)
    # Full-batch fitting has no shuffle. Random numbers are drawn only for initialisation
    # and, in heads with dropout, for the training passes; both come from a stream seeded
    # here, so a refit to N epochs repeats the first N epochs of a longer fit exactly.
    # Heads without dropout draw nothing while training, so their results are unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = build_head() if build_head else nn.Linear(x.shape[1], len(classes))
        head = head.to(context.device)
        training_rng = torch.random.get_rng_state()
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
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(training_rng)
            loss = criterion(head(x[train]), y[train])
            training_rng = torch.random.get_rng_state()
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


def nested_epoch(
    features, targets, fold, folds, *, classes, config, seed, context, build_head=None
):
    """Choose ``fold``'s epoch from its training matches only; return (epoch, selection).

    Each inner fit trains on all but one training match and scores that match at
    every epoch. The inner curves are averaged with equal weight per match, then the
    usual rule applies to the mean: highest macro-F1, lower loss breaks ties, stop
    after ``patience`` epochs without improvement. The validation match is never read.
    """
    search = replace(config, patience=config.epochs)  # Full curves; patience applies to the mean.
    inner = inner_match_folds(fold, folds)
    extra = {"build_head": build_head} if build_head else {}
    histories = [
        fit_classifier(
            features,
            targets,
            f,
            classes=classes,
            config=search,
            seed=seed,
            context=context,
            **extra,
        )[1]["history"]
        for f in inner
    ]
    rows, best_epoch, best_score, best_loss = [], 0, -1.0, float("inf")
    for epoch_rows in zip(*histories, strict=True):
        scores = [row["validation_macro_f1"] for row in epoch_rows]
        epoch, score = epoch_rows[0]["epoch"], float(np.mean(scores))
        loss = float(np.mean([row["validation_loss"] for row in epoch_rows]))
        rows.append(
            dict(
                epoch=epoch,
                mean_validation_macro_f1=score,
                mean_validation_loss=loss,
                validation_macro_f1=dict(
                    zip((f.validation_match_id for f in inner), scores, strict=True)
                ),
            )
        )
        if score > best_score or (score == best_score and loss < best_loss):
            best_epoch, best_score, best_loss = epoch, score, loss
        if epoch - best_epoch >= config.patience:
            break
    return best_epoch, dict(
        epoch=best_epoch,
        mean_validation_macro_f1=best_score,
        inner_validation_match_ids=[f.validation_match_id for f in inner],
        history=rows,
    )


def fit_folds(
    features,
    targets,
    folds,
    *,
    classes,
    config,
    context,
    build_head=None,
    epoch_mode="nested",
):
    """Return one list of (head, run, probabilities) per fold, in seed order.

    No fold's epoch is chosen on its own validation match. With two folds each seed
    borrows the other fold's early-stopping epoch; that reuses both matches, because
    the recipient validation match trained the donor model. With three or more folds
    the epoch comes from ``nested_epoch``, so the validation match is scored once and
    takes no part in its own model. Both then refit to exactly the chosen epoch.

    ``epoch_mode="other-folds"`` uses the TCN's rule instead: every fold is fitted for
    all epochs, each fold's curve is averaged over seeds, and a fold is scored at the
    epoch best on the mean curve of the other folds, the same epoch for every seed.
    It is slightly optimistic: the other folds' models trained on the scored match.
    """
    mode = selection_mode(config, len(folds), epoch_mode)
    extra = {"build_head": build_head} if build_head else {}
    # Other-folds needs every fold's full curve, so nothing stops early there.
    first = replace(config, patience=config.epochs) if mode == "other-folds" else config
    initial = [
        [
            fit_classifier(
                features,
                targets,
                fold,
                classes=classes,
                config=first,
                seed=seed,
                context=context,
                **extra,
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
    if mode == "other-folds":
        curves = [
            np.mean(
                [[row["validation_macro_f1"] for row in run["history"]] for _, run, _ in runs],
                axis=0,
            )
            for runs in initial
        ]
        chosen = other_folds_epochs(curves)
    results = []
    for index, fold in enumerate(folds):
        other = 1 - index
        runs = []
        for position, seed in enumerate(config.seeds):
            if mode == "other-folds":
                epoch = chosen[index]
                others = [i for i in range(len(folds)) if i != index]
                source = dict(
                    epoch_selection="other-folds",
                    epoch_source_fold=None,
                    selection=dict(
                        epoch=epoch,
                        mean_other_folds_macro_f1=float(
                            np.mean([curves[i][epoch - 1] for i in others])
                        ),
                        source_validation_match_ids=[folds[i].validation_match_id for i in others],
                        history=[
                            dict(
                                epoch=e,
                                mean_other_folds_macro_f1=float(
                                    np.mean([curves[i][e - 1] for i in others])
                                ),
                                seed_mean_macro_f1=dict(
                                    zip(
                                        (folds[i].validation_match_id for i in others),
                                        (float(curves[i][e - 1]) for i in others),
                                        strict=True,
                                    )
                                ),
                            )
                            for e in range(1, len(curves[index]) + 1)
                        ],
                    ),
                )
            elif mode == "cross-fold":
                epoch = initial[other][position][1]["best_epoch"]
                source = dict(
                    epoch_selection="cross-fold",
                    epoch_source_fold=other + 1,
                    epoch_source_validation_match_id=folds[other].validation_match_id,
                )
            else:
                epoch, selection = nested_epoch(
                    features,
                    targets,
                    fold,
                    folds,
                    classes=classes,
                    config=config,
                    seed=seed,
                    context=context,
                    **extra,
                )
                source = dict(epoch_selection="nested", epoch_source_fold=None, selection=selection)
            head, run, probabilities = fit_classifier(
                features,
                targets,
                fold,
                classes=classes,
                config=replace(config, fixed_epochs=epoch),
                seed=seed,
                context=context,
                **extra,
            )
            run.update(**source, optimistic=initial[index][position][1])
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


def prepare_run(
    manifest, output, *, context: RunContext, config, exclude_uncertain=False, epoch_mode="nested"
):
    """Open the training split and its whole-match folds; refuse bad inputs before writing.

    Returns (dataset, targets, folds, mode). Shared by every model on this protocol.
    """
    if Path(output).exists():
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
    mode = selection_mode(config, len(folds), epoch_mode)
    for fold in folds:
        # Fail before writing anything if a fold, or a nested inner fold, lacks a class.
        for part in (fold, *(inner_match_folds(fold, folds) if mode == "nested" else ())):
            training_class_weights(targets[list(part.train_indices)], classes)
    return dataset, targets, folds, mode


def describe_run(dataset, targets, *, config, context, mode, model, exclude_uncertain):
    """Settings, versions and inputs saved as config.json beside a run's results."""
    classes = dataset.classes
    return {
        "model": model,
        "settings": asdict(config),
        "classes": list(classes),
        "weights": WEIGHTS,
        "normalisation": {"mean": RGB_MEAN, "std": RGB_STD},
        "spatial_preprocessing": "Whole-map OpenCV INTER_AREA resize; no crop or augmentation",
        "optimiser": "AdamW, full-batch",
        "epoch_mode": mode,
        "selection": {
            "cross-fold": "Other fold's early-stopping epoch, matched by seed; "
            "refit to that exact epoch",
            "nested": "Best mean macro-F1 over leave-one-match-out fits on the training matches "
            "only, matched by seed; refit on all of them to that exact epoch",
            "fixed": "Exactly fixed_epochs updates; no early stopping",
            "other-folds": "The TCN's rule: every fold fitted for all epochs; each fold scored "
            "at the epoch best on the other folds' seed-mean macro-F1 curve (earliest on ties), "
            "the same epoch for every seed; refit to that exact epoch",
        }[mode],
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


def run_baseline(
    manifest,
    output,
    *,
    context: RunContext,
    config=None,
    exclude_uncertain=False,
    epoch_mode="nested",
):
    """Extract fixed features once, run match folds, and save to a fresh directory.

    No statistics are fitted during feature extraction, so all development clips
    can share that pass. Fitting and class weighting use only the training indices.
    """
    config = config or BaselineConfig()
    output = Path(output)
    dataset, targets, folds, mode = prepare_run(
        manifest,
        output,
        context=context,
        config=config,
        exclude_uncertain=exclude_uncertain,
        epoch_mode=epoch_mode,
    )
    output.mkdir(parents=True, exist_ok=False)
    run_config = describe_run(
        dataset,
        targets,
        config=config,
        context=context,
        mode=mode,
        model="resnet18-mean-linear",
        exclude_uncertain=exclude_uncertain,
    )
    _write_json(output / "config.json", run_config)
    write_new_text(output / "input-manifest.csv", dataset.manifest.read_text(encoding="utf-8-sig"))
    print(
        f"Using {len(dataset)} development clips; exclusions: {list(dataset.excluded_clip_ids)}",
        flush=True,
    )
    print(f"Class counts: {run_config['class_counts']}", flush=True)
    frames = extract_frame_features(dataset, output, context=context)
    features = frames["features"].mean(dim=1)  # Plan A ignores frame order and elapsed time.
    _write_json(output / "samples.json", frames["samples"])
    save_checkpoint(
        {
            "features": features,
            "targets": targets,
            "clip_ids": [r.clip_id for r in dataset.records],
        },
        output / "features.pt",
    )
    return fit_and_save(
        features,
        targets,
        dataset,
        folds,
        output,
        config=config,
        context=context,
        run_config=run_config,
        encoder_hash=frames["encoder_sha256"],
    )


BASELINE_LIMITS = (
    "Temporal mean pooling ignores movement order and elapsed time.",
    "No synthetic pretraining, detection, temporal localisation or final test evaluation.",
)


def fit_and_save(
    features,
    targets,
    dataset,
    folds,
    output,
    *,
    config,
    context,
    run_config,
    encoder_hash,
    build_head=None,
    title="Plan A development results",
    model_limits=BASELINE_LIMITS,
):
    """Fit every fold and seed on ready-made inputs, then write the run's result files.

    ``features`` is whatever the head consumes, one row per dataset record. This is the
    shared protocol: other models pass their own inputs, ``build_head`` and wording.
    """
    classes = dataset.classes
    mode = run_config["epoch_mode"]
    print(f"Fitting classifiers with {mode} epochs", flush=True)
    fitted = fit_folds(
        features,
        targets,
        folds,
        classes=classes,
        config=config,
        context=context,
        **({"build_head": build_head} if build_head else {}),
        **({"epoch_mode": "other-folds"} if mode == "other-folds" else {}),
    )
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
            if "selection" in run:
                _write_json(
                    output / f"fold-{number}-seed-{seed}-selection-history.json",
                    run["selection"].pop("history"),
                )
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
    summary["epoch_mode"] = mode
    summary["seed_ensemble"] = seed_ensemble(predictions, classes, config.seeds)
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
        {
            "cross-fold": "Cross-fold still reuses both matches: each validation match trained "
            "the other fold's model. This is not nested CV or an independent test score.",
            "nested": "Nested selection keeps each validation match out of its own model and "
            "epoch choice, but every match is still development data: the settings were chosen "
            "while looking at them. Inner fits train on one match fewer than the refit. "
            "This is not an independent test score.",
            "fixed": "Fixed epochs avoid validation-based stopping only if N is chosen before "
            "seeing these scores. Repeated tuning still uses the development matches.",
            "other-folds": "Other-folds is slightly optimistic: the epoch for a match comes "
            "from the other folds' models, and each of those trained on the scored match. "
            "Nested selection is the stricter estimate. This is not an independent test score.",
        }[mode],
        "The optimistic column selects and scores the epoch on the same validation match."
        if config.fixed_epochs is None
        else "Optimistic selection is not run in fixed-epoch mode.",
        "Seed spread describes initialisation variability, not uncertainty across unseen matches.",
        *model_limits,
    ]
    _write_json(output / "summary.json", summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(predictions[0]))
    writer.writeheader()
    writer.writerows(predictions)
    write_new_text(output / "predictions.csv", buffer.getvalue())
    write_new_text(
        output / "report.md", baseline_report(fold_results, summary, dataset, title=title)
    )
    print(f"Saved development results: {output / 'report.md'}", flush=True)
    return summary


def seed_ensemble(predictions, classes, seeds) -> dict:
    """The TCN's seed ensemble: each clip's class probabilities averaged over seeds, then
    scored once over all held-out clips pooled together (not averaged per fold)."""
    by_clip = {}
    for row in predictions:
        by_clip.setdefault(row["clip_id"], {"truth": row["truth"], "probs": {}})
        by_clip[row["clip_id"]]["probs"][row["seed"]] = [row[f"p_{name}"] for name in classes]
    if any(sorted(clip["probs"]) != sorted(seeds) for clip in by_clip.values()):
        raise ValueError("Every clip needs one prediction per seed for the seed ensemble")
    truth = [classes.index(clip["truth"]) for clip in by_clip.values()]
    mean = np.array([np.mean(list(clip["probs"].values()), axis=0) for clip in by_clip.values()])
    per_seed = [
        classification_metrics(
            truth,
            np.array([np.argmax(clip["probs"][seed]) for clip in by_clip.values()]),
            classes,
        )["macro_f1"]
        for seed in seeds
    ]
    return {
        "method": "mean probability over seeds per clip; metrics pooled over all held-out clips",
        "clips": len(by_clip),
        **classification_metrics(truth, mean.argmax(1), classes),
        "pooled_seed_macro_f1": per_seed,
    }


def baseline_report(fold_results, summary, dataset, title="Plan A development results"):
    """Build a plain Markdown report; all spread values describe variation across seeds."""
    mode = summary["epoch_mode"]
    lines = [
        f"# {title}",
        "",
        f"Main result: {mode}. Macro-F1 includes all five classes; values are mean +/- SD across seeds.",
        {
            "cross-fold": "Cross-fold uses the other fold's selected epoch for the same seed.",
            "nested": "Nested picks each epoch by leave-one-match-out on the training matches "
            "only, for the same seed, then refits on all of them.",
            "fixed": "Fixed mode trains exactly N epochs and scores only epoch N.",
            "other-folds": "Other-folds (the TCN's rule) scores each match at the epoch best on "
            "the other folds' seed-mean curve; slightly optimistic.",
        }[mode],
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
        *(
            [
                "",
                f"Seed ensemble (mean probability over seeds, pooled over all "
                f"{summary['seed_ensemble']['clips']} held-out clips): macro-F1 "
                f"{summary['seed_ensemble']['macro_f1']:.3f}, accuracy "
                f"{summary['seed_ensemble']['accuracy']:.3f}.",
            ]
            if summary.get("seed_ensemble")
            else []
        ),
        "",
        "## Epochs used",
        "",
        "| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    source_text = {"nested": "training matches", "other-folds": "other folds"}.get(mode, "fixed N")
    for fold in fold_results:
        name = next(
            r.match_folder for r in dataset.records if r.match_id == fold["validation_match_id"]
        )
        for run in fold["runs"]:
            own = run["optimistic"]["best_epoch"] if run["optimistic"] else "not run"
            lines.append(
                f"| {name} | {run['seed']} | {run['evaluation_epoch']} | "
                f"{run['epoch_source_fold'] or source_text} | {own} |"
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
