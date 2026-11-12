"""Residual dilated TCN for five-class tactical clips, with a time-blind control.

Input is a clip's per-frame features (tactic_features.py) at a fixed frame rate, so elapsed
time is preserved: clips are never stretched to a fixed length. Clips of different lengths
are padded and masked; padded steps are zero at every convolution input, so a clip's output
does not depend on what it is batched with. Dilated residual blocks see short bursts and
longer build-ups; masked mean+max pooling then covers the whole clip.

Protocol (docs/tactic-definitions.md, README): choose settings by leave-one-match-out
cross-validation on the training matches only; the test match is scored only when asked,
once, by models trained on all training matches. Report macro-F1, per-class recall and a
confusion matrix, never accuracy alone. Importing this module trains nothing.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TACTIC_CLASSES, set_seed
from .storage import save_checkpoint, write_new_text
from .tactic_features import cached_detections, clip_features, feature_dim, load_dataset


@dataclass
class TacticConfig:
    grid: int = 8
    step: int = 2  # 10 fps clips -> 5 fps steps; the start frame is randomised in training
    channels: int = 32
    kernel: int = 3
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.2
    epochs: int = 150
    eval_every: int = 10
    batch: int = 8
    lr: float = 1e-3
    warmup_epochs: int = 5
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    jitter: float = 0.005  # normalised position noise in training
    drop: float = 0.05  # fraction of detections dropped in training
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    models: Tuple[str, ...] = ("tcn", "control")


def receptive_field(kernel: int, dilations: Sequence[int]) -> int:
    """Steps one output of the conv stack sees (two convolutions per block)."""
    return 1 + 2 * (kernel - 1) * sum(dilations)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        # Per-step channel norm: padded steps cannot leak into the statistics.
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _norm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self._norm(self.norm1, self.conv1(x)))) * mask
        h = self.drop(self._norm(self.norm2, self.conv2(h))) * mask
        return F.gelu(x + h) * mask


class ResidualTCN(nn.Module):
    """(batch, features, steps) + mask (batch, 1, steps) -> class logits."""

    def __init__(self, in_dim, n_classes, channels=32, kernel=3, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so the convolutions stay centred")
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.blocks = nn.ModuleList(ResidualBlock(channels, kernel, d, dropout) for d in dilations)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) * mask
        for block in self.blocks:
            h = block(h, mask)
        return self.head(masked_pool(h, mask))


class PooledControl(nn.Module):
    """Time-blind control: the same features, pooled over the clip before any model sees
    them, so step order cannot matter. If it matches the TCN, order is not being used."""

    def __init__(self, in_dim, n_classes, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(masked_pool(x, mask))


def masked_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean and max over valid steps -> (batch, 2 * channels)."""
    mean = (h * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    peak = h.masked_fill(mask == 0, float("-inf")).amax(-1)
    return torch.cat([mean, peak], 1)


def build_model(kind: str, in_dim: int, cfg: TacticConfig) -> nn.Module:
    n = len(TACTIC_CLASSES)
    if kind == "tcn":
        return ResidualTCN(in_dim, n, cfg.channels, cfg.kernel, cfg.dilations, cfg.dropout)
    if kind == "control":
        return PooledControl(in_dim, n, cfg.channels, cfg.dropout)
    raise ValueError(f"unknown model {kind!r}")


def pad_batch(feats: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of (steps, dim) -> x (batch, dim, max_steps) and mask (batch, 1, max_steps)."""
    longest = max(len(f) for f in feats)
    x = np.zeros((len(feats), feats[0].shape[1], longest), np.float32)
    mask = np.zeros((len(feats), 1, longest), np.float32)
    for i, f in enumerate(feats):
        x[i, :, : len(f)] = f.T
        mask[i, 0, : len(f)] = 1
    return torch.from_numpy(x), torch.from_numpy(mask)


@dataclass
class Normaliser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feats: Sequence[np.ndarray]) -> "Normaliser":
        allsteps = np.concatenate(feats, 0)
        return cls(allsteps.mean(0), np.maximum(allsteps.std(0), 1e-3))

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return ((f - self.mean) / self.std).astype(np.float32)


def classification_metrics(y_true, y_pred, n_classes: int = len(TACTIC_CLASSES)) -> Dict:
    """Per-class precision/recall/F1/support, macro-F1 over classes present in y_true,
    accuracy and the confusion matrix (rows = true, columns = predicted)."""
    conf = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    per_class, f1s = {}, []
    for c in range(n_classes):
        tp, support, predicted = conf[c, c], conf[c].sum(), conf[:, c].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[TACTIC_CLASSES[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        if support:
            f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": float(np.trace(conf) / max(conf.sum(), 1)),
        "per_class": per_class,
        "confusion": conf.tolist(),
    }


def class_weights(ys: Sequence[int], n_classes: int = len(TACTIC_CLASSES)) -> torch.Tensor:
    counts = np.bincount(np.asarray(ys, int), minlength=n_classes).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def predict(model: nn.Module, feats: Sequence[np.ndarray], device="cpu") -> np.ndarray:
    """Class probabilities (clips, classes), one clip at a time (no padding)."""
    model.eval()
    out = []
    with torch.no_grad():
        for f in feats:
            x, mask = pad_batch([f])
            out.append(F.softmax(model(x.to(device), mask.to(device)), -1)[0].cpu().numpy())
    return np.stack(out)


def train_model(
    kind: str,
    train: Sequence[Tuple[Dict, int]],
    cfg: TacticConfig,
    seed: int,
    *,
    epochs: Optional[int] = None,
    validate: Optional[Callable[[nn.Module, Normaliser], float]] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, Normaliser, List[Tuple[int, float]]]:
    """Train on (detections, class) pairs. validate(model, norm) is called every
    cfg.eval_every epochs and its scores returned as [(epoch, score)]."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = cfg.epochs if epochs is None else epochs
    dets, ys = [d for d, _ in train], [y for _, y in train]
    norm = Normaliser.fit([clip_features(d, step=cfg.step, grid=cfg.grid) for d in dets])
    model = build_model(kind, feature_dim(cfg.grid), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear warm-up, then a constant rate. The schedule must not depend on the total
    # epoch count: the epoch chosen in cross-validation (a checkpoint inside a longer run)
    # has to be the same model as a final run trained for exactly that many epochs.
    warm_steps = max(cfg.warmup_epochs * math.ceil(len(train) / cfg.batch), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warm_steps))
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(ys).to(device), label_smoothing=cfg.label_smoothing
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in np.array_split(rng.permutation(len(train)), math.ceil(len(train) / cfg.batch)):
            feats = [
                norm(
                    clip_features(
                        dets[i],
                        step=cfg.step,
                        offset=int(rng.integers(cfg.step)),
                        grid=cfg.grid,
                        rng=rng,
                        jitter=cfg.jitter,
                        drop=cfg.drop,
                    )
                )
                for i in idx
            ]
            x, mask = pad_batch(feats)
            y = torch.tensor([ys[i] for i in idx], device=device)
            loss = loss_fn(model(x.to(device), mask.to(device)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if validate is not None and (epoch % cfg.eval_every == 0 or epoch == epochs):
            history.append((epoch, float(validate(model, norm))))
    return model, norm, history


def _eval_features(dets: Sequence[Dict], norm: Normaliser, cfg: TacticConfig) -> List[np.ndarray]:
    return [norm(clip_features(d, step=cfg.step, grid=cfg.grid)) for d in dets]


def cross_validate(kind: str, rows: List[Dict], dets: Dict, cfg: TacticConfig, device="cpu"):
    """Leave one training match out, then score each held-out match at the epoch that
    was best on the OTHER held-out matches (mean over seeds; earliest on ties), so no
    match chooses its own epoch. That cross-fold score is the main result; picking one
    epoch on all folds and scoring the same folds is kept as "optimistic". "pooled"
    gives per-class metrics at the cross-fold epochs: every training clip predicted once
    by a model that never saw its match, per seed and as a seed ensemble (mean
    probabilities), with per-clip rows. "final_epoch" (best mean over all folds) is the
    epoch count for training on all training matches."""
    train_rows = [r for r in rows if r["split"] == "train"]
    matches = sorted({r["match"] for r in train_rows})
    if len(matches) < 2:
        raise ValueError("cross-validation needs at least two training matches")
    runs = []
    for held in matches:
        fit = [(dets[r["clip_id"]], r["y"]) for r in train_rows if r["match"] != held]
        val = [r for r in train_rows if r["match"] == held]
        for seed in cfg.seeds:
            probs_at: List[np.ndarray] = []

            def validate(model, norm, val=val, probs_at=probs_at):
                probs = predict(
                    model, _eval_features([dets[r["clip_id"]] for r in val], norm, cfg), device
                )
                probs_at.append(probs)
                return classification_metrics([r["y"] for r in val], probs.argmax(1))["macro_f1"]

            _, _, history = train_model(kind, fit, cfg, seed, validate=validate, device=device)
            runs.append({"held_out": held, "seed": seed, "history": history, "probs": probs_at})
    epochs = [e for e, _ in runs[0]["history"]]
    scores = {
        m: np.array([[h[1] for h in r["history"]] for r in runs if r["held_out"] == m])
        for m in matches
    }  # (seeds, checkpoints) per held-out match
    fold_curve = {m: scores[m].mean(0) for m in matches}
    mean = np.mean([fold_curve[m] for m in matches], 0)
    best = int(np.argmax(mean))
    chosen = cross_fold_choice(fold_curve)
    per_fold = {m: float(fold_curve[m][chosen[m]]) for m in matches}
    # Clips in the order the folds visited them, matching each run's probability rows.
    ordered = [r for m in matches for r in train_rows if r["match"] == m]
    y_true = [r["y"] for r in ordered]
    per_seed = [
        np.concatenate(
            [
                next(r for r in runs if r["held_out"] == m and r["seed"] == s)["probs"][chosen[m]]
                for m in matches
            ]
        )
        for s in cfg.seeds
    ]
    seed_scores = [classification_metrics(y_true, p.argmax(1))["macro_f1"] for p in per_seed]
    mean_probs = np.mean(per_seed, 0)
    return {
        "runs": [{k: v for k, v in r.items() if k != "probs"} for r in runs],
        "curve": [(e, float(v)) for e, v in zip(epochs, mean)],
        "cross_fold_macro_f1": float(np.mean(list(per_fold.values()))),
        "cross_fold_epochs": {m: epochs[chosen[m]] for m in matches},
        "per_fold_macro_f1": per_fold,
        "optimistic_macro_f1": float(mean[best]),
        "final_epoch": epochs[best],
        "pooled": {
            "seed_macro_f1": seed_scores,
            "seed_macro_f1_mean": float(np.mean(seed_scores)),
            "seed_macro_f1_std": float(np.std(seed_scores)),
            "ensemble": classification_metrics(y_true, mean_probs.argmax(1)),
        },
        "predictions": prediction_rows(kind, ordered, mean_probs),
    }


def cross_fold_choice(fold_curves: Dict[str, np.ndarray]) -> Dict[str, int]:
    """For each held-out match, the checkpoint index that was best (earliest on ties) on
    the mean validation curve of the OTHER held-out matches."""
    return {
        m: int(np.argmax(np.mean([c for o, c in fold_curves.items() if o != m], 0)))
        for m in fold_curves
    }


def prediction_rows(kind: str, rows: Sequence[Dict], probs: np.ndarray) -> List[Dict]:
    """One CSV row per clip: true and predicted class with every class probability."""
    return [
        {
            "model": kind,
            "clip_id": r["clip_id"],
            "match": r["match"],
            "true": r["label"],
            "predicted": TACTIC_CLASSES[int(p.argmax())],
            "duration_s": r["duration_s"],
            **{f"p_{c}": round(float(v), 4) for c, v in zip(TACTIC_CLASSES, p)},
        }
        for r, p in zip(rows, probs)
    ]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_new_text(path, buffer.getvalue())


def majority_baseline(rows: List[Dict]) -> Dict:
    """Macro-F1 of always predicting the most common training class, per CV fold and test."""
    train_rows = [r for r in rows if r["split"] == "train"]
    out = {}
    for held in sorted({r["match"] for r in train_rows}):
        fit = [r["y"] for r in train_rows if r["match"] != held]
        val = [r["y"] for r in train_rows if r["match"] == held]
        guess = int(np.bincount(fit).argmax())
        out[held] = classification_metrics(val, [guess] * len(val))["macro_f1"]
    guess = int(np.bincount([r["y"] for r in train_rows]).argmax())
    test = [r["y"] for r in rows if r["split"] == "test"]
    out["test"] = classification_metrics(test, [guess] * len(test))["macro_f1"] if test else None
    out["class"] = TACTIC_CLASSES[guess]
    return out


def run_tactic_experiment(
    dataset,
    output,
    cfg: TacticConfig = TacticConfig(),
    *,
    evaluate_test: bool = False,
    exclude_uncertain: bool = False,
    cache_dir="runs/tactic-cache",
    device: str = "cpu",
    log=print,
    rows: Optional[List[Dict]] = None,
    dets: Optional[Dict] = None,
) -> Dict:
    """Cross-validate each model on the training matches, then (only with evaluate_test)
    train on all training matches and score the test match once. Writes report.json,
    report.md, config.json, and with evaluate_test predictions.csv and checkpoints."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; choose a new --output folder, e.g. runs/tactic-tcn-002"
        )
    rows = load_dataset(dataset) if rows is None else rows
    uncertain = [r["clip_id"] for r in rows if r.get("uncertain")]
    if uncertain and not exclude_uncertain:
        raise ValueError(
            f"uncertain boundary in {', '.join(uncertain)}; resolve the annotation or pass "
            "--exclude-uncertain and report the exclusion"
        )
    if uncertain:
        log(f"Excluded {len(uncertain)} uncertain-boundary clips: {', '.join(uncertain)}")
        rows = [r for r in rows if not r.get("uncertain")]
    output.mkdir(parents=True)
    if dets is None:
        log(f"Detections for {len(rows)} clips (cached in {cache_dir})")
        dets = cached_detections(rows, cache_dir, log=log)
    report = {
        "dataset": str(dataset),
        "config": asdict(cfg),
        "receptive_field_s": receptive_field(cfg.kernel, cfg.dilations) * cfg.step / 10.0,
        "clips": {
            split: {
                c: sum(r["split"] == split and r["label"] == c for r in rows)
                for c in TACTIC_CLASSES
            }
            for split in ("train", "test")
        },
        "excluded_uncertain": uncertain,
        "majority": majority_baseline(rows),
        "models": {},
        "test_evaluated": evaluate_test,
    }
    test_rows = [r for r in rows if r["split"] == "test"]
    train_pairs = [(dets[r["clip_id"]], r["y"]) for r in rows if r["split"] == "train"]
    predictions, cv_predictions = [], []
    for kind in cfg.models:
        log(f"Cross-validating {kind} ({len(cfg.seeds)} seeds x leave-one-match-out)")
        cv = cross_validate(kind, rows, dets, cfg, device)
        log(
            f"  cross-fold macro-F1 {cv['cross_fold_macro_f1']:.3f} (optimistic "
            f"{cv['optimistic_macro_f1']:.3f}); final epoch count {cv['final_epoch']}"
        )
        cv_predictions += cv["predictions"]
        entry = {
            "cv": {k: v for k, v in cv.items() if k not in ("runs", "predictions")},
            "cv_runs": cv["runs"],
        }
        if evaluate_test and test_rows:
            log(
                f"Training {kind} on all training matches for {cv['final_epoch']} epochs; scoring test"
            )
            probs, seed_scores = [], []
            for seed in cfg.seeds:
                model, norm, _ = train_model(
                    kind, train_pairs, cfg, seed, epochs=cv["final_epoch"], device=device
                )
                p = predict(
                    model,
                    _eval_features([dets[r["clip_id"]] for r in test_rows], norm, cfg),
                    device,
                )
                probs.append(p)
                seed_scores.append(
                    classification_metrics([r["y"] for r in test_rows], p.argmax(1))["macro_f1"]
                )
                save_checkpoint(
                    {
                        "kind": kind,
                        "state_dict": model.state_dict(),
                        "norm_mean": norm.mean,
                        "norm_std": norm.std,
                        "config": asdict(cfg),
                        "epochs": cv["final_epoch"],
                        "classes": TACTIC_CLASSES,
                    },
                    output / "models" / f"{kind}-seed{seed}.pt",
                )
            mean_probs = np.mean(probs, 0)
            entry["test"] = {
                "seed_macro_f1": seed_scores,
                "seed_macro_f1_mean": float(np.mean(seed_scores)),
                "seed_macro_f1_std": float(np.std(seed_scores)),
                "ensemble": classification_metrics(
                    [r["y"] for r in test_rows], mean_probs.argmax(1)
                ),
            }
            predictions += prediction_rows(kind, test_rows, mean_probs)
        report["models"][kind] = entry
    write_new_text(output / "config.json", json.dumps(asdict(cfg), indent=2) + "\n")
    write_new_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    write_new_text(output / "report.md", render_report(report))
    if cv_predictions:
        _write_csv(output / "cv_predictions.csv", cv_predictions)
    if predictions:
        _write_csv(output / "predictions.csv", predictions)
    return report


def render_report(report: Dict) -> str:
    """A short Markdown summary of report.json."""
    lines = [
        "# Tactic TCN experiment",
        "",
        f"Dataset: `{report['dataset']}`. Receptive field of the conv stack: "
        f"{report['receptive_field_s']:.1f} s (pooling then covers the whole clip).",
        "",
        "| Split | " + " | ".join(TACTIC_CLASSES) + " |",
        "|---|" + "---|" * len(TACTIC_CLASSES),
    ]
    for split, counts in report["clips"].items():
        lines.append(f"| {split} | " + " | ".join(str(counts[c]) for c in TACTIC_CLASSES) + " |")
    maj = report["majority"]
    folds = [m for m in maj if m not in ("test", "class")]
    if report.get("excluded_uncertain"):
        lines += [
            "",
            "Excluded (uncertain boundary): " + ", ".join(report["excluded_uncertain"]) + ".",
        ]
    lines += [
        "",
        "## Cross-validation (training matches only)",
        "",
        "Each training match is held out in turn and scored (mean macro-F1 over seeds) at the "
        "epoch that was best on the other match, so no match picks its own epoch. "
        "Optimistic picks one epoch on all folds and scores the same folds; use it only as a "
        "comparison. Final epochs: used when training on all training matches.",
        "",
        "| Model | Cross-fold | "
        + " | ".join(f"held out: {m} (epoch)" for m in folds)
        + " | Optimistic | Final epochs |",
        "|---|---|" + "---|" * len(folds) + "---|---|",
        f"| majority ({maj['class']}) | {np.mean([maj[m] for m in folds]):.3f} | "
        + " | ".join(f"{maj[m]:.3f}" for m in folds)
        + " | – | – |",
    ]
    for kind, entry in report["models"].items():
        cv = entry["cv"]
        lines.append(
            f"| {kind} | {cv['cross_fold_macro_f1']:.3f} | "
            + " | ".join(
                f"{cv['per_fold_macro_f1'][m]:.3f} ({cv['cross_fold_epochs'][m]})" for m in folds
            )
            + f" | {cv['optimistic_macro_f1']:.3f} | {cv['final_epoch']} |"
        )
    lines += [
        "",
        "### Per class, pooled over held-out matches",
        "",
        "At the cross-fold epochs every training clip is predicted once, by models that never "
        "saw its match. Pooled macro-F1 can differ from the per-fold mean above. Per-clip "
        "predictions: `cv_predictions.csv`.",
    ]
    for kind, entry in report["models"].items():
        if "pooled" in entry["cv"]:
            lines += _class_section(kind, entry["cv"]["pooled"])
    if not report["test_evaluated"]:
        lines += [
            "",
            "The test match was not scored (run with --evaluate-test once settings are fixed).",
        ]
        return "\n".join(lines) + "\n"
    lines += ["", "## Test match (scored once)", ""]
    lines.append(f"Majority baseline macro-F1: {maj['test']:.3f}.")
    for kind, entry in report["models"].items():
        if "test" in entry:
            lines += _class_section(kind, entry["test"])
    return "\n".join(lines) + "\n"


def _class_section(kind: str, scores: Dict) -> List[str]:
    """Macro-F1 per seed and for the ensemble, then per-class metrics and confusion."""
    ens = scores["ensemble"]
    lines = [
        "",
        f"#### {kind}",
        "",
        f"Macro-F1 per seed: {scores['seed_macro_f1_mean']:.3f} ± "
        f"{scores['seed_macro_f1_std']:.3f}; seed ensemble: {ens['macro_f1']:.3f} "
        f"(accuracy {ens['accuracy']:.3f}).",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for c in TACTIC_CLASSES:
        m = ens["per_class"][c]
        lines.append(
            f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |"
        )
    lines += [
        "",
        "Confusion (rows true, columns predicted):",
        "",
        "| | " + " | ".join(TACTIC_CLASSES) + " |",
        "|---|" + "---|" * len(TACTIC_CLASSES),
    ]
    for c, row in zip(TACTIC_CLASSES, ens["confusion"]):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
    return lines
