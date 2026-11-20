"""Temporal tactic models on the baseline's frozen frame features and protocol.

Every model here sees the same sampled frames, folds, epoch selection, seeds, class
weights and metrics as Plan A; only the head and its inputs differ. This workflow
never opens the test split.

With ``inputs="clip-data"`` the per-frame rows are the shared clip data that the clip-data
TCN reads (72 detector channels plus the ResNet numbers, compressed per fold); see
``ClipDataHead``.
"""

from __future__ import annotations

import csv
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import TACTIC_CLASSES
from .context import RunContext
from .storage import write_new_text
from .tactic_features import (
    WEIGHTS,
    extract_frame_features,
    load_frame_features,
    save_frame_features,
)
from .tactic_model import TacticBaseline, build_baseline, file_sha256
from .tactic_positions import KIND, channel_columns, extract_position_features, native_dataset
from .tactic_training import (
    BaselineConfig,
    describe_run,
    fit_and_save,
    fit_folds,
    prepare_run,
)

MODELS = ("mean", "thirds", "random-thirds", "gru")
DEFAULT_MODELS = ("mean", "thirds", "gru")  # random-thirds is a control, run on request.
TIME_SCALE_S = 30.0  # Fixed, not fitted: seconds are divided by this before entering the GRU.
# The GRU's settings. Three configurations were tried, each fixed before its score was
# seen: the first (32/32, no dropout, last-state readout) overfitted; then smaller with
# dropout; then the same with mean+max pooling over all real frames, the final default.
# Pass the old values to reproduce old runs; each run saves its own in config.json.
GRU_READOUTS = ("mean-max", "last")
GRU_DEFAULTS = {"projection": 8, "hidden": 16, "dropout": 0.3, "readout": "mean-max"}
SMALL_GRU = {"projection": 8, "hidden": 16, "dropout": 0.3, "readout": "last"}
FIRST_GRU = {"projection": 32, "hidden": 32, "dropout": 0.0, "readout": "last"}
MODEL_NAMES = {
    "mean": "resnet18-mean-linear",
    "thirds": "resnet18-thirds-linear",
    "random-thirds": "resnet18-random-thirds-linear",
    "gru": "resnet18-gru-linear",
}
MODEL_PREFIXES = {"images": "resnet18", "positions": "positions", "clip-data": "clipdata"}
DISPLAY_NAMES = {
    "mean": "Mean pooling + linear",
    "thirds": "Temporal average pooling (3 segments) + linear classifier",
    "random-thirds": "Random-segment average pooling + linear classifier (control)",
    "gru": "GRU",
}
TITLES = {
    "mean": "Mean pooling (Plan A) development results",
    "thirds": f"{DISPLAY_NAMES['thirds']} development results",
    "random-thirds": f"{DISPLAY_NAMES['random-thirds']} development results",
    "gru": "GRU development results",
}
LIMITS = {
    "mean": ("Temporal mean pooling ignores movement order and elapsed time.",),
    "thirds": (
        "Temporal average pooling (3 segments) keeps only coarse order: early, middle and "
        "late averages. It does not see elapsed time, so a short and a long clip look alike.",
    ),
    "random-thirds": (
        "Control: the same three segment averages and head as temporal average pooling, but "
        "frames are assigned to the segments at random, so it has the same extra inputs "
        "without the frame order.",
    ),
    "gru": (
        "The GRU reads frames in order with their elapsed time, but has far more weights "
        "than training clips, so it can fit noise.",
    ),
}
INPUTS = ("images", "positions", "clip-data")
POSITION_LIMIT = (
    "Position features come from the colour detector: about 1 frame in 9 misses a player, "
    "about 1 T position in 6 is an inferred carrier, and the inferred carrier can misfire "
    "on a planted bomb. Counts have no player identity."
)
# Clip data: the same per-row features as train-tactics --clip-data (TacticConfig.resnet_pcs).
CLIP_DATA_PCS = 32
CLIP_DATA_MODELS = ("mean", "thirds", "gru")
CLIP_DATA_LIMIT = (
    "Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map "
    "region, drawn after looking at training labels, plus ResNet numbers compressed per fold. "
    "Rows already carry elapsed time and the regions reached so far."
)
SHARED_LIMIT = (
    "No synthetic pretraining, detection, temporal localisation or final test evaluation."
)


def mean_inputs(features: torch.Tensor) -> torch.Tensor:
    """Plan A's clip feature: the plain average of all sampled frames, repeats included."""
    return features.mean(dim=1)


def thirds_inputs(features: torch.Tensor, repeat_mask: torch.Tensor) -> torch.Tensor:
    """Average the early, middle and late third of each clip's frames, then join them.

    Repeated frames are dropped first, so a short clip is not weighted towards a
    duplicate. Part sizes differ by at most one; an empty part (fewer than three
    distinct frames) uses the clip average. Output shape (clips, 3 * features).
    """
    rows = []
    for clip, repeats in zip(features, repeat_mask, strict=True):
        distinct = clip[~repeats]
        whole = distinct.mean(dim=0)
        parts = torch.tensor_split(distinct, 3)
        rows.append(torch.cat([part.mean(dim=0) if len(part) else whole for part in parts]))
    return torch.stack(rows)


def clip_seed(clip_id: str) -> int:
    """A fixed seed per clip, the same on every machine and run."""
    return zlib.crc32(clip_id.encode("utf-8"))


def random_thirds_inputs(features, repeat_mask, clip_ids) -> torch.Tensor:
    """Control for thirds: identical groups and sizes, but frames assigned at random.

    Each clip's distinct frames are shuffled with ``clip_seed(clip_id)`` and then cut
    exactly as ``thirds_inputs`` cuts them, so only the time order is removed.
    """
    if len(clip_ids) != len(features):
        raise ValueError("Need one clip ID per clip")
    shuffled = torch.zeros_like(features)
    for i, (clip, repeats, clip_id) in enumerate(zip(features, repeat_mask, clip_ids, strict=True)):
        distinct = clip[~repeats]
        order = np.random.default_rng(clip_seed(clip_id)).permutation(len(distinct))
        shuffled[i, : len(distinct)] = distinct[torch.as_tensor(order)]
    # Distinct frames now come first in random order; mark the rest as repeats.
    counts = (~repeat_mask).sum(dim=1, keepdim=True)
    moved = torch.arange(features.shape[1]).unsqueeze(0) >= counts
    return thirds_inputs(shuffled, moved)


def sequence_inputs(
    features: torch.Tensor, timestamps_s: torch.Tensor, repeat_mask: torch.Tensor
) -> torch.Tensor:
    """Pack each clip's distinct frames, in order, with their real times.

    Output shape (clips, frames, features + 3). The extra channels are seconds since
    the clip start and seconds since the previous frame, both divided by TIME_SCALE_S,
    and a flag that is 1 for a real frame. Repeated frames are removed and the row is
    padded with zeros at the end, so no time step is invented or counted twice.
    """
    clips, steps, width = features.shape
    packed = torch.zeros(clips, steps, width + 3, dtype=features.dtype)
    for i in range(clips):
        keep = ~repeat_mask[i]
        count = int(keep.sum())
        times = timestamps_s[i][keep].to(features.dtype)
        packed[i, :count, :width] = features[i][keep]
        packed[i, :count, width] = times / TIME_SCALE_S
        packed[i, 1:count, width + 1] = (times[1:] - times[:-1]) / TIME_SCALE_S
        packed[i, :count, width + 2] = 1
    return packed


class TemporalGRU(nn.Module):
    """A small GRU over ``sequence_inputs`` rows, then a linear layer on a pooled readout.

    Frame features are standardised first. The statistics have no weights and come
    only from the frames this head is trained on: full-batch training sees the same
    batch every step, so the running average used at validation equals the training
    fold's own mean and variance. Without this the GRU barely learns in the epoch
    budget, because clips differ by much less than the features' shared level.

    ``readout="mean-max"`` (default) joins the mean and the maximum of the GRU outputs over
    each clip's real frames, so every frame reaches the classifier, not only the last;
    padding never enters either. ``readout="last"`` uses the last real state, as the
    earlier GRUs did. ``dropout`` applies to the projected frame inputs and to the readout
    before the classifier, in training only; it has no weights.
    """

    def __init__(
        self,
        feature_dim=512,
        projection=GRU_DEFAULTS["projection"],
        hidden=GRU_DEFAULTS["hidden"],
        n_classes=5,
        dropout=GRU_DEFAULTS["dropout"],
        readout=GRU_DEFAULTS["readout"],
    ):
        super().__init__()
        if readout not in GRU_READOUTS:
            raise ValueError(f"readout must be one of {', '.join(GRU_READOUTS)}")
        self.feature_dim = feature_dim
        self.readout = readout
        self.standardise = nn.BatchNorm1d(feature_dim, affine=False, momentum=None)
        self.project = nn.Linear(feature_dim, projection)  # Shared by every frame.
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(projection + 2, hidden, batch_first=True)
        self.state_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * (2 if readout == "mean-max" else 1), n_classes)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        width = self.feature_dim
        if packed.ndim != 3 or packed.shape[2] != width + 3:
            raise ValueError("Expected sequence_inputs rows with shape (clips, frames, features+3)")
        lengths = packed[:, :, width + 2].sum(dim=1).long()
        if (lengths < 1).any():
            raise ValueError("Every clip needs at least one real frame")
        real = packed[:, :, width + 2] > 0
        features = torch.zeros_like(packed[:, :, :width])
        features[real] = self.standardise(packed[:, :, :width][real])  # Padding stays zero.
        projected = self.input_dropout(self.project(features))
        steps = torch.cat([projected, packed[:, :, width : width + 2]], 2)
        # Packing stops each clip at its last real frame, so padding never reaches the state.
        sequence = pack_padded_sequence(
            steps, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, last = self.gru(sequence)
        if self.readout == "last":
            return self.classifier(self.state_dropout(last[-1]))
        # Unpacked outputs keep the clip order; padded steps are zeros, masked out below.
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=packed.shape[1])
        mask = (torch.arange(packed.shape[1], device=lengths.device) < lengths[:, None])[..., None]
        mean = (outputs * mask).sum(dim=1) / lengths[:, None].to(outputs.dtype)
        peak = outputs.masked_fill(~mask, float("-inf")).amax(dim=1)
        return self.classifier(self.state_dropout(torch.cat([mean, peak], dim=1)))


def model_inputs(name, frames: dict, classes, gru=None):
    """Return (inputs, build_head, description) for one model name.

    ``gru`` overrides the GRU's settings (``GRU_DEFAULTS``), e.g. ``FIRST_GRU``.
    """
    features = frames["features"]
    if name == "mean":
        return mean_inputs(features), None, {"pooling": "mean over all sampled frames"}
    if name == "thirds":
        return (
            thirds_inputs(features, frames["repeat_mask"]),
            None,
            {
                "pooling": "mean of each of three time-ordered segments, concatenated; repeats dropped"
            },
        )
    if name == "gru":
        width = features.shape[2]
        sizes = {**GRU_DEFAULTS, **(gru or {})}
        if set(sizes) != set(GRU_DEFAULTS):
            raise ValueError(f"gru settings must be {', '.join(GRU_DEFAULTS)}")
        settings = {**sizes, "time_scale_s": TIME_SCALE_S}
        return (
            sequence_inputs(features, frames["timestamps_s"], frames["repeat_mask"]),
            lambda: TemporalGRU(
                width,
                sizes["projection"],
                sizes["hidden"],
                len(classes),
                sizes["dropout"],
                sizes["readout"],
            ),
            {
                **settings,
                "standardisation": "per feature, from the training fold's frames only",
                "time_inputs": "seconds since clip start and since the previous frame",
                "repeats": "dropped; rows padded at the end and packed to their real length",
            },
        )
    if name == "random-thirds":
        return (
            random_thirds_inputs(features, frames["repeat_mask"], frames["clip_ids"]),
            None,
            {
                "pooling": "the three segment averages, but distinct frames assigned to the "
                "segments at random; same segment sizes",
                "assignment_seed": "zlib.crc32(clip_id), numpy default_rng permutation",
            },
        )
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}")


def load_clip_rows(folder, clip_ids) -> dict:
    """The shared clip data for these clips, as one padded per-row cache.

    Loading calls the clip-data TCN's own loader (``tactic_tcn._load_clip_inputs``, which
    reads ``clip_data.load_clip_data``). That loader reads every clip its folder lists, so
    a folder listing any non-training clip is refused before a clip file is opened: pass
    a train-only copy. Each row is the raw detector channels followed by the raw ResNet
    numbers; nothing is fitted here. ``repeat_mask`` marks the padding after each clip's
    last row (clip-data rows are real time, so there are no repeated frames).
    """
    from .tactic_tcn import TacticConfig, _load_clip_inputs

    folder = Path(folder)
    with (folder / "index.csv").open(encoding="utf-8", newline="") as handle:
        other = [row["clip_id"] for row in csv.DictReader(handle) if row["split"] != "train"]
    if other:
        raise ValueError(
            f"{folder} lists {len(other)} non-training clips; use a train-only clip-data copy "
            "so the test split is never read"
        )
    config = TacticConfig(input="clip-data", resnet_pcs=CLIP_DATA_PCS)
    clips, info = _load_clip_inputs(folder, [{"clip_id": c} for c in clip_ids], config)
    first = clips[clip_ids[0]]
    detector_width, resnet_width = first["detector"].shape[1], first["resnet"].shape[1]
    lengths = [len(clips[c]["detector"]) for c in clip_ids]
    features = torch.zeros(len(clip_ids), max(lengths), detector_width + resnet_width)
    timestamps = torch.zeros(len(clip_ids), max(lengths), dtype=torch.float64)
    padding = torch.ones(len(clip_ids), max(lengths), dtype=torch.bool)
    for i, (clip_id, length) in enumerate(zip(clip_ids, lengths, strict=True)):
        clip = clips[clip_id]
        if len(clip["resnet"]) != length or len(clip["t_s"]) != length:
            raise ValueError(f"{clip_id}: detector, ResNet and time rows differ in number")
        features[i, :length, :detector_width] = torch.as_tensor(
            clip["detector"], dtype=torch.float32
        )
        features[i, :length, detector_width:] = torch.as_tensor(
            np.asarray(clip["resnet"], np.float32)
        )
        timestamps[i, :length] = torch.as_tensor(clip["t_s"], dtype=torch.float64)
        padding[i, :length] = False
    return {
        "features": features,
        "timestamps_s": timestamps,
        "repeat_mask": padding,
        "clip_ids": list(clip_ids),
        "samples": [{"clip_id": c, "rows": n} for c, n in zip(clip_ids, lengths, strict=True)],
        "encoder_sha256": None,
        "feature_names": list(first["detector_names"])
        + [f"resnet_{j}" for j in range(resnet_width)],
        "detector_width": detector_width,
        "clip_data": info,
    }


class ClipDataHead(nn.Module):
    """Mean pooling, temporal average pooling or the GRU on clip-data rows.

    Input: ``sequence_inputs`` rows of raw clip data (detector channels, then ResNet
    numbers). On its first training call, which in ``fit_classifier`` is always the
    fold's full training batch, the head fits the clip-data TCN's per-row transform on
    those rows only, by calling its code: ``ResnetCompressor`` (PCA to ``pcs`` numbers)
    and ``Normaliser`` (per-channel scaling) via ``tactic_tcn._raw_features``. These are
    then fixed buffers. A new head is built for every fit, so under nested selection the
    transform comes from the inner training clips only, and validation clips never shape
    it. The pooling is unchanged: the plain mean over rows, ``thirds_inputs``, or
    ``TemporalGRU``. Padding never enters.
    """

    def __init__(
        self, pooling, detector_width, resnet_width, n_classes=5, pcs=CLIP_DATA_PCS, gru=None
    ):
        super().__init__()
        if pooling not in CLIP_DATA_MODELS:
            raise ValueError(f"pooling must be one of {', '.join(CLIP_DATA_MODELS)}")
        self.pooling, self.detector_width, self.pcs = pooling, detector_width, pcs
        self.width = detector_width + resnet_width
        rows = detector_width + pcs
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("pca_mean", torch.zeros(resnet_width))
        self.register_buffer("components", torch.zeros(pcs, resnet_width))
        self.register_buffer("row_mean", torch.zeros(rows))
        self.register_buffer("row_std", torch.ones(rows))
        if pooling == "gru":
            sizes = {**GRU_DEFAULTS, **(gru or {})}
            self.gru = TemporalGRU(
                rows,
                sizes["projection"],
                sizes["hidden"],
                n_classes,
                sizes["dropout"],
                sizes["readout"],
            )
        else:
            self.classifier = nn.Linear(rows * (3 if pooling == "thirds" else 1), n_classes)

    def fit_transform(self, raw: torch.Tensor) -> None:
        """Fit the PCA and scaling on these raw training rows (rows, width), as train_model
        does for the clip-data TCN."""
        from .clip_data import ResnetCompressor
        from .tactic_tcn import Normaliser, TacticConfig, _raw_features

        rows = raw.detach().cpu().numpy()
        clip = {
            "detector": rows[:, : self.detector_width],
            "resnet": rows[:, self.detector_width :],
        }
        config = TacticConfig(input="clip-data", resnet_pcs=self.pcs)
        compressor = ResnetCompressor(self.pcs).fit([clip["resnet"]])
        if len(compressor.components) < self.pcs:
            raise ValueError(
                f"Need at least {self.pcs} training rows and ResNet numbers for the PCA"
            )
        norm = Normaliser.fit([_raw_features(clip, config, compressor)], compressor)
        for name, value in (
            ("pca_mean", compressor.mean),
            ("components", compressor.components),
            ("row_mean", norm.mean),
            ("row_std", norm.std),
        ):
            getattr(self, name).copy_(torch.as_tensor(np.asarray(value, np.float32)))
        self.fitted.fill_(True)

    def transform(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled rows (clips, steps, detector + pcs), zero at padding, and the real-row mask."""
        width = self.width
        if packed.ndim != 3 or packed.shape[2] != width + 3:
            raise ValueError(
                "Expected sequence_inputs rows of raw clip data (clips, rows, width+3)"
            )
        real = packed[:, :, width + 2] > 0
        if not bool(self.fitted):
            if not self.training:
                raise ValueError("Fit the clip-data transform on training rows first")
            self.fit_transform(packed[:, :, :width][real])
        raw = packed[:, :, :width]
        compressed = (raw[:, :, self.detector_width :] - self.pca_mean) @ self.components.T
        rows = torch.cat([raw[:, :, : self.detector_width], compressed], 2)
        rows = (rows - self.row_mean) / self.row_std
        return rows * real.unsqueeze(2), real

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        rows, real = self.transform(packed)
        if self.pooling == "gru":
            return self.gru(torch.cat([rows, packed[:, :, self.width :]], 2))
        if self.pooling == "thirds":
            return self.classifier(thirds_inputs(rows, ~real))
        lengths = real.sum(dim=1, keepdim=True).to(rows.dtype)
        return self.classifier(rows.sum(dim=1) / lengths)


def clip_data_inputs(name, frames: dict, classes, gru=None):
    """(inputs, build_head, description) for one model on clip-data rows."""
    if name not in CLIP_DATA_MODELS:
        raise ValueError(f"Clip data supports {', '.join(CLIP_DATA_MODELS)}, not {name!r}")
    width = frames["features"].shape[2]
    detector = frames["detector_width"]
    sizes = {**GRU_DEFAULTS, **(gru or {})}
    if set(sizes) != set(GRU_DEFAULTS):
        raise ValueError(f"gru settings must be {', '.join(GRU_DEFAULTS)}")
    description = {
        "rows": f"{detector} detector channels + {CLIP_DATA_PCS} ResNet PCA numbers, as the "
        "clip-data TCN",
        "transform": "ResNet PCA and per-channel scaling fitted on each head's training rows "
        "only (inner training clips under nested selection), with the clip-data TCN's code",
        "pooling": {
            "mean": "mean over all rows",
            "thirds": "mean of each of three time-ordered segments, concatenated",
            "gru": "GRU over the rows in order, with elapsed time",
        }[name],
    }
    if name == "gru":
        description.update(sizes, time_scale_s=TIME_SCALE_S)
    return (
        sequence_inputs(frames["features"], frames["timestamps_s"], frames["repeat_mask"]),
        lambda: ClipDataHead(name, detector, width - detector, len(classes), gru=sizes),
        description,
    )


def average_scores(fitted, seeds) -> dict:
    """Equal fold average per seed, then the mean over seeds, as in the main summary."""
    per_seed = [
        [float(np.mean([runs[k][1]["metrics"][key] for runs in fitted])) for k in range(len(seeds))]
        for key in ("macro_f1", "accuracy")
    ]
    return {"macro_f1": float(np.mean(per_seed[0])), "accuracy": float(np.mean(per_seed[1]))}


def shuffled_label_check(
    inputs, targets, folds, *, classes, config, context, build_head, count, epoch_mode="nested"
):
    """Refit the whole protocol on labels shuffled within each match; return the scores.

    Every match keeps its class counts, so the folds and class weights stay valid.
    Scores are against the shuffled labels: a pipeline that leaks nothing should
    land near chance, far below the real score.
    """
    extra = {"build_head": build_head} if build_head else {}
    rows = []
    for permutation in range(count):
        generator = np.random.default_rng(permutation)
        labels = targets.clone()
        for fold in folds:  # A fold's validation clips are exactly one match.
            index = np.array(fold.validation_indices)
            labels[index] = targets[generator.permutation(index)]
        fitted = fit_folds(
            inputs,
            labels,
            folds,
            classes=classes,
            config=config,
            context=context,
            **extra,
            **({"epoch_mode": epoch_mode} if epoch_mode != "nested" else {}),
        )
        rows.append({"permutation": permutation, **average_scores(fitted, config.seeds)})
        print(f"  shuffle {permutation}: macro-F1 {rows[-1]['macro_f1']:.3f}", flush=True)
    return rows


def _spread(values) -> dict:
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
    )


def _write_json(path, value):
    write_new_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_temporal(
    manifest,
    output,
    *,
    context: RunContext,
    config=None,
    exclude_uncertain=False,
    models=DEFAULT_MODELS,
    features=None,
    shuffles=0,
    inputs="images",
    drop_channels=(),
    epoch_mode="nested",
    gru=None,
    clip_data=None,
):
    """Train the chosen models on one shared per-frame cache; save one folder per model.

    ``inputs`` picks the per-frame features: frozen ResNet-18 image features, player
    positions from the minimap detector (``tactic_positions``), or the shared clip data
    (``clip_data``, a train-only build-clip-data folder; see ``ClipDataHead``). ``features`` reuses an
    earlier cache instead of extracting again. ``drop_channels`` removes position
    channels, e.g. ("bomb_badge",), for an ablation. ``shuffles`` repeats each model's
    whole fit that many times on shuffled labels. ``epoch_mode="other-folds"`` uses
    the TCN's epoch rule instead of nested selection (see ``fit_folds``). ``gru``
    overrides the GRU's settings, e.g. ``FIRST_GRU`` to reproduce the first GRU.
    """
    config = config or BaselineConfig()
    output = Path(output)
    models = tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"Choose distinct models from {', '.join(MODELS)}")
    if isinstance(shuffles, bool) or not isinstance(shuffles, int) or shuffles < 0:
        raise ValueError("shuffles must be a non-negative integer")
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {', '.join(INPUTS)}")
    drop_channels = tuple(drop_channels)
    if drop_channels and inputs != "positions":
        raise ValueError("drop_channels applies only to position features")
    if (inputs == "clip-data") != (clip_data is not None):
        raise ValueError("clip-data inputs need a clip_data folder, and only they take one")
    if inputs == "clip-data":
        if features is not None:
            raise ValueError("clip-data inputs read the clip_data folder, not a feature cache")
        if set(models) - set(CLIP_DATA_MODELS):
            raise ValueError(f"Clip data supports {', '.join(CLIP_DATA_MODELS)}")
    dropped = channel_columns(drop_channels) if drop_channels else []
    dataset, targets, folds, mode = prepare_run(
        manifest,
        output,
        context=context,
        config=config,
        exclude_uncertain=exclude_uncertain,
        epoch_mode=epoch_mode,
    )
    classes = dataset.classes
    if inputs == "positions":
        # The detector reads the clips at native size; same clips, frames and times.
        cache_dataset, source, cache_name = native_dataset(dataset), KIND, "position-features.pt"
    elif inputs == "clip-data":
        cache_dataset, source, cache_name = dataset, "clip-data", None
    else:
        cache_dataset, source, cache_name = dataset, WEIGHTS, "frame-features.pt"
    if features is not None:
        frames = load_frame_features(features, cache_dataset, source)
    if inputs == "clip-data":  # Read before any output, so a bad folder leaves nothing behind.
        frames = load_clip_rows(clip_data, [r.clip_id for r in dataset.records])
        info = frames["clip_data"]
        source = f"clip-data {info['data_version']} / {info['regions_version']}"
    output.mkdir(parents=True, exist_ok=False)
    write_new_text(output / "input-manifest.csv", dataset.manifest.read_text(encoding="utf-8-sig"))
    print(
        f"Using {len(dataset)} development clips; exclusions: {list(dataset.excluded_clip_ids)}",
        flush=True,
    )
    if features is None and inputs != "clip-data":
        if inputs == "positions":
            frames = extract_position_features(cache_dataset, context=context)
        else:
            frames = extract_frame_features(dataset, output, context=context)
        save_frame_features(output / cache_name, frames, cache_dataset, source)
        frames = load_frame_features(output / cache_name, cache_dataset, source)
    _write_json(output / "samples.json", frames["samples"])
    names = frames.get("feature_names")
    if dropped:
        keep = [i for i in range(frames["features"].shape[2]) if i not in set(dropped)]
        frames = {**frames, "features": frames["features"][:, :, keep]}
        names = [names[i] for i in keep]
    comparison = {
        "epoch_mode": mode,
        "inputs": inputs,
        "feature_source": source,
        "feature_width": int(frames["features"].shape[2]),
        "dropped_channels": list(drop_channels),
        "models": {},
        "frame_features": str(features) if features is not None else cache_name,
        # Clip-data rows have no repeats; their mask marks padding only.
        "repeated_frames": 0 if inputs == "clip-data" else int(frames["repeat_mask"].sum()),
    }
    if inputs == "clip-data":
        comparison.update(
            frame_features=str(clip_data),
            clip_data=frames["clip_data"],
            row_width=frames["detector_width"] + CLIP_DATA_PCS,
        )
    prefix = MODEL_PREFIXES[inputs]
    for name in models:
        if inputs == "clip-data":
            model_input, build_head, description = clip_data_inputs(name, frames, classes, gru)
        else:
            model_input, build_head, description = model_inputs(name, frames, classes, gru)
        run_config = describe_run(
            dataset,
            targets,
            config=config,
            context=context,
            mode=mode,
            model=MODEL_NAMES[name].replace("resnet18", prefix),
            exclude_uncertain=exclude_uncertain,
        )
        run_config["temporal"] = description
        run_config["inputs"] = {
            "kind": inputs,
            "source": source,
            "width": comparison["feature_width"],
            "dropped_channels": list(drop_channels),
            "feature_names": names,
        }
        if inputs == "positions":
            # Settings that only describe image extraction do not apply here.
            run_config["spatial_preprocessing"] = "Native-size clips for the detector; no resize"
            run_config["weights"] = None
            run_config["normalisation"] = None
        if inputs == "clip-data":
            run_config["inputs"]["clip_data"] = frames["clip_data"]
            run_config["spatial_preprocessing"] = "None here: rows come from build-clip-data"
            run_config["weights"] = None
            run_config["normalisation"] = description["transform"]
        if build_head:
            run_config["trainable_parameters"] = sum(p.numel() for p in build_head().parameters())
        else:
            run_config["trainable_parameters"] = (model_input.shape[1] + 1) * len(classes)
        folder = output / name
        folder.mkdir()
        _write_json(folder / "config.json", run_config)
        print(f"\n{name}: {run_config['trainable_parameters']} trainable weights", flush=True)
        summary = fit_and_save(
            model_input,
            targets,
            dataset,
            folds,
            folder,
            config=config,
            context=context,
            run_config=run_config,
            encoder_hash=frames["encoder_sha256"],
            build_head=build_head,
            title={
                "images": TITLES[name],
                "positions": f"Position features: {TITLES[name]}",
                "clip-data": f"Clip data: {TITLES[name]}",
            }[inputs],
            model_limits=(
                *LIMITS[name],
                *((POSITION_LIMIT,) if inputs == "positions" else ()),
                *((CLIP_DATA_LIMIT,) if inputs == "clip-data" else ()),
                SHARED_LIMIT,
            ),
        )
        entry = {
            "folder": name,
            "model": run_config["model"],
            "trainable_parameters": run_config["trainable_parameters"],
            "macro_f1": summary["model"]["macro_f1"],
            "accuracy": summary["model"]["accuracy"],
            "optimistic_macro_f1": summary["optimistic"]["macro_f1"]
            if summary["optimistic"]
            else None,
            "majority": summary["majority"],
            "seed_ensemble": {
                key: summary["seed_ensemble"][key] for key in ("macro_f1", "accuracy", "clips")
            },
            "shuffled": None,
        }
        if shuffles:
            print(f"{name}: {shuffles} shuffled-label refits", flush=True)
            rows = shuffled_label_check(
                model_input,
                targets,
                folds,
                classes=classes,
                config=config,
                context=context,
                build_head=build_head,
                count=shuffles,
                epoch_mode=epoch_mode,
            )
            real = summary["model"]["macro_f1"]["mean"]
            entry["shuffled"] = {
                "count": shuffles,
                "macro_f1": _spread([r["macro_f1"] for r in rows]),
                "accuracy": _spread([r["accuracy"] for r in rows]),
                "at_least_real": int(sum(r["macro_f1"] >= real for r in rows)),
            }
            _write_json(
                folder / "shuffled-labels.json",
                {
                    "method": "Labels permuted within each match with numpy default_rng(permutation); "
                    "whole protocol refitted; scored against the shuffled labels; equal fold "
                    "average, then mean over seeds",
                    "real": {
                        key: summary["model"][key]["mean"] for key in ("macro_f1", "accuracy")
                    },
                    **entry["shuffled"],
                    "permutations": rows,
                },
            )
        comparison["models"][name] = entry
    _write_json(output / "summary.json", comparison)
    write_new_text(output / "report.md", comparison_report(output, comparison, dataset))
    print(f"Saved comparison: {output / 'report.md'}", flush=True)
    return comparison


@dataclass
class TemporalCheckpoint:
    """One saved temporal head, rebuilt and ready to score per-frame caches.

    ``encoder`` is the frozen ResNet-18 for image runs and None for position runs.
    ``predict`` takes the same per-frame dict as a saved cache (``features``,
    ``timestamps_s``, ``repeat_mask``, ``clip_ids``), with every position channel
    still present, and returns the head's logits.
    """

    name: str
    inputs: str
    head: nn.Module
    encoder: TacticBaseline | None
    saved: dict
    config: dict
    dropped_columns: list = field(default_factory=list)

    def predict(self, frames: dict) -> torch.Tensor:
        width = self.config.get("inputs", {}).get("width")
        if self.dropped_columns:
            keep = [
                i for i in range(frames["features"].shape[2]) if i not in set(self.dropped_columns)
            ]
            frames = {**frames, "features": frames["features"][:, :, keep]}
        if width is not None and frames["features"].shape[2] != width:
            raise ValueError(
                f"Expected {width} numbers per frame, got {frames['features'].shape[2]}"
            )
        inputs, _, _ = model_inputs(self.name, frames, self.saved["classes"])
        device = next(self.head.parameters()).device
        with torch.no_grad():
            return self.head(inputs.to(device)).cpu()


def _find_encoder(checkpoint: Path, encoder) -> Path:
    """The run's encoder.pt: next to the checkpoint, in its run folder, or beside the
    frame cache the run reused (named in the run's summary.json)."""
    if encoder is not None:
        return Path(encoder)
    model_folder, run_folder = checkpoint.parent, checkpoint.parent.parent
    candidates = [model_folder / "encoder.pt", run_folder / "encoder.pt"]
    summary = run_folder / "summary.json"
    if summary.is_file():
        cache = Path(
            json.loads(summary.read_text(encoding="utf-8"))
            .get("frame_features", "")
            .replace("\\", "/")
        )
        # A reused cache is recorded as given on the command line, from the repository root.
        candidates += [
            cache.parent / "encoder.pt",
            run_folder.parent.parent / cache.parent / "encoder.pt",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"No encoder.pt found for {checkpoint}; pass encoder=")


def load_temporal(checkpoint, *, context: RunContext, encoder=None) -> TemporalCheckpoint:
    """Restore any temporal run's head offline: mean, thirds, random-thirds or GRU, on
    image or position inputs.

    The model name and settings come from the checkpoint and its folder's config.json.
    Image runs also restore encoder.pt and refuse it unless its sha256 matches the one
    saved with the head. Position runs refuse a checkpoint made by a different detector
    or channel layout. Plan A runs still load with ``tactic_model.load_baseline``.
    """
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix, _, rest = saved["model"].partition("-")
    name = rest.removesuffix("-linear")
    if prefix == "clipdata":
        raise ValueError(f"{checkpoint}: clip-data heads cannot be reloaded here yet")
    if prefix not in ("resnet18", "positions") or name not in MODELS:
        raise ValueError(f"{checkpoint}: {saved['model']!r} is not a temporal model")
    if saved["classes"] != list(TACTIC_CLASSES):
        raise ValueError(f"{checkpoint}: class order does not match this code")
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if config["model"] != saved["model"] or config["manifest_sha256"] != saved["manifest_sha256"]:
        raise ValueError(f"{checkpoint}: config.json in this folder belongs to another run")
    inputs = "positions" if prefix == "positions" else "images"
    restored_encoder, dropped = None, []
    if inputs == "images":
        encoder_path = _find_encoder(checkpoint, encoder)
        if saved["encoder_sha256"] is None or file_sha256(encoder_path) != saved["encoder_sha256"]:
            raise ValueError(f"{encoder_path} does not match this checkpoint's encoder")
        restored_encoder = build_baseline(context=context, pretrained=False)
        restored_encoder.encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        restored_encoder.eval()
    else:
        if config["inputs"]["source"] != KIND:
            raise ValueError(f"{checkpoint}: made from different position features ({KIND!r})")
        dropped = (
            channel_columns(config["inputs"]["dropped_channels"])
            if config["inputs"]["dropped_channels"]
            else []
        )
    state = saved["classifier"]
    n_classes = len(saved["classes"])
    if name == "gru":
        settings = config["temporal"]
        head = TemporalGRU(
            state["project.weight"].shape[1],
            settings["projection"],
            settings["hidden"],
            n_classes,
            settings.get("dropout", 0.0),  # Runs saved before dropout existed had none,
            settings.get("readout", "last"),  # and read the last state.
        )
    else:
        head = nn.Linear(state["weight"].shape[1], n_classes)
    head.load_state_dict(state)
    return TemporalCheckpoint(
        name=name,
        inputs=inputs,
        head=head.to(context.device).eval(),
        encoder=restored_encoder,
        saved=saved,
        config=config,
        dropped_columns=dropped,
    )


def comparison_report(output, comparison, dataset) -> str:
    """Put the models side by side, from the files each model's run just saved."""
    names = list(comparison["models"])
    folds = {
        name: [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                (Path(output) / name).glob("fold-*.json"), key=lambda p: int(p.stem.split("-")[1])
            )
            if path.stem.split("-")[1].isdigit() and path.stem.count("-") == 1
        ]
        for name in names
    }
    first = folds[names[0]]
    mode = comparison["epoch_mode"]

    def spread_text(spread):
        return f"{spread['mean']:.3f} +/- {spread['std']:.3f}"

    if comparison.get("inputs") == "clip-data":
        info = comparison["clip_data"]
        inputs_line = (
            f"Inputs: clip data ({info['folder']}; {info['data_version']}, regions "
            f"{info['regions_version']}): {comparison['row_width']} numbers per row "
            f"({info['detector_channels']} detector channels + {CLIP_DATA_PCS} ResNet PCA numbers, "
            "PCA and scaling fitted on each fit's training clips only)."
        )
    else:
        inputs_line = (
            f"Inputs: {comparison.get('inputs', 'images')} ({comparison.get('feature_width', '?')} "
            "numbers per frame"
            + (
                f"; dropped {', '.join(comparison['dropped_channels'])}"
                if comparison.get("dropped_channels")
                else ""
            )
            + ")."
        )
    lines = [
        "# Temporal tactic models: development results",
        "",
        inputs_line,
        "",
        f"All models share the frame features, {len(first)} whole-match folds, {mode} epoch choice,",
        "seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row",
        "averages matches equally per seed first. "
        + (
            "`mean` is mean pooling on these rows."
            if comparison.get("inputs") == "clip-data"
            else "`mean` is Plan A, refitted on the same cache."
        ),
        "",
        "| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        entry = comparison["models"][name]
        optimistic = entry["optimistic_macro_f1"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {entry['trainable_parameters']} | "
            f"{spread_text(entry['macro_f1'])} | "
            f"{spread_text(entry['accuracy'])} | "
            f"{spread_text(optimistic) if optimistic else 'not run'} |"
        )
    majority = comparison["models"][names[0]]["majority"]
    lines += [
        f"| majority guess | 0 | {majority['macro_f1']:.3f} | {majority['accuracy']:.3f} | - |",
        "",
        "## Macro-F1 for each validation match",
        "",
        "| Validation match | Clips | "
        + " | ".join(DISPLAY_NAMES[name] for name in names)
        + " | Majority |",
        "|---|---:|" + "---:|" * (len(names) + 1),
    ]
    for index, fold in enumerate(first):
        match = next(
            r.match_folder for r in dataset.records if r.match_id == fold["validation_match_id"]
        )
        scores = " | ".join(
            spread_text(folds[name][index]["seed_spread"]["macro_f1"]) for name in names
        )
        lines.append(
            f"| {match} | {len(fold['validation_clip_ids'])} | {scores} | "
            f"{fold['majority']['macro_f1']:.3f} |"
        )
    lines += [
        "",
        "## F1 for each class",
        "",
        "Mean over validation matches and seeds. Clips are totals across matches.",
        "",
        "| Class | Clips | " + " | ".join(DISPLAY_NAMES[name] for name in names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    for label in dataset.classes:
        clips = sum(fold["validation_counts"][label] for fold in first)
        scores = " | ".join(
            f"{np.mean([run['metrics']['per_class'][label]['f1'] for fold in folds[name] for run in fold['runs']]):.3f}"
            for name in names
        )
        lines.append(f"| {label} | {clips} | {scores} |")
    shuffled = {n: comparison["models"][n]["shuffled"] for n in names}
    if any(shuffled.values()):
        lines += [
            "",
            "## Shuffled-label check",
            "",
            "Labels are shuffled within each match and the whole procedure is refitted.",
            "A sound procedure scores near chance on shuffled labels.",
            "",
            "| Model | Real macro-F1 | Shuffled macro-F1, mean (min to max) | Shuffles at or above real |",
            "|---|---:|---:|---:|",
        ]
        for name in names:
            check = shuffled[name]
            if check:
                f1 = check["macro_f1"]
                lines.append(
                    f"| {DISPLAY_NAMES[name]} | "
                    f"{comparison['models'][name]['macro_f1']['mean']:.3f} | "
                    f"{f1['mean']:.3f} ({f1['min']:.3f} to {f1['max']:.3f}) | "
                    f"{check['at_least_real']} of {check['count']} |"
                )
    clips = len(dataset)
    seed_sd = max(comparison["models"][name]["macro_f1"]["std"] for name in names)
    per_match = [
        max(f["seed_spread"]["macro_f1"]["mean"] for f in folds[name])
        - min(f["seed_spread"]["macro_f1"]["mean"] for f in folds[name])
        for name in names
    ]
    lines += [
        "",
        "## Reading these numbers",
        "",
        f"There are only {clips} clips from {len(first)} matches. Small differences between a model",
        "and the baseline are within noise and do not show that one is better:",
        "",
        f"- Changing only the random seed moves overall macro-F1 by up to {seed_sd:.3f} (largest SD above).",
        f"- The same model differs by up to {max(per_match):.3f} macro-F1 between validation matches.",
        "- One clip is several points of F1 for a class with two or three validation clips.",
        "",
        "Treat a gap smaller than those as a tie. These are development scores on the training",
        "matches, not test results; the test split is never opened here.",
        f"Repeated frames in this run: {comparison['repeated_frames']}.",
        "",
    ]
    return "\n".join(lines)
