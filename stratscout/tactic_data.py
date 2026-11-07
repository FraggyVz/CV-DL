"""Shared five-class clip inputs and whole-match validation, independent of model choice.

Imports perform no I/O. Construction reads only the supplied split manifest; videos
are decoded on indexing. Resizing is optional; scaling, normalisation, augmentation
and model training belong to the caller.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .config import TACTIC_CLASSES
from .context import RunContext


@dataclass(frozen=True)
class TacticClip:
    clip_id: str
    path: Path
    label: str
    match_id: str
    match_folder: str
    split: str
    start_s: float
    end_s: float
    duration_s: float
    frames: int
    definition_version: str
    source_video: str


@dataclass(frozen=True)
class MatchFold:
    """Indices refer to the supplied dataset, including any explicit exclusions."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_match_ids: tuple[str, ...]
    validation_match_id: str


def _read_manifest(path: Path, split: str, exclude_uncertain: bool):
    required = {
        "clip_id",
        "clip",
        "label",
        "match_id",
        "match_folder",
        "split",
        "start_s",
        "end_s",
        "duration_s",
        "frames",
        "status",
        "boundary_uncertain",
        "definition_version",
        "source_video",
    }
    records, excluded = [], []
    seen_ids, seen_paths, folder_ids, versions = set(), set(), {}, set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing manifest columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            prefix = f"{path}:{line} ({row.get('clip_id', '')})"
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"{prefix}: malformed manifest row")
            for key in ("clip_id", "match_id", "match_folder", "definition_version"):
                if not row[key].strip():
                    raise ValueError(f"{prefix}: {key} must not be empty")
            if row["split"] != split:
                raise ValueError(f"{prefix}: expected split {split!r}, got {row['split']!r}")
            if row["status"] != "reviewed" or row["label"] not in TACTIC_CLASSES:
                raise ValueError(f"{prefix}: only reviewed five-class labels are supported")
            flag = row["boundary_uncertain"].lower()
            if flag not in ("true", "false"):
                raise ValueError(f"{prefix}: boundary_uncertain must be True or False")
            if row["clip_id"] in seen_ids:
                raise ValueError(f"{prefix}: duplicate clip_id")
            seen_ids.add(row["clip_id"])
            previous = folder_ids.setdefault(row["match_folder"], row["match_id"])
            if previous != row["match_id"]:
                raise ValueError(f"{prefix}: one match folder has conflicting match IDs")
            versions.add(row["definition_version"])
            if len(versions) > 1:
                raise ValueError(f"{prefix}: mixed definition versions; review before combining")
            try:
                start, end, duration = (float(row[k]) for k in ("start_s", "end_s", "duration_s"))
                count = int(row["frames"])
            except ValueError as exc:
                raise ValueError(f"{prefix}: invalid timing or frame count") from exc
            if (
                not all(math.isfinite(v) for v in (start, end, duration))
                or start < 0
                or end <= start
                or duration <= 0
                or count < 1
                or not math.isclose(end - start, duration, abs_tol=0.011, rel_tol=0)
            ):
                raise ValueError(f"{prefix}: inconsistent timing or frame count")
            relative = Path(row["clip"])
            clip_path = (path.parent / relative).resolve()
            if (
                not row["clip"]
                or relative.is_absolute()
                or not clip_path.is_relative_to(path.parent)
            ):
                raise ValueError(f"{prefix}: clip must be a relative path within the split folder")
            if clip_path in seen_paths:
                raise ValueError(f"{prefix}: duplicate clip path")
            seen_paths.add(clip_path)
            if flag == "true":
                if not exclude_uncertain:
                    raise ValueError(
                        f"{prefix}: uncertain boundary; resolve the annotation or explicitly "
                        "use exclude_uncertain=True and report the exclusion"
                    )
                excluded.append(row["clip_id"])
                continue
            if not clip_path.is_file():
                raise FileNotFoundError(f"{prefix}: missing clip {clip_path}")
            records.append(
                TacticClip(
                    row["clip_id"],
                    clip_path,
                    row["label"],
                    row["match_id"],
                    row["match_folder"],
                    split,
                    start,
                    end,
                    duration,
                    count,
                    row["definition_version"],
                    row["source_video"],
                )
            )
    if not records:
        raise ValueError(f"{path}: no eligible clips")
    if excluded:
        warnings.warn(
            f"Excluded {len(excluded)} uncertain-boundary clips: {', '.join(excluded)}",
            UserWarning,
            stacklevel=3,
        )
    return tuple(records), tuple(excluded)


class TacticClipDataset(Dataset):
    """Read an exported split as RGB uint8 clips in (time, height, width, channels).

    Targets follow TACTIC_CLASSES, never the legacy CLASS_TO_IDX. Samples also
    contain relative timestamps, source frame indices and annotation duration.
    Metadata is for tracing/evaluation, not model input. Short clips repeat actual
    frames with repeated timestamps and a repeat_mask; no positions are invented.

    Native resolution is retained unless size=(height, width) is supplied. There
    is no implicit normalisation, augmentation, device transfer or random split.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        context: RunContext,
        n_frames: int = 16,
        size: tuple[int, int] | None = None,
        split: str = "train",
        exclude_uncertain: bool = False,
    ):
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("n_frames must be a positive integer")
        if size is not None and (
            len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size)
        ):
            raise ValueError("size must contain positive integer (height, width)")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'; validation comes from train folds")
        self.context = context
        self.manifest = Path(manifest).resolve()
        self.n_frames = n_frames
        self.size = tuple(size) if size is not None else None
        self.split = split
        self.classes = tuple(TACTIC_CLASSES)
        self.records, self.excluded_clip_ids = _read_manifest(
            self.manifest, split, exclude_uncertain
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        cap = cv2.VideoCapture(str(record.path))
        try:
            if not cap.isOpened():
                raise ValueError(f"{record.clip_id}: cannot open {record.path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{record.clip_id}: invalid video FPS")
            if not math.isfinite(count) or count != record.frames:
                raise ValueError(f"{record.clip_id}: video frame count differs from manifest")
            # The exporter includes both rounded endpoints: duration is (frames - 1) / fps.
            if abs((record.frames - 1) / fps - record.duration_s) > 1 / fps + 0.011:
                raise ValueError(f"{record.clip_id}: video timing differs from manifest")
            indices = np.rint(np.linspace(0, record.frames - 1, self.n_frames)).astype(np.int64)
            wanted = set(indices.tolist())
            selected = {}
            # Sequential decoding avoids inaccurate random seeks and catches truncated clips.
            for frame_index in range(record.frames):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"{record.clip_id}: cannot decode frame {frame_index}")
                if frame_index in wanted:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if self.size is not None:
                        height, width = self.size
                        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                    selected[frame_index] = rgb
        finally:
            cap.release()
        return {
            "frames": np.stack([selected[i] for i in indices]),
            "target": self.classes.index(record.label),
            "timestamps_s": indices.astype(np.float64) / fps,
            "frame_indices": indices,
            "repeat_mask": np.concatenate(([False], indices[1:] == indices[:-1])),
            "duration_s": record.duration_s,
            "fps": fps,
            "metadata": {
                "clip_id": record.clip_id,
                "match_id": record.match_id,
                "match_folder": record.match_folder,
                "split": record.split,
                "definition_version": record.definition_version,
                "source_video": record.source_video,
                "source_start_s": record.start_s,
                "source_end_s": record.end_s,
            },
        }


def whole_match_folds(dataset: TacticClipDataset) -> tuple[MatchFold, ...]:
    """Leave one training match out at a time; never read a neighbouring test manifest.

    All clips/variants with the same match_id stay together. With Furia and Spirit
    this yields the two swaps. Reinitialise the model for each fold; any fitted
    preprocessing or class weights must use that fold's training indices only.
    """
    if dataset.split != "train" or any(r.split != "train" for r in dataset.records):
        raise ValueError("validation folds accept only the train split, never test")
    match_ids = sorted({r.match_id for r in dataset.records})
    if len(match_ids) < 2:
        raise ValueError("validation folds need at least two training matches")
    folds = []
    for held_out in match_ids:
        train = tuple(i for i, r in enumerate(dataset.records) if r.match_id != held_out)
        validation = tuple(i for i, r in enumerate(dataset.records) if r.match_id == held_out)
        folds.append(
            MatchFold(train, validation, tuple(m for m in match_ids if m != held_out), held_out)
        )
    return tuple(folds)
