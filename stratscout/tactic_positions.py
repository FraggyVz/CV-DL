"""Per-frame player-position features from the minimap detector, for the tactic models.

Each sampled frame becomes counts on a fixed 8x8 grid over the radar crop, with the
loader's frames, times and repeat flags, so the cache lines up with frame-features.pt.
Observed icons, the inferred bomb carrier and the bomb badge stay in separate channels;
missing players are never filled in, and there is no tracker identity. The detector
and its settings are the ones checked in docs/detector-check.md.
"""

from __future__ import annotations

import numpy as np
import torch

from .context import RunContext
from .tactic_data import TacticClipDataset
from .tactic_model import file_sha256

GRID = 8
CHANNELS = ("t_observed", "ct_observed", "t_inferred_carrier", "bomb_badge")
TOTALS = ("t_seen", "ct_seen", "t_inferred_present", "bomb_badge_present")
DETECTOR = "colour + icon split (live_preview.detect, spectator settings, rim band 4)"
KIND = f"positions: {DETECTOR}; {GRID}x{GRID} grid; channels {', '.join(CHANNELS)}"


def feature_names() -> list[str]:
    cells = [
        f"{channel}[{row},{col}]"
        for channel in CHANNELS
        for row in range(GRID)
        for col in range(GRID)
    ]
    return cells + list(TOTALS)


def channel_columns(names) -> list[int]:
    """Columns of the named channels, with their totals, e.g. ("bomb_badge",)."""
    unknown = set(names) - set(CHANNELS)
    if unknown:
        raise ValueError(f"Unknown channels {sorted(unknown)}; choose from {', '.join(CHANNELS)}")
    size = GRID * GRID
    columns = []
    for name in names:
        index = CHANNELS.index(name)
        columns += list(range(index * size, (index + 1) * size))
        columns.append(len(CHANNELS) * size + index)
    return sorted(columns)


def _cell(x, y, width, height) -> tuple[int, int]:
    row = min(max(int(y * GRID / height), 0), GRID - 1)
    col = min(max(int(x * GRID / width), 0), GRID - 1)
    return row, col


def frame_positions(found: dict, width: int, height: int) -> np.ndarray:
    """One detector result -> (channels*GRID*GRID + 4) counts. Nothing is invented:
    an undetected player just lowers the counts; the inferred carrier is not a T count."""
    grid = np.zeros((len(CHANNELS), GRID, GRID), np.float32)
    for x, y, team in found["players"]:
        grid[
            (
                CHANNELS.index("t_observed") if team == "T" else CHANNELS.index("ct_observed"),
                *_cell(x, y, width, height),
            )
        ] += 1
    if found["carrier"] is not None:
        grid[(CHANNELS.index("t_inferred_carrier"), *_cell(*found["carrier"], width, height))] += 1
    if found["bomb"] is not None:
        grid[
            (
                CHANNELS.index("bomb_badge"),
                *_cell(found["bomb"]["x"], found["bomb"]["y"], width, height),
            )
        ] += 1
    totals = grid.sum(axis=(1, 2))
    totals[2:] = np.minimum(totals[2:], 1)  # Presence flags.
    return np.concatenate([grid.ravel(), totals])


def extract_position_features(dataset: TacticClipDataset, *, context: RunContext) -> dict:
    """Run the detector on every sampled frame of every clip, at the clips' native size.

    The detector also needs the empty-map background, estimated from each clip's whole
    source match video as in bomb-preview. Returns the same keys as
    ``extract_frame_features``, plus the feature names; ``encoder_sha256`` is None.
    """
    import cv2

    from .bomb import SPECTATOR_T_HUE
    from .live_preview import detect, spectator_context
    from .rounds import video_background

    if dataset.size is not None:
        raise ValueError("Position features need the clips at native size (size=None)")
    detector_context = spectator_context()
    backgrounds = {}
    features, samples = [], []
    for index, record in enumerate(dataset.records):
        source = record.source_video
        if source not in backgrounds:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"{record.clip_id}: cannot open source video {source}")
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                backgrounds[source] = cv2.cvtColor(video_background(cap, total), cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
        background = backgrounds[source]
        sample = dataset[index]
        frames = sample["frames"]
        if frames.shape[1:3] != background.shape[:2]:
            raise ValueError(f"{record.clip_id}: clip and source video sizes differ")
        height, width = frames.shape[1:3]
        rows = [
            frame_positions(
                detect(frame, 4, SPECTATOR_T_HUE, background, context=detector_context),
                width,
                height,
            )
            for frame in frames
        ]
        features.append(torch.from_numpy(np.stack(rows)).unsqueeze(0))
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
            print(f"Detected positions: {index + 1}/{len(dataset)}", flush=True)
    return {
        "features": torch.cat(features),
        "samples": samples,
        "encoder_sha256": None,
        "feature_names": feature_names(),
    }


def native_dataset(dataset: TacticClipDataset) -> TacticClipDataset:
    """The same clips and sampling at native size, as the detector needs."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Exclusions were already reported.
        native = TacticClipDataset(
            dataset.manifest,
            context=dataset.context,
            n_frames=dataset.n_frames,
            size=None,
            split=dataset.split,
            exclude_uncertain=bool(dataset.excluded_clip_ids),
        )
    if [r.clip_id for r in native.records] != [r.clip_id for r in dataset.records]:
        raise ValueError("Native-size dataset has different clips; check the exclusions")
    return native
