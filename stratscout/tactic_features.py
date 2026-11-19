"""Per-frame features for exported tactical clips, for the tactic sequence models.

Two independent feature sources live here; each model uses one of them.

**Detector features (TCN, clip-data, predict-tactics).** Reads a clip dataset written by
export-segments (<dataset>/<split>/manifest.csv and clips/<class>/*.mp4), runs the
spectator-radar detector on every frame and caches the detections, then turns them into
fixed-size per-frame feature vectors for a sequence model.

Features describe where each team is, not who is who: a soft occupancy grid per team, team
counts, centroids and spread, the C4 badge and elapsed seconds. No tracker is used, so slot
swaps and frozen dead players cannot enter the features. Detections are video-only; labels
are read for training targets and never reach the detector.

Each match's background (empty map) is the median of frames sampled across all of that
match's clips, because players stand in different places in different clips. It uses
pixels only, never labels.

**Frozen ResNet-18 frame features (tactic-baseline, tactic-temporal).** Extracted once and
shared by every model there. The cache keeps each sampled frame's feature, time and repeat
flag, so a model can pool, order or time them as it likes. Averaging a clip's frames gives
exactly the Plan A clip feature. A cache is reused only for the same manifest, sampling,
feature source and clip files. Nothing here fits statistics or opens another split.

Importing this module reads nothing.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import TACTIC_CLASSES
from .context import RunContext
from .storage import save_checkpoint
from .tactic_data import TacticClipDataset
from .tactic_model import WEIGHTS, build_baseline, file_sha256, normalise_frames

# Bump when detection settings change so stale caches are not reused.
DETECTION_VERSION = "spectator-split-1"
DEFAULT_GRID = 8
N_SCALARS = 13  # counts 2, centroids 4, spreads 2, bomb 4, elapsed 1


def load_dataset(root) -> List[Dict]:
    """Rows of <root>/train and <root>/test manifest.csv with absolute clip paths."""
    root = Path(root)
    rows = []
    for split in ("train", "test"):
        manifest = root / split / "manifest.csv"
        if not manifest.exists():
            raise ValueError(f"no {manifest}; extract the dataset as described in the README")
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["label"] not in TACTIC_CLASSES:
                    raise ValueError(f"{manifest}: unknown class {row['label']!r}")
                flag = row.get("boundary_uncertain", "False").strip().lower()
                if flag not in ("true", "false"):
                    raise ValueError(f"{manifest}: boundary_uncertain must be True or False")
                rows.append(
                    {
                        "clip_id": row["clip_id"],
                        "split": split,
                        "label": row["label"],
                        "y": TACTIC_CLASSES.index(row["label"]),
                        "match": row["match_folder"],
                        "duration_s": float(row["duration_s"]),
                        "uncertain": flag == "true",
                        "path": root / split / row["clip"],
                    }
                )
    if not rows:
        raise ValueError(f"{root} has no clips")
    missing = [str(r["path"]) for r in rows if not r["path"].exists()]
    if missing:
        raise ValueError(f"{len(missing)} clips listed but missing, e.g. {missing[0]}")
    return rows


def match_background(paths: List[Path], per_clip: int = 8) -> np.ndarray:
    """RGB median of frames sampled evenly across one match's clips: the empty map."""
    import cv2

    frames = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(n - 1, 0), per_clip).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        raise ValueError("no readable frames for the background")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_clip(path: Path, background: np.ndarray) -> Dict[str, np.ndarray]:
    """Every frame of one clip -> players (frame, x, y, team, inferred) and bomb
    (present, x, y, carried) per frame, with x, y normalised to the crop."""
    import cv2

    from .bomb import SPECTATOR_T_HUE, carrier_visible
    from .live_preview import detect, spectator_context

    context = spectator_context()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    players, bombs, k = [], [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            d = detect(rgb, 4, SPECTATOR_T_HUE, background, context=context)
            for x, y, team in d["players"]:
                players.append((k, x / w, y / h, 0.0 if team == "T" else 1.0, 0.0))
            if d["carrier"] is not None:
                cx, cy = d["carrier"]
                players.append((k, cx / w, cy / h, 0.0, 1.0))
            b = d["bomb"]
            if b is None:
                bombs.append((0.0, 0.0, 0.0, 0.0))
            else:
                carried = float(carrier_visible(d["rim"]))
                bombs.append((1.0, b["x"] / w, b["y"] / h, carried))
            k += 1
    finally:
        cap.release()
    if not k:
        raise ValueError(f"{path} has no frames")
    return {
        "players": np.asarray(players, np.float32).reshape(-1, 5),
        "bomb": np.asarray(bombs, np.float32).reshape(-1, 4),
        "fps": np.float32(fps),
        "n_frames": np.int32(k),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def cached_detections(rows: List[Dict], cache_dir, log=print) -> Dict[str, Dict[str, np.ndarray]]:
    """Detections for every clip, read from <cache_dir>/<clip_id>.npz when the clip file
    and detection version match, otherwise computed (background per match) and saved."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for row in rows:
        file = cache_dir / f"{row['clip_id']}.npz"
        digest = _digest(row["path"])
        if file.exists():
            with np.load(file) as data:
                if str(data["version"]) == DETECTION_VERSION and str(data["digest"]) == digest:
                    out[row["clip_id"]] = {k: data[k] for k in ("players", "bomb", "fps")}
                    continue
        todo.append((row, file, digest))
    if not todo:
        return out
    by_match = defaultdict(list)
    for row in rows:
        by_match[row["match"]].append(row["path"])
    backgrounds: Dict[str, np.ndarray] = {}
    for i, (row, file, digest) in enumerate(todo, 1):
        if row["match"] not in backgrounds:
            log(f"  background for {row['match']} from {len(by_match[row['match']])} clips")
            backgrounds[row["match"]] = match_background(by_match[row["match"]])
        log(f"  detecting {i}/{len(todo)}: {row['clip_id']}")
        det = detect_clip(row["path"], backgrounds[row["match"]])
        tmp = file.with_name(file.stem + ".tmp.npz")
        np.savez_compressed(tmp, version=DETECTION_VERSION, digest=digest, **det)
        tmp.replace(file)
        out[row["clip_id"]] = {k: det[k] for k in ("players", "bomb", "fps")}
    return out


def feature_dim(grid: int = DEFAULT_GRID) -> int:
    return 2 * grid * grid + N_SCALARS


def clip_features(
    det: Dict[str, np.ndarray],
    *,
    step: int = 2,
    offset: int = 0,
    grid: int = DEFAULT_GRID,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 0.0,
    drop: float = 0.0,
) -> np.ndarray:
    """Detections -> (steps, feature_dim) float32, keeping every step-th frame from offset.

    Per step: a bilinear occupancy grid per team (T then CT, each summing to the team's
    detected count / 5), counts / 5, centroids, spreads, bomb present/x/y/carried and
    elapsed seconds / 30. Elapsed time is real time, so a slow attack stays slow.
    With rng, jitter (normalised units) and drop (fraction of detections) augment."""
    if grid < 2:
        raise ValueError("grid must be at least 2")
    players, bomb, fps = det["players"], det["bomb"], float(det["fps"])
    frames = np.arange(offset, len(bomb), step)
    if not len(frames):
        frames = np.array([0])
    if rng is not None and drop > 0 and len(players):
        players = players[rng.random(len(players)) >= drop]
    if rng is not None and jitter > 0 and len(players):
        players = players.copy()
        players[:, 1:3] += rng.normal(0, jitter, (len(players), 2)).astype(np.float32)
    feats = np.zeros((len(frames), feature_dim(grid)), np.float32)
    order = np.argsort(players[:, 0], kind="stable") if len(players) else np.array([], int)
    players = players[order]
    starts = np.searchsorted(players[:, 0], frames, "left") if len(players) else None
    ends = np.searchsorted(players[:, 0], frames, "right") if len(players) else None
    g2 = grid * grid
    for i, f in enumerate(frames):
        row = feats[i]
        if starts is not None:
            here = players[starts[i] : ends[i]]
            for team in (0, 1):
                pts = here[here[:, 3] == team][:, 1:3]
                if not len(pts):
                    continue
                _splat(row[team * g2 : (team + 1) * g2], pts, grid)
                base = 2 * g2
                row[base + team] = len(pts) / 5.0
                row[base + 2 + 2 * team : base + 4 + 2 * team] = pts.mean(0)
                row[base + 6 + team] = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
        row[2 * g2 + 8 : 2 * g2 + 12] = bomb[f]
        row[2 * g2 + 12] = f / fps / 30.0
    return feats


def _splat(cells: np.ndarray, pts: np.ndarray, grid: int) -> None:
    """Add each point to the 4 nearest cell centres with bilinear weights of 1/5 in total."""
    g = np.clip(pts * grid - 0.5, 0, grid - 1)
    i0 = np.minimum(np.floor(g).astype(int), grid - 2) if grid > 1 else np.zeros_like(g, int)
    frac = g - i0
    for dx in (0, 1):
        for dy in (0, 1):
            wx = frac[:, 0] if dx else 1 - frac[:, 0]
            wy = frac[:, 1] if dy else 1 - frac[:, 1]
            ix = np.minimum(i0[:, 0] + dx, grid - 1)
            iy = np.minimum(i0[:, 1] + dy, grid - 1)
            np.add.at(cells, iy * grid + ix, wx * wy / 5.0)


# --- Frozen ResNet-18 frame features (tactic-baseline and tactic-temporal) ---
def extract_frame_features(dataset: TacticClipDataset, output, *, context: RunContext) -> dict:
    """Encode every sampled frame of every clip; save the encoder as ``output/encoder.pt``.

    Returns ``features`` (clips, frames, 512) in dataset order, the per-clip ``samples``
    records and ``encoder_sha256``. Frame order is time order within each clip.
    """
    output = Path(output)
    model = build_baseline(context=context).eval()
    save_checkpoint(
        {k: v.cpu() for k, v in model.encoder.state_dict().items()}, output / "encoder.pt"
    )
    features, samples = [], []
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        tensor = normalise_frames(sample["frames"], context=context)
        features.append(model.encode_frames(tensor.unsqueeze(0)).cpu())
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
    return {
        "features": torch.cat(features),
        "samples": samples,
        "encoder_sha256": file_sha256(output / "encoder.pt"),
    }


def _identity(dataset: TacticClipDataset, weights=WEIGHTS) -> dict:
    """What a cache was made from. ``weights`` names the feature source: the ResNet
    weights, or the position detector (see ``tactic_positions.KIND``)."""
    return {
        "clip_ids": [r.clip_id for r in dataset.records],
        "manifest_sha256": file_sha256(dataset.manifest),
        "n_frames": dataset.n_frames,
        "size": list(dataset.size) if dataset.size else None,
        "weights": weights,
    }


def save_frame_features(path, frames: dict, dataset: TacticClipDataset, weights=WEIGHTS) -> None:
    """Write the per-frame cache with the times, repeat flags and inputs it was made from."""
    samples = frames["samples"]
    save_checkpoint(
        {
            "features": frames["features"],
            "timestamps_s": torch.tensor([s["timestamps_s"] for s in samples], dtype=torch.float64),
            "repeat_mask": torch.tensor([s["repeat_mask"] for s in samples], dtype=torch.bool),
            "samples": samples,
            "encoder_sha256": frames["encoder_sha256"],
            **({"feature_names": frames["feature_names"]} if "feature_names" in frames else {}),
            **_identity(dataset, weights),
        },
        path,
    )


def load_frame_features(path, dataset: TacticClipDataset, weights=WEIGHTS) -> dict:
    """Reuse a saved cache only if it was made from this manifest, clip order, sampling,
    feature source and the same clip files.

    The manifest hash alone misses a clip re-exported under the same name, so every
    clip's current sha256 is checked against the one saved when it was encoded.
    """
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    for key, expected in _identity(dataset, weights).items():
        if saved.get(key) != expected:
            raise ValueError(f"{path}: cached {key} does not match this run; extract again")
    if saved["features"].shape[:2] != (len(dataset), dataset.n_frames):
        raise ValueError(f"{path}: cached features have the wrong shape; extract again")
    for record, sample in zip(dataset.records, saved["samples"], strict=True):
        if sample.get("clip_sha256") != file_sha256(record.path):
            raise ValueError(
                f"{path}: clip {record.clip_id} has changed since it was cached; extract again"
            )
    return saved
