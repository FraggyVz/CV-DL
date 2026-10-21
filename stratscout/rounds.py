"""Rounds: estimate round starts from a prepared minimap video.

Every round begins with a freeze time (~20 s) in which all Ts stand in T spawn and all CTs
in CT spawn. Frames where nearly all team colour sits in both spawn boxes are grouped into
freeze runs; each run starts a round. Longer runs are timeouts or halftime. A recording
that begins mid-round gets a partial first round. These are video estimates, not demo truth.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# Spawn centres as fractions of the crop, measured on the 494x500 spectator-radar crops.
T_SPAWN = (0.870, 0.350)
CT_SPAWN = (0.294, 0.702)
SPAWN_HALF_SIZE = 0.071  # half the spawn box side, as a fraction of crop width
T_HUE, CT_HUE = 27, 94


def video_background(cap: cv2.VideoCapture, total: int, samples: int = 60) -> np.ndarray:
    """Median of frames spread across the match (BGR): the map without players."""
    frames = []
    for i in np.linspace(0, max(total - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def in_spawns(
    bgr: np.ndarray,
    background: np.ndarray,
    min_pixels: int = 300,
    min_fraction: float = 0.8,
) -> bool:
    """True when both teams' moving colour is concentrated in their own spawn boxes."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    moving = np.abs(bgr.astype(np.int16) - background.astype(np.int16)).sum(-1) > 40
    vivid = (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 120) & moving
    r = int(SPAWN_HALF_SIZE * w)
    for target, (fx, fy), tol in ((T_HUE, T_SPAWN, 4), (CT_HUE, CT_SPAWN, 6)):
        team = vivid & (np.abs(hue - target) <= tol)
        cx, cy = int(fx * w), int(fy * h)
        inside = int(team[max(cy - r, 0) : cy + r, max(cx - r, 0) : cx + r].sum())
        if inside < min_pixels or inside < min_fraction * team.sum():
            return False
    return True


def rounds_from_flags(
    flags: Sequence[Tuple[float, bool]],
    duration: float,
    bridge_seconds: float = 10.0,
    min_freeze: float = 3.0,
    min_gap: float = 40.0,
    partial_after: float = 30.0,
) -> List[Dict]:
    """(seconds, in_spawns) samples -> rounds with start, freeze length and end seconds.

    Freeze samples closer than bridge_seconds join one run; runs shorter than min_freeze
    are ignored; a run within min_gap of the previous start belongs to the same round.
    If the first freeze starts later than partial_after, the footage before it is a round
    that began before the recording."""
    runs: List[List[float]] = []
    for t, ok in flags:
        if not ok:
            continue
        if runs and t - runs[-1][1] <= bridge_seconds:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    starts: List[List[float]] = []
    for a, b in runs:
        if b - a < min_freeze:
            continue
        if starts and a - starts[-1][0] <= min_gap:
            starts[-1][1] = b
        else:
            starts.append([a, b])
    rounds = []
    if not starts or starts[0][0] > partial_after:
        rounds.append({"start": 0.0, "freeze": 0.0, "partial": True})
    rounds += [{"start": a, "freeze": b - a, "partial": False} for a, b in starts]
    for k, r in enumerate(rounds):
        r["round"] = k + 1
        r["end"] = rounds[k + 1]["start"] if k + 1 < len(rounds) else duration
    return rounds


def detect_rounds(video: Path, sample_every: int = 2) -> List[Dict]:
    """Scan a prepared minimap video (a 5 fps variant is fastest) and estimate its rounds."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        background = video_background(cap, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flags, i = [], 0
        while cap.grab():
            if i % sample_every == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    flags.append((i / fps, in_spawns(bgr, background)))
            i += 1
    finally:
        cap.release()
    return rounds_from_flags(flags, duration=i / fps)


def cached_rounds(video: Path, cache_dir: Path = Path("runs/round-cache")) -> List[Dict]:
    """detect_rounds, cached by the video's path, size and modification time."""
    video = Path(video)
    st = video.stat()
    key = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    path = Path(cache_dir) / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rounds"]
    rounds = detect_rounds(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"video": str(video), "rounds": rounds}, indent=2), "utf-8")
    return rounds


def round_at(rounds: List[Dict], seconds: float) -> Dict:
    """The round containing seconds (the first round if before every start)."""
    current = rounds[0]
    for r in rounds:
        if r["start"] <= seconds:
            current = r
    return current
