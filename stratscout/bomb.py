"""Bomb: colour-then-shape C4 badge detection and dropped-bomb candidates from minimap videos.

Measured on the 494x500 spectator radar, where the C4 badge is an orange rectangle drawn
on top of the carrier's icon and players are numbered circles. Carried and dropped badges
look identical, so state comes from the carrier's rim and direction arrow showing at the
badge edge, and from motion. Video-only: no demo truth is used here.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .segmentation import blob_shape, looks_like_bomb
from .storage import write_new_text

# Spectator-radar T yellow measured on both 14 October matches (badge orange is ~16).
SPECTATOR_T_HUE = 27.0


def find_bomb(
    frame: np.ndarray,
    shape_only: bool = False,
    *,
    context,
) -> Optional[Dict[str, float]]:
    """RGB frame -> the most badge-like blob {x, y, area, aspect, extent, corner_fill, hue}
    in pixels, or None.

    By default the frame is first split by colour: only badge-orange pixels (Config.bomb_hue)
    form blobs, so a badge touching its yellow carrier stays a separate rectangle. With
    shape_only, every saturated pixel forms blobs and the badge often merges with the
    carrier (26-32% of frames found on two matches, against 94-96% with the colour split)."""
    cfg = context.config
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if shape_only:
        mask = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    else:
        hue = hsv[:, :, 0]
        mask = (
            (hue >= cfg.bomb_hue[0])
            & (hue <= cfg.bomb_hue[1])
            & (hsv[:, :, 1] >= cfg.bomb_sv_min)
            & (hsv[:, :, 2] >= cfg.bomb_sv_min)
        )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if not shape_only:
        # Opening strips thin anti-aliased fringes (e.g. a fading yellow death marker) that
        # fall inside the orange band and would otherwise widen the badge's outline.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x0, y0, bw, bh, a = (int(v) for v in stats[i])
        if not cfg.bomb_area[0] <= a <= cfg.bomb_area[1]:
            continue
        shape = blob_shape(lab[y0 : y0 + bh, x0 : x0 + bw] == i)
        if not looks_like_bomb(a, shape, context=context):
            continue
        hue = float(np.median(hsv[:, :, 0][lab == i]))
        found = {
            "x": float(cents[i][0]),
            "y": float(cents[i][1]),
            "box": (x0, y0, bw, bh),
            "area": a,
            "hue": hue,
            **shape,
        }
        if best is None or shape["corner_fill"] > best["corner_fill"]:
            best = found
    return best


def carrier_rim(
    frame: np.ndarray,
    bomb: Dict,
    band: int = 4,
    t_hue: float = SPECTATOR_T_HUE,
    hue_tol: float = 4.0,
    *,
    context,
) -> Dict[str, int]:
    """T-yellow and white pixels in a thin band around the badge rectangle.

    A carried badge is drawn over its carrier, whose icon edge (yellow) and direction arrow
    (white) show just outside the badge. A dropped badge sits on bare map. Measured on one
    match: 99% of moving (carried) badge frames had >= 4 such pixels; most still, lone
    badges had none. Teammates further away fall outside the band."""
    cfg = context.config
    h, w = frame.shape[:2]
    x0, y0, bw, bh = bomb["box"]
    ox0, oy0 = max(x0 - band, 0), max(y0 - band, 0)
    ox1, oy1 = min(x0 + bw + band, w), min(y0 + bh + band, h)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[oy0:oy1, ox0:ox1]), cv2.COLOR_RGB2HSV)
    ring = np.ones(hsv.shape[:2], bool)
    ring[y0 - oy0 : y0 - oy0 + bh, x0 - ox0 : x0 - ox0 + bw] = False
    hue = hsv[:, :, 0].astype(np.float32)
    d = np.minimum(np.abs(hue - t_hue), 180 - np.abs(hue - t_hue))
    vivid = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    white = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)
    return {
        "yellow": int((ring & vivid & (d <= hue_tol)).sum()),
        "white": int((ring & white).sum()),
    }


def carrier_visible(rim: Dict[str, int], min_pixels: int = 4) -> bool:
    return rim["yellow"] >= min_pixels or rim["white"] >= min_pixels


def scan_video(
    path: Path,
    band: int = 4,
    t_hue: float = SPECTATOR_T_HUE,
    shape_only: bool = False,
    *,
    context,
) -> Tuple[List[Dict], float]:
    """Prepared minimap video -> one row per frame and the video fps. Prepared exports are
    already normal game speed, so seconds = frame / fps from the trim start."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    rows: List[Dict] = []
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            b = find_bomb(rgb, shape_only, context=context)
            row = {"frame": i, "seconds": round(i / fps, 3), "x": None, "y": None}
            row.update(rim_yellow=0, rim_white=0, carried=0)
            if b is not None:
                rim = carrier_rim(rgb, b, band, t_hue, context=context)
                row.update(x=round(b["x"], 1), y=round(b["y"], 1))
                row.update(rim_yellow=rim["yellow"], rim_white=rim["white"])
                row["carried"] = int(carrier_visible(rim))
            rows.append(row)
            i += 1
    finally:
        cap.release()
    return rows, fps


def dropped_candidates(
    rows: List[Dict],
    fps: float,
    min_seconds: float = 2.0,
    still_px: float = 3.0,
    max_gap_frames: int = 2,
) -> List[Dict]:
    """Runs where the badge stays within still_px of where the run began with no carrier
    showing at its edge, lasting at least min_seconds. Short detection gaps are bridged.

    These are review candidates, not labels: a planted bomb or a frozen spectator view can
    match the same rule."""
    runs: List[Dict] = []
    cur = None
    for r in rows:
        lone = r["x"] is not None and not r["carried"]
        if lone and cur and r["frame"] - cur["end"] <= max_gap_frames + 1:
            if abs(r["x"] - cur["x"]) <= still_px and abs(r["y"] - cur["y"]) <= still_px:
                cur["end"] = r["frame"]
                continue
        if lone:
            if cur:
                runs.append(cur)
            cur = {"start": r["frame"], "end": r["frame"], "x": r["x"], "y": r["y"]}
        elif cur and r["frame"] - cur["end"] > max_gap_frames:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        frames = run["end"] - run["start"] + 1
        if frames / fps >= min_seconds:
            out.append(
                {
                    "start_frame": run["start"],
                    "end_frame": run["end"],
                    "start_seconds": round(run["start"] / fps, 2),
                    "end_seconds": round(run["end"] / fps, 2),
                    "x": run["x"],
                    "y": run["y"],
                }
            )
    return out


class BombStateTracker:
    """Frame-by-frame version of the dropped_candidates rule for live display.

    States: "carried" (carrier's rim or arrow at the badge edge), "alone" (no carrier, not
    yet still for long enough), "dropped" (still and alone for min_seconds) and "missing"
    (no badge found)."""

    def __init__(
        self,
        fps: float,
        min_seconds: float = 2.0,
        still_px: float = 3.0,
        max_gap_frames: int = 2,
    ):
        self.fps = fps
        self.min_frames = int(round(min_seconds * fps))
        self.still_px = still_px
        self.max_gap_frames = max_gap_frames
        self.reset()

    def reset(self) -> None:
        self.anchor = None
        self.still_frames = 0
        self.gap = 0

    def update(self, x: Optional[float], y: Optional[float], carried: bool, frames: int = 1) -> str:
        """Advance by frames (more than 1 when playback skips frames) and return the state."""
        if x is None:
            self.gap += frames
            if self.gap > self.max_gap_frames:
                self.reset()
            return "missing"
        self.gap = 0
        if carried:
            self.anchor, self.still_frames = None, 0
            return "carried"
        if self.anchor is not None and max(abs(x - self.anchor[0]), abs(y - self.anchor[1])) <= (
            self.still_px
        ):
            self.still_frames += frames
        else:
            self.anchor, self.still_frames = (x, y), frames
        return "dropped" if self.still_frames >= self.min_frames else "alone"

    def still_seconds(self) -> float:
        return self.still_frames / self.fps


def _csv(rows: List[Dict], fields: List[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_bomb_scan(
    video: Path,
    output: Path,
    band: int = 4,
    min_seconds: float = 2.0,
    t_hue: float = SPECTATOR_T_HUE,
    shape_only: bool = False,
    *,
    context,
) -> Dict:
    """Scan one prepared video and write track.csv, candidates.csv, summary.json and an
    enlarged crop of each candidate into a new output directory."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new folder: {output}")
    rows, fps = scan_video(video, band, t_hue, shape_only, context=context)
    runs = dropped_candidates(rows, fps, min_seconds)
    write_new_text(
        output / "track.csv",
        _csv(rows, ["frame", "seconds", "x", "y", "rim_yellow", "rim_white", "carried"]),
    )
    fields = ["start_frame", "end_frame", "start_seconds", "end_seconds", "x", "y", "crop"]
    cap = cv2.VideoCapture(str(video))
    try:
        for k, run in enumerate(runs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, (run["start_frame"] + run["end_frame"]) // 2)
            ok, bgr = cap.read()
            run["crop"] = ""
            if ok:
                x, y = int(run["x"]), int(run["y"])
                crop = bgr[max(y - 40, 0) : y + 40, max(x - 50, 0) : x + 50]
                crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                name = f"candidate-{k:03d}-{run['start_seconds']:.0f}s.png"
                ok, png = cv2.imencode(".png", crop)
                if ok:
                    with (output / name).open("xb") as stream:
                        stream.write(png.tobytes())
                    run["crop"] = name
    finally:
        cap.release()
    write_new_text(output / "candidates.csv", _csv(runs, fields))
    found = sum(r["x"] is not None for r in rows)
    summary = {
        "video": str(video),
        "fps": fps,
        "frames": len(rows),
        "frames_with_badge": found,
        "dropped_candidates": len(runs),
        "parameters": {
            "rim_band_px": band,
            "min_seconds": min_seconds,
            "t_hue": t_hue,
            "shape_only": shape_only,
            "bomb_hue": None if shape_only else list(context.config.bomb_hue),
            "bomb_area": list(context.config.bomb_area),
            "bomb_min_aspect": context.config.bomb_min_aspect,
            "bomb_min_extent": context.config.bomb_min_extent,
            "bomb_min_corner_fill": context.config.bomb_min_corner_fill,
        },
        "claims": "Unreviewed video-only candidates. Not labels and not detector accuracy.",
    }
    write_new_text(output / "summary.json", json.dumps(summary, indent=2))
    return summary
