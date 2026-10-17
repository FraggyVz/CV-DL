"""Bomb: shape-based C4 badge detection and dropped-bomb candidates from minimap videos.

Measured on the 494x500 spectator radar, where the C4 badge is an orange rectangle beside
the carrier and players are numbered circles. Carried and dropped badges look identical,
so state comes from motion and T proximity. Video-only: no demo truth is used here.
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
    hue_band: Optional[Tuple[float, float]] = None,
    *,
    context,
) -> Optional[Dict[str, float]]:
    """RGB frame -> the most badge-like blob {x, y, area, aspect, extent, corner_fill, hue}
    in pixels, or None. Shape alone decides unless hue_band (OpenCV hue, 0-180) is given."""
    cfg = context.config
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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
        if hue_band is not None and not hue_band[0] <= hue <= hue_band[1]:
            continue
        found = {"x": float(cents[i][0]), "y": float(cents[i][1]), "area": a, "hue": hue, **shape}
        if best is None or shape["corner_fill"] > best["corner_fill"]:
            best = found
    return best


def t_pixels_near(
    frame: np.ndarray,
    x: float,
    y: float,
    radius: int,
    t_hue: float = SPECTATOR_T_HUE,
    hue_tol: float = 4.0,
    *,
    context,
) -> int:
    """Saturated T-coloured pixels within radius of (x, y). The tight tolerance keeps the
    orange badge itself out; a carrier half-hidden under the badge still contributes."""
    cfg = context.config
    h, w = frame.shape[:2]
    x0, x1 = max(int(x) - radius, 0), min(int(x) + radius + 1, w)
    y0, y1 = max(int(y) - radius, 0), min(int(y) + radius + 1, h)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    d = np.minimum(np.abs(hue - t_hue), 180 - np.abs(hue - t_hue))
    ok = (d <= hue_tol) & (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ok &= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return int(ok.sum())


def scan_video(
    path: Path,
    near_px: int = 40,
    t_hue: float = SPECTATOR_T_HUE,
    hue_band: Optional[Tuple[float, float]] = None,
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
            b = find_bomb(rgb, hue_band, context=context)
            row = {"frame": i, "seconds": round(i / fps, 3), "x": None, "y": None, "t_pixels": 0}
            if b is not None:
                row.update(x=round(b["x"], 1), y=round(b["y"], 1))
                row["t_pixels"] = t_pixels_near(
                    rgb, b["x"], b["y"], near_px, t_hue, context=context
                )
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
    max_t_pixels: int = 15,
    max_gap_frames: int = 2,
) -> List[Dict]:
    """Runs where the badge stays within still_px of where the run began and almost no T
    colour is nearby, lasting at least min_seconds. Short detection gaps are bridged.

    These are review candidates, not labels: a planted bomb, a carrier hidden under the
    badge or a frozen spectator view can match the same rule."""
    runs: List[Dict] = []
    cur = None
    for r in rows:
        lone = r["x"] is not None and r["t_pixels"] <= max_t_pixels
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


def _csv(rows: List[Dict], fields: List[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_bomb_scan(
    video: Path,
    output: Path,
    near_px: int = 40,
    min_seconds: float = 2.0,
    t_hue: float = SPECTATOR_T_HUE,
    hue_band: Optional[Tuple[float, float]] = None,
    *,
    context,
) -> Dict:
    """Scan one prepared video and write track.csv, candidates.csv, summary.json and an
    enlarged crop of each candidate into a new output directory."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new folder: {output}")
    rows, fps = scan_video(video, near_px, t_hue, hue_band, context=context)
    runs = dropped_candidates(rows, fps, min_seconds)
    write_new_text(output / "track.csv", _csv(rows, ["frame", "seconds", "x", "y", "t_pixels"]))
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
            "near_px": near_px,
            "min_seconds": min_seconds,
            "t_hue": t_hue,
            "hue_band": hue_band,
            "bomb_area": list(context.config.bomb_area),
            "bomb_min_aspect": context.config.bomb_min_aspect,
            "bomb_min_extent": context.config.bomb_min_extent,
            "bomb_min_corner_fill": context.config.bomb_min_corner_fill,
        },
        "claims": "Unreviewed video-only candidates. Not labels and not detector accuracy.",
    }
    write_new_text(output / "summary.json", json.dumps(summary, indent=2))
    return summary
