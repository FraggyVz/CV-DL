"""Segmentation: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_TEAM_HUES = {"T": 17.0, "CT": 105.0}


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180 - d)


def learn_background(frames: np.ndarray, sample: int = 150) -> np.ndarray:
    """Median frame: the map, with the players removed."""
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    return np.median(frames[idx].astype(np.float32), axis=0).astype(np.uint8)


def learn_team_hues(
    frames: np.ndarray,
    sample: int = 60,
    sat_min: int = None,
    val_min: int = None,
    *,
    context,
) -> Dict[str, float]:
    """Measure the two real hues from the footage instead of trusting the defaults."""
    sat_min = context.config.sat_min if sat_min is None else sat_min
    val_min = context.config.val_min if val_min is None else val_min
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    hues: List[float] = []
    for i in idx:
        hsv = cv2.cvtColor(frames[int(i)], cv2.COLOR_RGB2HSV)
        m = (hsv[:, :, 1] >= sat_min) & (hsv[:, :, 2] >= val_min)
        hues += hsv[:, :, 0][m].astype(float).tolist()
    if not hues:
        raise RuntimeError("no saturated pixels — lower sat_min/val_min")
    hist = np.bincount(np.round(hues).astype(int) % 180, minlength=180).astype(float)
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="same")[2:-2]
    h = np.arange(180)
    warm = int(np.argmax(np.where(h < 45, sm, 0)))
    cool = int(np.argmax(np.where((h > 80) & (h < 135), sm, 0)))
    print(f"learned team hues:  T (orange) ~{warm}   CT (blue) ~{cool}")
    return {"T": float(warm), "CT": float(cool)}


def find_icons_teams(
    frame: np.ndarray,
    team_hues: Optional[Dict[str, float]] = None,
    background: Optional[np.ndarray] = None,
    sat_min: int = None,
    val_min: int = None,
    area: Tuple[int, int] = None,
    hue_tol: float = None,
    bg_thresh: int = None,
    max_per_team: int = None,
    *,
    context,
) -> List[List[float]]:
    """-> [[x, y, team], ...] with x,y normalised and team 0 = T (orange), 1 = CT (blue)."""
    hues = team_hues or DEFAULT_TEAM_HUES
    sat_min = context.config.sat_min if sat_min is None else sat_min
    val_min = context.config.val_min if val_min is None else val_min
    area = context.config.icon_area if area is None else area
    hue_tol = context.config.hue_tol if hue_tol is None else hue_tol
    bg_thresh = context.config.bg_thresh if bg_thresh is None else bg_thresh
    max_per_team = context.config.n_per_team if max_per_team is None else max_per_team
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= sat_min) & (hsv[:, :, 2] >= val_min)
    if background is not None:
        mask &= np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(-1) > bg_thresh
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found: List[Tuple[int, float, List[float]]] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not area[0] <= a <= area[1]:
            continue
        bw, bh = (stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) > 3 * max(min(bw, bh), 1):
            continue
        ys, xs = np.nonzero(lab == i)
        sat = hsv[ys, xs, 1].astype(np.float32)
        wt = sat**2
        x = float((xs * wt).sum() / wt.sum()) / w
        y = float((ys * wt).sum() / wt.sum()) / h
        core = sat >= np.percentile(sat, 60)
        hue = float(np.median(hsv[ys, xs, 0][core]))
        best = min(context.config.teams, key=lambda t: _hue_dist(hue, hues[t]))
        if _hue_dist(hue, hues[best]) > hue_tol:
            continue
        found.append((context.config.teams.index(best), float(a), [x, y]))
    out: List[List[float]] = []
    for ti in (0, 1):
        same = sorted([f for f in found if f[0] == ti], key=lambda f: -f[1])[:max_per_team]
        out += [[p[0], p[1], float(ti)] for _, _, p in same]
    return out
