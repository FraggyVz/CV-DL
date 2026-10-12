"""Coordinates: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import SITE_ANCHORS


def world_to_radar(world: np.ndarray, fit: Dict[str, float], *, context) -> np.ndarray:
    """(..., 2) world X,Y -> (..., 2) crop-normalised x right, y down. NaN stays NaN."""
    u = (
        (world[..., 0] - context.config.overview_pos_x)
        / context.config.overview_scale
        / context.config.overview_px
    )
    v = (
        (context.config.overview_pos_y - world[..., 1])
        / context.config.overview_scale
        / context.config.overview_px
    )
    return np.stack([u * fit["s"] + fit["ox"], v * fit["s"] + fit["oy"]], -1).astype(np.float32)


def _fit_score(uv: np.ndarray, walk: np.ndarray) -> Tuple[float, float]:
    H, W = walk.shape
    x, y = ((uv[:, 0] * W).astype(int), (uv[:, 1] * H).astype(int))
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    on_map = walk[y[inside], x[inside]].mean() * inside.mean() if inside.any() else 0.0
    occ = np.zeros_like(walk, np.uint8)
    occ[y[inside], x[inside]] = 1
    occ = cv2.dilate(occ, np.ones((5, 5), np.uint8)).astype(bool)
    return (float(on_map), float((occ & walk).sum() / max(walk.sum(), 1)))


def fit_radar(rounds: List[Dict], bg: np.ndarray, dark: int = 25, *, context) -> Dict[str, float]:
    pts = np.concatenate([r["world"].reshape(-1, 2) for r in rounds])
    pts = pts[np.isfinite(pts).all(1)]
    walk = bg.mean(-1) > dark
    ident = {"s": 1.0, "ox": 0.0, "oy": 0.0}
    best, best_val = (ident, -1.0)
    for s in np.arange(0.9, 1.101, 0.01):
        for ox in np.arange(-0.06, 0.0601, 0.005):
            for oy in np.arange(-0.06, 0.0601, 0.005):
                f = {"s": float(s), "ox": float(ox), "oy": float(oy)}
                on, cov = _fit_score(world_to_radar(pts, f, context=context), walk)
                if on * cov > best_val:
                    best, best_val = (f, on * cov)
    on0, cov0 = _fit_score(world_to_radar(pts, ident, context=context), walk)
    on1, cov1 = _fit_score(world_to_radar(pts, best, context=context), walk)
    print(f"overview only : {on0 * 100:5.1f}% on map, {cov0 * 100:5.1f}% of map covered")
    print(f"fitted        : {on1 * 100:5.1f}% on map, {cov1 * 100:5.1f}% of map covered   {best}")
    return best


def show_fit(rounds: List[Dict], bg: Optional[np.ndarray], fit: Dict[str, float], *, context):
    bg = context.state["PLOT_BG"] if bg is None else bg
    pts = np.concatenate(
        [world_to_radar(r["world"], fit, context=context).reshape(-1, 2) for r in rounds]
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if bg is not None:
        ax.imshow(bg, extent=[0, 1, 1, 0])
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="lime", alpha=0.35)
    for name, (x, y) in SITE_ANCHORS.items():
        ax.text(
            x,
            y,
            name,
            color="white",
            fontsize=14,
            ha="center",
            va="center",
            weight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"demo positions on your radar (n={len(pts)})")
    plt.show()
