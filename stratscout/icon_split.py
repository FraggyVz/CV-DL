"""Icon split: count and place overlapping player icons of the same team.

Every player icon is a disc of the same size, so a same-team blob much larger than one
icon holds several. The number of icons is estimated from blob area, and their centres
from k-means on the blob's pixel coordinates. Each team gets its own colour mask, so a T
touching a CT stays two blobs. Video-only; the original colour baseline in segmentation.py
is unchanged and remains the comparison point.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .segmentation import blob_shape, looks_like_bomb

TEAMS = ("T", "CT")


@dataclass
class SplitParams:
    """single_area: one icon's blob area (px). A blob of at least split_ratio x that holds
    several icons; each icon after the first adds about extra_area x single_area, because
    overlapping discs share pixels."""

    # Fitted by icon-count-eval on Furia-Falcons and Spirit-Falcons (494x500 spectator radar);
    # held-out NaVi-Falcons all-alive frames showing 5 per team rose from 9% to 63%.
    single_area: float = 246.0
    split_ratio: float = 1.15
    extra_area: float = 0.6
    hue_tol: float = 6.0
    min_area: int = 80


def team_blobs(
    frame: np.ndarray,
    team_hues: Dict[str, float],
    background: Optional[np.ndarray] = None,
    params: SplitParams = SplitParams(),
    *,
    context,
) -> List[Dict]:
    """RGB frame -> same-team blobs {team, area, xs, ys}. Static map colour (with a
    background) and the bomb badge are excluded."""
    cfg = context.config
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    vivid = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    if background is not None:
        moving = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(-1)
        vivid &= moving > cfg.bg_thresh
    kernel = np.ones((3, 3), np.uint8)
    blobs = []
    for ti, team in enumerate(TEAMS):
        d = np.abs(hue - team_hues[team])
        mask = (vivid & (np.minimum(d, 180 - d) <= params.hue_tol)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x0, y0, bw, bh, a = (int(v) for v in stats[i])
            if a < params.min_area:
                continue
            if looks_like_bomb(
                a, blob_shape(lab[y0 : y0 + bh, x0 : x0 + bw] == i), context=context
            ):
                continue
            ys, xs = np.nonzero(lab[y0 : y0 + bh, x0 : x0 + bw] == i)
            blobs.append({"team": ti, "area": a, "xs": xs + x0, "ys": ys + y0})
    return blobs


def icons_in_blob(area: float, params: SplitParams) -> int:
    """Estimated number of icons in a same-team blob of this area."""
    if area < params.split_ratio * params.single_area:
        return 1
    extra = (area - params.single_area) / (params.extra_area * params.single_area)
    return max(2, int(round(extra)) + 1)


def split_centres(xs: np.ndarray, ys: np.ndarray, k: int) -> List[tuple]:
    """k icon centres for one blob: k-means on its pixel coordinates."""
    if k <= 1 or len(xs) < k:
        return [(float(xs.mean()), float(ys.mean()))]
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    cv2.setRNGSeed(0)
    _, _, centres = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return [(float(cx), float(cy)) for cx, cy in centres]


def find_icons_split(
    frame: np.ndarray,
    team_hues: Dict[str, float],
    background: Optional[np.ndarray] = None,
    params: SplitParams = SplitParams(),
    *,
    context,
) -> List[List[float]]:
    """-> [[x, y, team], ...] like find_icons_teams (x, y normalised; team 0 = T, 1 = CT),
    with overlapping same-team icons split into separate detections."""
    h, w = frame.shape[:2]
    out = []
    for b in team_blobs(frame, team_hues, background, params, context=context):
        k = icons_in_blob(b["area"], params)
        for cx, cy in split_centres(b["xs"], b["ys"], k):
            out.append([cx / w, cy / h, float(b["team"])])
    return out
