"""Icon split: count and place overlapping player icons of the same team.

Every player icon is a disc of the same size with a white direction arrow attached. Each
team gets its own colour mask, so a T touching a CT stays two blobs. A blob with no white
arrow attached is a death marker (✕), not a player. The number of icons in a blob is chosen
by fitting k equal discs (k-means centres) and keeping the k whose union best matches the
blob (IoU); the older area rule is kept for comparison. Each team is capped at five.
Video-only; the original colour baseline in segmentation.py is unchanged.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .segmentation import blob_shape, looks_like_bomb

TEAMS = ("T", "CT")


@dataclass
class SplitParams:
    """method "discs" counts icons by disc fitting; "area" uses blob area: a blob of at
    least split_ratio x single_area holds several icons, each extra one adding about
    extra_area x single_area. single_area also sets the disc radius. Blobs touching fewer
    than min_arrow_px white pixels are death markers (0 disables). max_per_team caps a
    team's icons (5v5); None disables."""

    # single_area is the median early-play blob area on Furia-Falcons and Spirit-Falcons.
    single_area: float = 246.0
    split_ratio: float = 1.15
    extra_area: float = 0.6
    hue_tol: float = 6.0
    min_area: int = 80
    max_per_team: Optional[int] = 5
    method: str = "discs"
    min_arrow_px: int = 8
    max_icons_per_blob: int = 5


def attached_white(frame: np.ndarray, xs: np.ndarray, ys: np.ndarray, band: int = 4) -> int:
    """White pixels within band px outside a blob: a player's direction arrow."""
    h, w = frame.shape[:2]
    x0, y0 = max(int(xs.min()) - band, 0), max(int(ys.min()) - band, 0)
    x1, y1 = min(int(xs.max()) + band + 1, w), min(int(ys.max()) + band + 1, h)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    m[ys - y0, xs - x0] = 1
    ring = cv2.dilate(m, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) & (1 - m)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    return int(((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 190) & (ring > 0)).sum())


def disc_fits(xs: np.ndarray, ys: np.ndarray, radius: float, k_max: int) -> List[float]:
    """IoU between the hole-filled blob and the union of k discs at k-means centres, for
    k = 1..k_max."""
    x0, y0 = int(xs.min()) - int(radius) - 2, int(ys.min()) - int(radius) - 2
    w = int(xs.max()) - x0 + int(radius) + 3
    h = int(ys.max()) - y0 + int(radius) + 3
    m = np.zeros((h, w), np.uint8)
    m[ys - y0, xs - x0] = 1
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, -1)
    fits = []
    for k in range(1, k_max + 1):
        if len(xs) < k:
            break
        discs = np.zeros_like(m)
        for cx, cy in split_centres(xs, ys, k):
            centre = (int(round(cx - x0)), int(round(cy - y0)))
            cv2.circle(discs, centre, int(round(radius)), 1, -1)
        fits.append(float((discs & filled).sum() / max((discs | filled).sum(), 1)))
    return fits


def team_blobs(
    frame: np.ndarray,
    team_hues: Dict[str, float],
    background: Optional[np.ndarray] = None,
    params: SplitParams = SplitParams(),
    *,
    context,
) -> List[Dict]:
    """RGB frame -> same-team blobs {team, area, xs, ys, white, fits}. Static map colour
    (with a background) and the bomb badge are excluded; death markers are kept here (see
    is_marker) so evaluations can count them."""
    cfg = context.config
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    vivid = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    if background is not None:
        moving = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(-1)
        vivid &= moving > cfg.bg_thresh
    kernel = np.ones((3, 3), np.uint8)
    radius = float(np.sqrt(params.single_area / np.pi))
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
            xs, ys = xs + x0, ys + y0
            k_max = min(
                params.max_icons_per_blob, max(2, int(np.ceil(a / (0.4 * params.single_area))))
            )
            blobs.append(
                {
                    "team": ti,
                    "area": a,
                    "xs": xs,
                    "ys": ys,
                    "white": attached_white(frame, xs, ys),
                    "fits": disc_fits(xs, ys, radius, k_max) if params.method == "discs" else [],
                }
            )
    return blobs


def is_marker(blob: Dict, params: SplitParams) -> bool:
    """A team-coloured blob with no attached arrow is a death marker (✕)."""
    return blob["white"] < params.min_arrow_px


def icons_in_blob(area: float, params: SplitParams) -> int:
    """Area rule: estimated number of icons in a same-team blob of this area."""
    if area < params.split_ratio * params.single_area:
        return 1
    extra = (area - params.single_area) / (params.extra_area * params.single_area)
    return max(2, int(round(extra)) + 1)


def blob_icons(blob: Dict, params: SplitParams) -> int:
    """Icons in one blob before the team cap: 0 for a death marker, else by method."""
    if is_marker(blob, params):
        return 0
    if params.method == "discs" and blob["fits"]:
        return int(np.argmax(blob["fits"])) + 1
    return icons_in_blob(blob["area"], params)


def allocate(blobs: Sequence[Dict], params: SplitParams, cap: Optional[int] = None) -> List[int]:
    """Icons per blob for one team, at most cap in total (params.max_per_team if None).

    While the team exceeds the cap, the blob losing least by giving up an icon gives one:
    the smallest drop in disc-fit IoU ("discs") or the least spare area ("area"). When every
    blob holds one icon, the worst-fitting (or smallest) blob is dropped."""
    cap = params.max_per_team if cap is None else cap
    ks = [blob_icons(b, params) for b in blobs]
    if cap is None:
        return ks
    discs = params.method == "discs"

    def loss(i: int) -> float:
        b, k = blobs[i], ks[i]
        if discs and len(b["fits"]) >= k:
            return b["fits"][k - 1] - b["fits"][k - 2]
        return b["area"] - params.single_area * (1 + params.extra_area * (k - 1))

    def single_quality(i: int) -> float:
        b = blobs[i]
        return b["fits"][0] if discs and b["fits"] else b["area"]

    while sum(ks) > max(cap, 0):
        multi = [i for i, k in enumerate(ks) if k > 1]
        if multi:
            ks[min(multi, key=loss)] -= 1
        else:
            ks[min((i for i, k in enumerate(ks) if k), key=single_quality)] = 0
    return ks


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
    t_reserved: int = 0,
    *,
    context,
) -> List[List[float]]:
    """-> [[x, y, team], ...] like find_icons_teams (x, y normalised; team 0 = T, 1 = CT),
    with death markers removed, overlapping same-team icons split and each team capped.
    t_reserved lowers the T cap, e.g. by one for a carrier hidden under the bomb badge."""
    h, w = frame.shape[:2]
    blobs = team_blobs(frame, team_hues, background, params, context=context)
    out = []
    for ti in (0, 1):
        team = [b for b in blobs if b["team"] == ti]
        cap = params.max_per_team
        if cap is not None and ti == 0:
            cap -= t_reserved
        for b, k in zip(team, allocate(team, params, cap)):
            if k:
                for cx, cy in split_centres(b["xs"], b["ys"], k):
                    out.append([cx / w, cy / h, float(ti)])
    return out
