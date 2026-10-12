"""Synthetic: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import CT_CLASSES, T_CLASSES
from .contracts import Round

MAP_POINTS_FILE = "map_points.json"

T_ROUTES = {
    "A_RUSH": [["t_spawn", "a_ramp", "a_approach", "a_site"]] * 5,
    "B_RUSH": [["t_spawn", "b_apps", "b_site"]] * 5,
    "A_SPLIT": [["t_spawn", "a_ramp", "a_approach", "a_site"]] * 3
    + [["t_spawn", "mid", "connector", "a_site"]] * 2,
    "B_SPLIT": [["t_spawn", "b_apps", "b_site"]] * 3
    + [["t_spawn", "mid", "b_short", "b_site"]] * 2,
    "DEFAULT": [
        ["t_spawn", "mid"],
        ["t_spawn", "b_apps"],
        ["t_spawn", "a_ramp"],
        ["t_spawn", "mid", "connector"],
        ["t_spawn", "a_ramp", "a_approach"],
    ],
}

RUSH_SPEED = {
    "A_RUSH": (0.3, 0.45),
    "B_RUSH": (0.3, 0.45),
    "A_SPLIT": (0.6, 0.9),
    "B_SPLIT": (0.6, 0.9),
    "DEFAULT": (0.8, 1.0),
}

CT_ANCHORS = ["a_site", "a_approach", "mid", "b_short", "b_site"]


def _grid(size: int, *, context):
    if size not in context.state["_GRID_CACHE"]:
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        context.state["_GRID_CACHE"][size] = (
            xx.astype(np.float32),
            yy.astype(np.float32),
        )
    return context.state["_GRID_CACHE"][size]


def _polyline(wp: np.ndarray, s: np.ndarray) -> np.ndarray:
    wp = np.asarray(wp, np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.clip(s, 0, 1) * max(cum[-1], 1e-09)
    i = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    frac = (d - cum[i]) / np.maximum(seg[i], 1e-09)
    return wp[i] + (wp[i + 1] - wp[i]) * frac[:, None]


def _jitter_on_map(
    p, sd: float, rng: np.random.Generator, tries: int = 40, *, context
) -> np.ndarray:
    """Random offset around p, re-drawn until it lands on walkable map (never on black)."""
    p = np.asarray(p, np.float64)
    size = context.state["MAP_MASK"].shape[0]
    for _ in range(tries):
        q = np.clip(p + rng.normal(0, sd, 2), 0.0, 0.999)
        if context.state["MAP_MASK"][int(q[1] * size), int(q[0] * size)]:
            return q
    return snap_to_map(p[None].astype(np.float32), context=context)[0].astype(np.float64)


def _walk(names, n, rng, delay=0.0, duration=1.0, jitter=0.008, *, context):
    wp = np.array([context.state["MAP_POINTS"][x] for x in names], np.float64)
    for k in range(1, len(wp) - 1):
        wp[k] = _jitter_on_map(wp[k], 0.02, rng, context=context)
    wp[0] = _jitter_on_map(wp[0], 0.06, rng, context=context)
    wp[-1] = _jitter_on_map(wp[-1], 0.06, rng, context=context)
    tt = np.linspace(0, 1, n)
    s = np.clip((tt - delay) / max(duration, 1e-06), 0, 1)
    s = s * s * (3 - 2 * s)
    return np.clip(_polyline(wp, s) + rng.normal(0, jitter, (n, 2)), 0.03, 0.97)


def _hold(anchor, n, rng, amp=0.012):
    drift = np.cumsum(rng.normal(0, amp, (n, 2)), axis=0) * 0.5
    return np.clip(
        np.array(anchor, np.float64)[None] + drift + rng.normal(0, 0.006, (n, 2)),
        0.03,
        0.97,
    )


def make_round(label: str, rng: np.random.Generator, n: int, *, context) -> np.ndarray:
    """Before the plant: the attackers run their tactic while the defenders hold their spots."""
    pos = np.zeros((n, context.state["N_PLAYERS"], 2), np.float32)
    lo, hi = RUSH_SPEED[label]
    late = 0.12 if "RUSH" in label else 0.3 if label == "DEFAULT" else 0.25
    for p, route in enumerate(T_ROUTES[label]):
        pos[:, p] = _walk(
            route, n, rng, rng.uniform(0.0, late), rng.uniform(lo, hi), context=context
        )
    for p, a in enumerate(CT_ANCHORS):
        pos[:, 5 + p] = _hold(context.state["MAP_POINTS"][a], n, rng)
    return snap_to_map(pos, context=context)


SAVE_POINTS = {
    "A": ["b_site", "b_short", "mid"],
    "B": ["a_approach", "connector", "a_site"],
}


def _walk_xy(wp, n, rng, delay=0.0, duration=1.0, jitter=0.008) -> np.ndarray:
    """Like _walk, but through explicit (x, y) waypoints instead of callout names."""
    wp = np.asarray(wp, np.float64)
    tt = np.linspace(0, 1, n)
    s = np.clip((tt - delay) / max(duration, 1e-06), 0, 1)
    s = s * s * (3 - 2 * s)
    return np.clip(_polyline(wp, s) + rng.normal(0, jitter, (n, 2)), 0.03, 0.97)


def make_post_plant(
    response: str,
    site: str,
    start: np.ndarray,
    rng: np.random.Generator,
    n: int,
    *,
    context,
) -> np.ndarray:
    """After the plant, (n, 10, 2). `start` is the last frame before the plant, (10, 2).
    RETAKE  - the defenders regroup and arrive on the site together.
    TRICKLE - they arrive one at a time, so each fights alone.
    SAVE    - they give the site up and move away to keep their weapons."""
    pos = np.zeros((n, context.state["N_PLAYERS"], 2), np.float32)
    site_xy = np.array(context.state["MAP_POINTS"]["a_site" if site == "A" else "b_site"])
    for p in range(context.config.n_per_team):
        pos[:, p] = _hold(_jitter_on_map(site_xy, 0.035, rng, context=context), n, rng, 0.008)
    stagger = np.sort(rng.uniform(0.0, 0.55, context.config.n_per_team))
    for k in range(context.config.n_per_team):
        s0 = start[context.config.n_per_team + k].astype(np.float64)
        goal = _jitter_on_map(site_xy, 0.03, rng, context=context)
        via = (
            [np.array(context.state["MAP_POINTS"]["ct_spawn"])]
            if np.linalg.norm(s0 - site_xy) > 0.25
            else []
        )
        if response == "RETAKE":
            path, delay, dur = (
                [s0, *via, goal],
                rng.uniform(0.25, 0.35),
                rng.uniform(0.4, 0.5),
            )
        elif response == "TRICKLE":
            path, delay, dur = ([s0, *via, goal], stagger[k], rng.uniform(0.25, 0.4))
        else:
            away = _jitter_on_map(
                context.state["MAP_POINTS"][SAVE_POINTS[site][k % 3]],
                0.03,
                rng,
                context=context,
            )
            path, delay, dur = (
                [s0, away],
                rng.uniform(0.0, 0.25),
                rng.uniform(0.5, 0.8),
            )
        pos[:, context.config.n_per_team + k] = _walk_xy(path, n, rng, delay, dur)
    return snap_to_map(pos, context=context)


def _drawn_background(size: int, rng: np.random.Generator, *, context) -> np.ndarray:
    """Last-resort abstraction: corridors drawn between callouts. Used only when neither real
    footage nor a map image is available."""
    bg = np.full((size, size, 3), (14, 18, 26), np.float32)
    corridors = [
        ["t_spawn", "a_ramp", "a_approach", "a_site"],
        ["t_spawn", "b_apps", "b_site"],
        ["t_spawn", "mid", "ct_spawn"],
        ["mid", "connector", "a_site"],
        ["mid", "b_short", "b_site"],
        ["a_site", "a_approach", "mid"],
    ]
    s_ = np.linspace(0, 1, 400)
    for names in corridors:
        pts = _polyline(np.array([context.state["MAP_POINTS"][x] for x in names]), s_) * size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xs = np.clip(pts[:, 0].astype(int) + dx, 0, size - 1)
                ys = np.clip(pts[:, 1].astype(int) + dy, 0, size - 1)
                bg[ys, xs] = (46, 52, 64)
    for name in ("a_site", "b_site"):
        cx, cy = (np.array(context.state["MAP_POINTS"][name]) * size).astype(int)
        r = size // 12
        bg[max(cy - r, 0) : cy + r, max(cx - r, 0) : cx + r] = (58, 64, 78)
    return np.clip(bg + rng.normal(0, 3.0, bg.shape), 0, 255)


def map_image_background(path: str, size: int, dark=(12, 16, 24)) -> np.ndarray:
    """Load a Mirage radar PNG and composite it over a dark panel, so it looks like the
    in-game radar rather than a white-background diagram."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.shape[2] == 4:
        rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32)
        a = (rgb.min(-1, keepdims=True) < 235).astype(np.float32)
    comp = rgb * a + np.array(dark, np.float32) * (1 - a)
    return cv2.resize(comp, (size, size), interpolation=cv2.INTER_AREA)


def build_background(size: int, rng: np.random.Generator, source=None, *, context) -> np.ndarray:
    """The radar the synthetic rounds are drawn on.

    Preference order, best first:
      1. `source` as an array  -> the median frame of YOUR footage: the real radar, exactly as
         the detector will see it, with players washed out. Nothing to download, nothing to
         align.
      2. `source` as a path    -> a Mirage radar image, composited over a dark panel.
      3. neither               -> the drawn abstraction.
    """
    if isinstance(source, np.ndarray):
        bg = cv2.resize(source.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        return np.clip(bg + rng.normal(0, 2.0, bg.shape), 0, 255)
    if isinstance(source, str) and Path(source).exists():
        bg = map_image_background(source, size)
        return np.clip(bg + rng.normal(0, 2.0, bg.shape), 0, 255)
    return _drawn_background(size, rng, context=context)


def plot_background(*, context) -> np.ndarray:
    """Background for every PLOT (never for synthetic training frames). Best first:
      1. CFG.map_image at full resolution — the clean radar screenshot. It frames the same area
         as the crop (checked against the footage median: correlation 0.986 on a plain resize).
      2. the median of your footage, 3. the drawn abstraction.
    Synthetic frames keep using build_background: the screenshot has the orange bomb icon baked
    in, and a fixed T-coloured blob in training frames would teach the detector the wrong thing."""
    p = Path(context.config.map_image)
    if p.exists():
        n = max(cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[:2])
        return map_image_background(str(p), n).astype(np.uint8)
    return build_background(
        context.config.minimap_px,
        np.random.default_rng(1234),
        context.state["MAP_BG_SOURCE"],
        context=context,
    ).astype(np.uint8)


def pick_map_points(
    background: np.ndarray,
    out_json: str = "map_points.json",
    names: Sequence[str] = None,
    *,
    context,
):
    """Click each callout once on the real radar, in the order printed. Saves map_points.json,
    which is picked up automatically next run."""
    names = list(names or context.state["MAP_POINTS"].keys())
    pts: Dict[str, Tuple[float, float]] = {}
    h, w = background.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(background.astype(np.uint8))

    def title():
        ax.set_title(f"click: {names[len(pts)]}" if len(pts) < len(names) else "done - close me")
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None or len(pts) >= len(names):
            return
        pts[names[len(pts)]] = (
            round(float(ev.xdata) / w, 3),
            round(float(ev.ydata) / h, 3),
        )
        ax.plot(ev.xdata, ev.ydata, "o", mfc="none", mec="#00ff88", ms=12, mew=2)
        if len(pts) == len(names):
            Path(out_json).write_text(json.dumps(pts, indent=1))
            print(f"saved {len(pts)} callouts -> {out_json}")
        title()

    fig.canvas.mpl_connect("button_press_event", on_click)
    title()
    plt.show()
    return pts


def walkable_mask(bg: np.ndarray, dark: float = 25.0) -> np.ndarray:
    """bool (H, W): True on the map's floor. The radar is near-black off the map and inside
    buildings, and no player can ever stand there."""
    m = (bg.astype(np.float32).mean(-1) > dark).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


def _nearest_walkable(mask: np.ndarray) -> np.ndarray:
    """(H, W, 2) pixel x, y of the nearest walkable pixel, for every pixel."""
    _, lab = cv2.distanceTransformWithLabels(
        (~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.where(mask)
    lut = np.zeros((lab.max() + 1, 2), np.float32)
    lut[lab[ys, xs]] = np.stack([xs, ys], 1)
    return lut[lab]


def snap_to_map(pos: np.ndarray, *, context) -> np.ndarray:
    """Move any synthetic player that landed off the map onto the nearest walkable pixel.
    Spawn and arrival jitter used to push attackers onto the black area beside T spawn."""
    size = context.state["MAP_MASK"].shape[0]
    ix = np.clip((pos[..., 0] * size).astype(int), 0, size - 1)
    iy = np.clip((pos[..., 1] * size).astype(int), 0, size - 1)
    off = ~context.state["MAP_MASK"][iy, ix]
    out = pos.copy()
    out[off] = (context.state["_NEAREST"][iy[off], ix[off]] + 0.5) / size
    return out


T_COLOUR = np.array([255, 150, 40], np.float32)

CT_COLOUR = np.array([70, 150, 255], np.float32)


def render_frames(pos, bg, size, rng, radius=1.7, *, context) -> np.ndarray:
    xx, yy = _grid(size, context=context)
    out = np.empty((pos.shape[0], size, size, 3), np.uint8)
    colours = np.stack(
        [T_COLOUR] * context.config.n_per_team + [CT_COLOUR] * context.config.n_per_team
    )
    for f in range(pos.shape[0]):
        px, py = (pos[f, :, 0] * size, pos[f, :, 1] * size)
        d2 = (xx[None] - px[:, None, None]) ** 2 + (yy[None] - py[:, None, None]) ** 2
        blob = np.exp(-d2 / (2 * radius**2))
        img = bg + np.einsum("psq,pc->sqc", blob, colours) + rng.normal(0, 3.0, bg.shape)
        out[f] = np.clip(img, 0, 255).astype(np.uint8)
    return out


P_PLANT = {
    "A_RUSH": 0.8,
    "B_RUSH": 0.8,
    "A_SPLIT": 0.75,
    "B_SPLIT": 0.75,
    "DEFAULT": 0.55,
}

P_CT_WIN_AFTER_PLANT = {"RETAKE": 0.45, "TRICKLE": 0.25, "SAVE": 0.03}

P_T_WIN_NO_PLANT = 0.3

SITE_OF = {"A_RUSH": "A", "A_SPLIT": "A", "B_RUSH": "B", "B_SPLIT": "B"}


def generate_split(
    n_matches: int,
    prefix: str,
    seed: int,
    render: bool,
    rounds_per_match: int = 20,
    *,
    context,
) -> List[Round]:
    rng = np.random.default_rng(seed)
    bg = build_background(
        context.config.minimap_px,
        np.random.default_rng(1234),
        context.state["MAP_BG_SOURCE"],
        context=context,
    )
    out = []
    for m in range(n_matches):
        t_w = rng.dirichlet(np.ones(len(T_CLASSES)) * 2.5)
        ct_w = rng.dirichlet(np.ones(len(CT_CLASSES)) * 2.5)
        for r in range(rounds_per_match):
            label = T_CLASSES[int(rng.choice(len(T_CLASSES), p=t_w))]
            pos = make_round(label, rng, context.config.seq_len, context=context)
            site = ct = post = None
            if rng.random() < P_PLANT[label]:
                site = SITE_OF.get(label) or ("A" if rng.random() < 0.5 else "B")
                ct = CT_CLASSES[int(rng.choice(len(CT_CLASSES), p=ct_w))]
                post = make_post_plant(
                    ct, site, pos[-1], rng, context.config.seq_len, context=context
                )
                winner = "CT" if rng.random() < P_CT_WIN_AFTER_PLANT[ct] else "T"
            else:
                winner = "T" if rng.random() < P_T_WIN_NO_PLANT else "CT"
            rd = Round(f"{prefix}_m{m:03d}", r, label, pos, None, winner, site, ct, post)
            if render:
                rd.frames = render_frames(pos, bg, context.config.minimap_px, rng, context=context)
                if post is not None:
                    rd.post_frames = render_frames(
                        post, bg, context.config.minimap_px, rng, context=context
                    )
            out.append(rd)
    return out
