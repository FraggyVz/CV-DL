"""
StratScout — auto-labeller
==========================

Your radar icons are a handful of *saturated* colours sitting on a *desaturated* map.
That is almost the easiest possible segmentation problem, so a human should not be clicking
two thousand of them by hand.

Workflow:

    from stratscout_autolabel import auto_label, ReviewTool

    frames = np.load("pilot_frames.npy")
    auto_label(frames, "pilot_labels.json")        # machine proposes
    ReviewTool(frames, "pilot_labels.json").run()  # you correct

Reviewing is roughly five times faster than clicking from scratch, because most frames need
no change at all — you glance, press n, move on.

Keep this honest in your report: these are *machine-proposed* labels that a human verified,
not hand-labelled ground truth. That is a normal and defensible annotation method, but it is
not the same claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from setup_capture_tools import LabelTool


# --------------------------------------------------------------------------- detection core

def _hue_dist(a, b):
    d = abs(a - b)
    return min(d, 180 - d)

def _candidates(frame, sat_min=90, val_min=90, area=(3, 200), exclude=None):
    """Every dot-shaped saturated blob, with its hue and a sharpened centre."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] >= sat_min) & (hsv[:, :, 2] >= val_min)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (area[0] <= a <= area[1]):
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) > 3 * max(min(bw, bh), 1):
            continue
        ys, xs = np.nonzero(lab == i)
        sat = hsv[ys, xs, 1].astype(np.float32)
        wt = sat ** 2                      # vivid core outweighs the pale facing-cone
        x = float((xs * wt).sum() / wt.sum()) / w
        y = float((ys * wt).sum() / wt.sum()) / h
        if exclude and any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in exclude):
            continue
        core = sat >= np.percentile(sat, 60)
        out.append({"hue": float(np.median(hsv[ys, xs, 0][core])), "x": x, "y": y, "area": a})
    return out


def find_icons(frame, sat_min=90, val_min=90, area=(3, 200), max_icons=5,
               exclude=None, palette_hues=None, hue_tol=12.0):
    c = _candidates(frame, sat_min, val_min, area, exclude)
    if palette_hues is not None:
        c = [d for d in c if min(_hue_dist(d["hue"], p) for p in palette_hues) <= hue_tol]
    c.sort(key=lambda d: -d["area"])
    return [[d["x"], d["y"]] for d in c[:max_icons]]

def learn_palette(frames, n_colors=5, sample=120, min_sep=5, exclude=None,
                  ignore_hues=((0, 6), (174, 180)), **kw):
    """Measure your five teammate hues from your own footage.

    `ignore_hues` drops red during learning only. In CS2 red is reserved for enemies and
    death markers, so it can never be a teammate colour — a safe prior, unlike guessing
    where orange ends.
    """
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    hues = [d["hue"] for i in idx for d in _candidates(frames[int(i)], exclude=exclude, **kw)]
    hues = [h for h in hues if not any(lo <= h <= hi for lo, hi in ignore_hues)]
    if not hues:
        raise RuntimeError("no blobs found — loosen sat_min/val_min or area")

    hist = np.bincount(np.round(hues).astype(int) % 180, minlength=180).astype(float)
    k = np.array([1., 2., 3., 2., 1.]); k /= k.sum()
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="same")[2:-2]

    peaks, taken = [], np.zeros(180, bool)
    for _ in range(n_colors):
        pick = next((int(h) for h in np.argsort(-sm) if not taken[h] and sm[h] > 0), None)
        if pick is None:
            break
        peaks.append(pick)
        for o in range(-min_sep, min_sep + 1):
            taken[(pick + o) % 180] = True
    peaks.sort()
    print(f"learned {len(peaks)} hues from {len(hues)} blobs:", peaks)
    return peaks

def auto_label(frames: np.ndarray, out_json: str, preview: int = 6, **kw) -> Dict[int, List]:
    """Propose icon centres for every frame and write them in LabelTool's format."""
    points = {i: find_icons(f, **kw) for i, f in enumerate(frames)}
    counts = np.array([len(v) for v in points.values()])
    Path(out_json).write_text(json.dumps(
        {"team_channel": "teammate", "n_frames": int(len(frames)), "auto": True,
         "points": {str(k): v for k, v in points.items() if v}}, indent=1))

    print(f"proposed {counts.sum()} icons across {len(frames)} frames")
    for k in range(0, 7):
        n = int((counts == k).sum())
        if n:
            print(f"  {n:4d} frames with {k} icons {'  <-- expected' if k == 5 else ''}")
    print(f"\nwrote {out_json} — now run ReviewTool to correct it")

    if preview:
        import matplotlib.pyplot as plt
        idx = np.linspace(0, len(frames) - 1, preview).astype(int)
        cols = min(preview, 6)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.4))
        for ax, i in zip(np.atleast_1d(axes), idx):
            ax.imshow(frames[i])
            h, w = frames[i].shape[:2]
            for x, y in points[int(i)]:
                ax.plot(x * w, y * h, "o", mfc="none", mec="#00ff88", ms=13, mew=2)
            ax.set_title(f"f{i} — {len(points[int(i)])}"); ax.axis("off")
        plt.tight_layout(); plt.show()
    return points


def tune(frames: np.ndarray, i: int = 0, **kw) -> None:
    """Show the saturation mask for one frame so you can pick thresholds by eye."""
    import matplotlib.pyplot as plt
    assert cv2 is not None, "pip install opencv-python"
    f = frames[i]
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    sat_min, val_min = kw.get("sat_min", 90), kw.get("val_min", 90)
    mask = (hsv[:, :, 1] >= sat_min) & (hsv[:, :, 2] >= val_min)
    pts = find_icons(f, **kw)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(f); axes[0].set_title("frame")
    axes[1].imshow(hsv[:, :, 1], cmap="viridis"); axes[1].set_title("saturation")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title(f"mask (S>={sat_min}, V>={val_min})")
    axes[3].imshow(f); axes[3].set_title(f"found {len(pts)}")
    h, w = f.shape[:2]
    for x, y in pts:
        axes[3].plot(x * w, y * h, "o", mfc="none", mec="#00ff88", ms=13, mew=2)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(); plt.show()
    print("too many blobs -> raise sat_min, or narrow `area`")
    print("missing dots    -> lower sat_min / val_min, or widen `area`")


# ------------------------------------------------------------------------------ review tool

class ReviewTool(LabelTool):
    """LabelTool, but right-click deletes the *nearest* point instead of the last one.

    That one change is what makes correcting proposals practical: you are fixing scattered
    mistakes, not undoing a stack in order.
    """

    def _click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        h, w = self.frames[self.i].shape[:2]
        pts = self.labels.setdefault(self.i, [])
        cx, cy = float(ev.xdata) / w, float(ev.ydata) / h
        if ev.button == 1:
            pts.append([cx, cy])
        elif ev.button == 3 and pts:
            d = [(p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts]
            pts.pop(int(np.argmin(d)))
        self._draw()


if __name__ == "__main__":
    print(__doc__)
