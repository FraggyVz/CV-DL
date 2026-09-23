"""
StratScout — capture & annotation toolkit
=========================================

Three jobs, in the order you will do them:

  1. pick_crop_box(video)          find where the radar sits, once per capture setup
  2. extract_clip(video, ...)      pull a fixed-length window of cropped radar frames
  3. LabelTool(frames)             click the icon centres to build detector labels

Install:  pip install opencv-python matplotlib numpy

Everything here writes plain .npy and .json so nothing is locked to this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.pyplot import box
import numpy as np

try:
    import cv2
except ImportError:  # keep the module importable so you can read it before installing
    cv2 = None


# ----------------------------------------------------------------------------- 1. crop box

def grab_frame(video: str, t_seconds: float = 10.0) -> np.ndarray:
    """Return one RGB frame from `video` at `t_seconds`, as uint8 (H, W, 3)."""
    assert cv2 is not None, "pip install opencv-python"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_seconds * fps))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame at t={t_seconds}s")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pick_crop_box(video: str, t_seconds: float = 10.0) -> Tuple[int, int, int, int]:
    """Click the radar's top-left then bottom-right corner. Returns (x0, y0, x1, y1).

    Zoom in with the magnifier first — a few pixels of slop here shifts every coordinate
    downstream, so it is worth being fussy. The box should hug the radar's outer edge and
    exclude the round timer and scoreboard.
    """
    import matplotlib.pyplot as plt

    frame = grab_frame(video, t_seconds)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame)
    ax.set_title("Click TOP-LEFT of the radar, then BOTTOM-RIGHT.  (zoom first)")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    (x0, y0), (x1, y1) = pts
    box = (int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)), int(max(y0, y1)))
    print(f"crop_box = {box}   size = {box[2]-box[0]} x {box[3]-box[1]} px")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(frame[box[1]:box[3], box[0]:box[2]])
    ax.set_title("verify: this should be the radar and nothing else")
    ax.axis("off")
    plt.show()
    return box


def check_radar_is_locked(video: str, crop_box: Tuple[int, int, int, int],
                          times: Sequence[float] = (5, 20, 40, 60)) -> None:
    """Show the same crop at several timestamps.

    THE test for `cl_radar_rotate 0` and `cl_radar_always_centered 0`: the map outline must
    be pixel-identical in all four panels while the dots move. If the map itself shifts or
    spins, stop — your capture settings are wrong and no amount of modelling fixes it.
    """
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = crop_box
    crops = [grab_frame(video, t)[y0:y1, x0:x1] for t in times]
    fig, axes = plt.subplots(1, len(crops) + 1, figsize=(3.2 * (len(crops) + 1), 3.4))
    for ax, c, t in zip(axes, crops, times):
        ax.imshow(c); ax.set_title(f"t={t}s"); ax.axis("off")
    diff = np.abs(crops[0].astype(int) - crops[-1].astype(int)).sum(-1)
    axes[-1].imshow(diff, cmap="inferno")
    axes[-1].set_title("difference\n(should be dots only)"); axes[-1].axis("off")
    plt.tight_layout(); plt.show()

    changed = (diff > 30).mean()
    print(f"{changed*100:.1f}% of pixels changed between first and last frame.")
    print("Under ~5% means a locked radar. Much more than that means it is moving.")


# ------------------------------------------------------------------------- 2. frame extract

def extract_clip(video: str, start_s: float, end_s: float,
                 crop_box: Tuple[int, int, int, int],
                 n_frames: int = 32, out_px: Optional[int] = None) -> np.ndarray:
    """Pull `n_frames` evenly spaced cropped radar frames. Returns uint8 (n_frames, H, W, 3).

    Evenly spaced rather than fixed-fps on purpose: rounds differ in length, and the
    classifier needs a fixed sequence length. A 20-second round and a 40-second round both
    become 32 steps, so the model sees the *shape* of the execute rather than its duration.
    """
    assert cv2 is not None, "pip install opencv-python"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    x0, y0, x1, y1 = crop_box
    idxs = np.linspace(start_s * fps, end_s * fps, n_frames).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        if out_px:
            rgb = cv2.resize(rgb, (out_px, out_px), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()

    if len(frames) != n_frames:
        raise RuntimeError(f"got {len(frames)}/{n_frames} frames — check start/end times")
    return np.stack(frames).astype(np.uint8)


# ------------------------------------------------------------------------------ 3. labelling

class LabelTool:
    """Click every player icon in a frame, then press a key to move on.

    Controls
    --------
      left click   add an icon at the cursor
      right click  undo the last icon in this frame
      n / enter    next frame
      b            previous frame
      s            save now
      q            save and quit

    Labels are stored as normalised (x, y) in [0,1] relative to the crop, which is exactly
    what StratScout's `positions` contract expects, so nothing needs converting later.
    """

    def __init__(self, frames: np.ndarray, out_json: str, team_channel: str = "teammate"):
        self.frames = frames
        self.out = Path(out_json)
        self.team_channel = team_channel
        self.labels: Dict[int, List[List[float]]] = {}
        if self.out.exists():
            saved = json.loads(self.out.read_text())
            self.labels = {int(k): v for k, v in saved.get("points", {}).items()}
            print(f"resuming — {len(self.labels)} frames already labelled")
        self.i = 0

    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self.frames[self.i])
        pts = self.labels.get(self.i, [])
        h, w = self.frames[self.i].shape[:2]
        for x, y in pts:
            self.ax.plot(x * w, y * h, "o", mfc="none", mec="#00ff88", ms=14, mew=2)
        done = len(self.labels)
        self.ax.set_title(f"frame {self.i+1}/{len(self.frames)} — {len(pts)} icons "
                          f"({done} frames labelled)")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        h, w = self.frames[self.i].shape[:2]
        pts = self.labels.setdefault(self.i, [])
        if ev.button == 1:
            pts.append([float(ev.xdata) / w, float(ev.ydata) / h])
        elif ev.button == 3 and pts:
            pts.pop()
        self._draw()

    def _key(self, ev):
        if ev.key in ("n", "enter", " "):
            self.i = min(self.i + 1, len(self.frames) - 1)
        elif ev.key == "b":
            self.i = max(self.i - 1, 0)
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            self.save()
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            return
        self._draw()

    def save(self):
        self.out.write_text(json.dumps(
            {"team_channel": self.team_channel,
             "n_frames": int(len(self.frames)),
             "points": {str(k): v for k, v in self.labels.items()}}, indent=1))
        print(f"saved {sum(len(v) for v in self.labels.values())} icons "
              f"across {len(self.labels)} frames -> {self.out}")

    def run(self):
        import matplotlib.pyplot as plt
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._draw()
        plt.show()


def load_labels(frames: np.ndarray, label_json: str, max_icons: int = 5
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Read a LabelTool file into arrays the notebook can train on.

    Returns
    -------
    frames_out : uint8 (N, H, W, 3)   only the frames you actually labelled
    positions  : float32 (N, max_icons, 2) normalised, padded with NaN where you clicked
                 fewer than `max_icons` icons (a player may genuinely be dead)
    """
    data = json.loads(Path(label_json).read_text())
    pts = {int(k): v for k, v in data["points"].items() if v}
    idx = sorted(pts)
    out = np.full((len(idx), max_icons, 2), np.nan, dtype=np.float32)
    for row, f in enumerate(idx):
        p = np.array(pts[f], dtype=np.float32)[:max_icons]
        out[row, :len(p)] = p
    return frames[idx], out

