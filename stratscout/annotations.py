"""Annotations: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import CT_CLASSES, T_CLASSES
from .segmentation import DEFAULT_TEAM_HUES, _hue_dist, find_icons_teams
from .video import match_id_of
from .visualisation import TEAM_EDGE, show_detections


def auto_label_icons(
    frames: np.ndarray,
    out_json: str,
    team_hues=None,
    background=None,
    preview: int = 6,
    *,
    context,
) -> Dict:
    points = {
        i: find_icons_teams(f, team_hues, background, context=context) for i, f in enumerate(frames)
    }
    nT = np.array([sum((p[2] == 0 for p in v)) for v in points.values()])
    nC = np.array([sum((p[2] == 1 for p in v)) for v in points.values()])
    Path(out_json).write_text(
        json.dumps(
            {
                "schema": "xy_team",
                "n_frames": int(len(frames)),
                "points": {str(k): v for k, v in points.items() if v},
            },
            indent=1,
        )
    )
    print(f"wrote {out_json}")
    print(f"  T  mean {nT.mean():.2f}/frame, {int((nT == 5).sum())} frames with 5")
    print(f"  CT mean {nC.mean():.2f}/frame, {int((nC == 5).sum())} frames with 5")
    if preview:
        show_detections(
            frames,
            np.linspace(0, len(frames) - 1, preview).astype(int),
            team_hues=team_hues,
            background=background,
            title="auto-proposed labels — correct these in ReviewTool",
            context=context,
        )
    return points


class ReviewTool:
    """Correct machine-proposed icon labels. See the controls above."""

    def __init__(self, frames: np.ndarray, out_json: str, team_hues=None, *, context):
        self.context = context
        self.frames, self.out = (frames, Path(out_json))
        self.hues = team_hues or DEFAULT_TEAM_HUES
        self.labels: Dict[int, List[List[float]]] = {}
        if self.out.exists():
            d = json.loads(self.out.read_text())
            self.labels = {int(k): v for k, v in d.get("points", {}).items()}
            print(f"resuming — {len(self.labels)} frames already labelled")
        self.i = 0

    def _team_at(self, fx, fy) -> int:
        f = self.frames[self.i]
        h, w = f.shape[:2]
        px, py = (int(fx * w), int(fy * h))
        patch = f[max(py - 2, 0) : py + 3, max(px - 2, 0) : px + 3].reshape(1, -1, 3)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)[0]
        sat = hsv[:, 1].astype(float)
        hue = float(np.median(hsv[sat >= np.percentile(sat, 60), 0]))
        return self.context.config.teams.index(
            min(self.context.config.teams, key=lambda t: _hue_dist(hue, self.hues[t]))
        )

    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self.frames[self.i])
        h, w = self.frames[self.i].shape[:2]
        pts = self.labels.get(self.i, [])
        for x, y, t in pts:
            self.ax.plot(x * w, y * h, "o", mfc="none", mec=TEAM_EDGE[int(t)], ms=14, mew=2)
        nT = sum((int(p[2]) == 0 for p in pts))
        self.ax.set_title(f"frame {self.i + 1}/{len(self.frames)} — {nT} T, {len(pts) - nT} CT")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        h, w = self.frames[self.i].shape[:2]
        cx, cy = (float(ev.xdata) / w, float(ev.ydata) / h)
        pts = self.labels.setdefault(self.i, [])
        if ev.button == 1:
            pts.append([cx, cy, float(self._team_at(cx, cy))])
        elif ev.button == 3 and pts:
            pts.pop(int(np.argmin([(p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts])))
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
            plt.close(self.fig)
            return
        self._draw()

    def save(self):
        self.out.write_text(
            json.dumps(
                {
                    "schema": "xy_team",
                    "n_frames": int(len(self.frames)),
                    "points": {str(k): v for k, v in self.labels.items() if v},
                },
                indent=1,
            )
        )
        print(f"saved {len(self.labels)} frames -> {self.out}")

    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._draw()
        plt.show()


def rounds_json_for(video, *, context) -> Path:
    """labels/<match_id>.rounds.json - one label file per replay in data/."""
    return Path(context.config.labels_dir) / f"{match_id_of(video)}.rounds.json"


class RoundLabeller:
    """Keys: left/right scrub (shift = 10x) | [ ] round start / end | 1-5 attacker tactic |
    a / b bomb planted now at A / B | 6 7 8 defender response RETAKE TRICKLE SAVE | x clear plant |
    t / c winner | enter commit | u undo | s save | q save and quit."""

    def __init__(
        self,
        video: str,
        crop_box,
        out_json: str = None,
        step_s: float = 2.0,
        *,
        context,
    ):
        self.context = context
        assert cv2 is not None, "pip install opencv-python"
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise FileNotFoundError(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.dur = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.box, self.step = (crop_box, step_s)
        self.out = Path(out_json or rounds_json_for(video, context=self.context))
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.t = 0.0
        self._reset()
        self.rounds: List[Dict] = []
        if self.out.exists():
            self.rounds = json.loads(self.out.read_text())["rounds"]
            print(f"resuming - {len(self.rounds)} rounds already marked")
        print(f"video {self.dur:.0f}s @ {self.fps:.0f}fps")

    def _reset(self):
        self.start = self.end = self.label = self.winner = None
        self.plant = self.site = self.ct_label = None

    def _frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.t * self.fps))
        ok, bgr = self.cap.read()
        if not ok:
            return np.zeros((100, 100, 3), np.uint8)
        x0, y0, x1, y1 = self.box
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]

    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self._frame())
        f = lambda v: "-" if v is None else f"{v:.0f}s"
        plant = "-" if self.plant is None else f"{self.site} at {self.plant:.0f}s"
        self.ax.set_title(
            f"t={self.t:.0f}s / {self.dur:.0f}s   round {len(self.rounds) + 1}\nstart {f(self.start)}  end {f(self.end)}  attackers {self.label or '-'}\nplant {plant}  defenders {self.ct_label or '-'}  winner {self.winner or '-'}",
            fontsize=10,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _commit(self):
        if self.start is None or self.end is None or (not self.label):
            print("need start ([), end (]) and an attacker tactic (1-5)")
            return
        if self.end <= self.start:
            print("end must come after start")
            return
        if self.plant is not None and (not self.start < self.plant < self.end):
            print("the plant must fall between start and end")
            return
        if self.ct_label and self.plant is None:
            print("a defender response (6-8) needs a plant (a / b)")
            return
        if self.plant is not None and (not self.ct_label):
            print("note: plant marked but no defender response (6-8) - saved without one")
        self.rounds.append(
            {
                "round_idx": len(self.rounds),
                "start_s": round(self.start, 2),
                "end_s": round(self.end, 2),
                "label": self.label,
                "winner": self.winner,
                "plant_s": None if self.plant is None else round(self.plant, 2),
                "plant_site": self.site,
                "ct_label": self.ct_label,
            }
        )
        print(
            f"round {len(self.rounds)}: {self.label}  plant {self.site or '-'}  defenders {self.ct_label or '-'}  winner {self.winner or '?'}"
        )
        self._reset()

    def _key(self, ev):
        step = self.step * (10 if ev.key in ("shift+right", "shift+left") else 1)
        if ev.key in ("right", "shift+right"):
            self.t = min(self.t + step, self.dur - 1)
        elif ev.key in ("left", "shift+left"):
            self.t = max(self.t - step, 0.0)
        elif ev.key == "[":
            self.start = self.t
        elif ev.key == "]":
            self.end = self.t
        elif ev.key in list("12345"):
            self.label = T_CLASSES[int(ev.key) - 1]
        elif ev.key in ("a", "b"):
            self.plant, self.site = (self.t, ev.key.upper())
        elif ev.key in ("6", "7", "8"):
            self.ct_label = CT_CLASSES[int(ev.key) - 6]
        elif ev.key == "x":
            self.plant = self.site = self.ct_label = None
        elif ev.key in ("t", "c"):
            self.winner = "T" if ev.key == "t" else "CT"
        elif ev.key == "enter":
            self._commit()
        elif ev.key == "u" and self.rounds:
            print("removed", self.rounds.pop()["label"])
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            self.save()
            plt.close(self.fig)
            return
        self._draw()

    def save(self):
        self.out.write_text(
            json.dumps(
                {
                    "attacker_tactics": T_CLASSES,
                    "defender_responses": CT_CLASSES,
                    "rounds": self.rounds,
                },
                indent=1,
            )
        )
        print(
            f"saved {len(self.rounds)} rounds -> {self.out}   { {t: sum((r['label'] == t for r in self.rounds)) for t in T_CLASSES} }   { {t: sum((r.get('ct_label') == t for r in self.rounds)) for t in CT_CLASSES} }"
        )

    def run(self):
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []
        self.fig, self.ax = plt.subplots(figsize=(7, 7.8))
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._draw()
        plt.show()
