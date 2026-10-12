"""Annotations: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from .config import CLASSES, EXCLUDED
from .visualisation import show_round


def demo_labels_path(demo: str, *, context) -> Path:
    return Path(context.config.labels_dir) / f"{demo}.demo_rounds.json"


def load_demo_labels(demo: str, *, context) -> Dict[int, str]:
    p = demo_labels_path(demo, context=context)
    return {int(k): v for k, v in json.loads(p.read_text())["rounds"].items()} if p.exists() else {}


class DemoRoundLabeller:
    KEYS = {str(i + 1): c for i, c in enumerate(CLASSES)} | {"x": EXCLUDED}

    def __init__(self, demo: str, *, context):
        self.context = context
        self.demo = demo
        self.rounds = [r for r in self.context.state["DEM_ROUNDS"] if r["demo"] == demo]
        if not self.rounds:
            raise ValueError(f"no parsed rounds for {demo}")
        self.labels = load_demo_labels(demo, context=self.context)
        unl = [i for i, r in enumerate(self.rounds) if r["round_idx"] not in self.labels]
        self.i = unl[0] if unl else 0
        print(f"{demo}: {len(self.rounds)} rounds, {len(self.labels)} already labelled")

    def save(self):
        p = demo_labels_path(self.demo, context=self.context)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "tactics": CLASSES,
                    "rounds": {str(k): v for k, v in sorted(self.labels.items())},
                },
                indent=1,
            )
        )
        counts = {c: sum((v == c for v in self.labels.values())) for c in CLASSES + [EXCLUDED]}
        print(f"saved {len(self.labels)}/{len(self.rounds)} -> {p}   {counts}")

    def _draw(self):
        r = self.rounds[self.i]
        self.ax.clear()
        show_round(
            r,
            self.ax,
            label=self.labels.get(r["round_idx"], "— unlabelled —"),
            context=self.context,
        )
        self.ax.set_xlabel(
            f"1 A_RUSH  2 B_RUSH  3 A_SPLIT  4 B_SPLIT  5 RETAKE  x exclude\nround {self.i + 1}/{len(self.rounds)}   labelled {len(self.labels)}",
            fontsize=8,
        )
        self.fig.canvas.draw_idle()

    def _key(self, ev):
        rid = self.rounds[self.i]["round_idx"]
        if ev.key in self.KEYS:
            self.labels[rid] = self.KEYS[ev.key]
            self.i = min(self.i + 1, len(self.rounds) - 1)
        elif ev.key == "backspace":
            self.labels.pop(rid, None)
        elif ev.key == "right":
            self.i = min(self.i + 1, len(self.rounds) - 1)
        elif ev.key == "left":
            self.i = max(self.i - 1, 0)
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            self.save()
            plt.close(self.fig)
            return
        self._draw()

    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(6.5, 7))
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._draw()
        plt.show()
