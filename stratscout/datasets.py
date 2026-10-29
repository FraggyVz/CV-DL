"""Datasets: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import Round
from .detector import gaussian_heatmap


def load_icon_labels(frames: np.ndarray, label_json: str, n_per_team: int = 5):
    """-> (frames_kept, positions (N, 10, 2)); slots 0-4 = T, 5-9 = CT, NaN where absent."""
    data = json.loads(Path(label_json).read_text())
    pts = {int(k): v for k, v in data["points"].items() if v}
    idx = sorted(pts)
    out = np.full((len(idx), n_per_team * 2, 2), np.nan, np.float32)
    for row, f in enumerate(idx):
        for ti in (0, 1):
            team = sorted([p[:2] for p in pts[f] if int(p[2]) == ti])[:n_per_team]
            if team:
                out[row, ti * n_per_team : ti * n_per_team + len(team)] = np.array(team, np.float32)
    return (frames[idx], out)


class IconDataset(Dataset):
    def __init__(self, frames: np.ndarray, positions: np.ndarray, *, context):
        self.context = context
        self.frames, self.positions = (frames, positions)
        self.out_px = self.context.config.minimap_px // self.context.config.det_stride

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        x = self.context.state["PRE"](self.frames[i])[0]
        y = gaussian_heatmap(
            self.positions[i],
            self.out_px,
            self.context.config.det_sigma,
            context=self.context,
        )
        return (x, y, torch.from_numpy(np.nan_to_num(self.positions[i], nan=-1.0)))


def synthetic_icon_dataset(
    rounds: List[Round], n_frames: int, seed: int, *, context
) -> IconDataset:
    rng = np.random.default_rng(seed)
    picks = [(r, t) for r in rounds for t in range(r.positions.shape[0])]
    sel = rng.permutation(len(picks))[:n_frames]
    fr = np.stack([picks[i][0].frames[picks[i][1]] for i in sel])
    po = np.stack([picks[i][0].positions[picks[i][1]] for i in sel])
    return IconDataset(fr, po, context=context)
