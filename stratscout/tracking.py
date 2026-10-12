"""Tracking: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .detector import decode_heatmap


class FixedSlotTracker:
    def __init__(self, n_slots: int, gate: float, max_miss: int, smooth: float = 0.6):
        self.n_slots, self.gate, self.max_miss, self.smooth = (
            n_slots,
            gate,
            max_miss,
            smooth,
        )

    def run(self, dets_per_frame: List[np.ndarray]) -> np.ndarray:
        n = len(dets_per_frame)
        out = np.full((n, self.n_slots, 2), np.nan, np.float32)
        pos: List[Optional[np.ndarray]] = [None] * self.n_slots
        vel = [np.zeros(2, np.float32) for _ in range(self.n_slots)]
        miss = [0] * self.n_slots
        for f, det in enumerate(dets_per_frame):
            pts = det[:, :2].astype(np.float32) if len(det) else np.zeros((0, 2), np.float32)
            active = [s for s in range(self.n_slots) if pos[s] is not None]
            pred = {s: pos[s] + vel[s] for s in active}
            pairs = sorted(
                (
                    (float(np.linalg.norm(pts[d] - pred[s])), s, d)
                    for s in active
                    for d in range(len(pts))
                )
            )
            used_s, used_d = (set(), set())
            for dist, s, d in pairs:
                if s in used_s or d in used_d or dist > self.gate:
                    continue
                new = pts[d]
                vel[s] = self.smooth * vel[s] + (1 - self.smooth) * (new - pos[s])
                pos[s], miss[s] = (new, 0)
                used_s.add(s)
                used_d.add(d)
            for d in (i for i in range(len(pts)) if i not in used_d):
                empty = [s for s in range(self.n_slots) if pos[s] is None]
                if empty:
                    s = empty[0]
                else:
                    lost = [s for s in range(self.n_slots) if s not in used_s and miss[s] > 0]
                    if not lost:
                        continue
                    s = max(lost, key=lambda k: miss[k])
                pos[s], vel[s], miss[s] = (pts[d], np.zeros(2, np.float32), 0)
                used_s.add(s)
            for s in range(self.n_slots):
                if s in used_s or pos[s] is None:
                    continue
                miss[s] += 1
                if miss[s] <= self.max_miss:
                    pos[s] = np.clip(pos[s] + vel[s] * 0.5, 0.0, 1.0)
                else:
                    vel[s] = np.zeros(2, np.float32)
            for s in range(self.n_slots):
                if pos[s] is not None:
                    out[f, s] = pos[s]
        for s in range(self.n_slots):
            known = np.where(~np.isnan(out[:, s, 0]))[0]
            if len(known) == 0:
                out[:, s] = 0.5
            else:
                out[: known[0], s] = out[known[0], s]
        order = np.argsort(out[:, :, 0].mean(axis=0))
        return out[:, order]


@torch.no_grad()
def positions_from_video(model: nn.Module, frames: np.ndarray, *, context) -> np.ndarray:
    """Full vision front-end: pixels -> (T, 10, 2). This is the deployed path."""
    model.eval()
    x = context.state["PRE"](frames).to(context.device)
    dets = decode_heatmap(
        model(x),
        context.config.det_peak_thresh,
        context.config.n_per_team,
        context=context,
    )
    per_team = []
    for team in context.config.teams:
        tr = FixedSlotTracker(
            context.config.n_per_team,
            context.config.track_gate,
            context.config.track_max_miss,
        )
        per_team.append(tr.run([d[team] for d in dets]))
    return np.concatenate(per_team, axis=1)


def positional_error(est: np.ndarray, gt: np.ndarray, *, context) -> float:
    errs = []
    for t in range(est.shape[0]):
        for c in range(2):
            sl = slice(c * context.config.n_per_team, (c + 1) * context.config.n_per_team)
            d = np.linalg.norm(est[t, sl][:, None, :] - gt[t, sl][None, :, :], axis=-1)
            errs.extend(d.min(axis=1).tolist())
    return float(np.mean(errs))
