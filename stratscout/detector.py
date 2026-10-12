"""Detector: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic import _grid


class MinimapDetector(nn.Module):
    def __init__(self, n_classes: int = 2, width: int = 32, out_stride: int = None, *, context):
        self.context = context
        super().__init__()
        out_stride = self.context.config.det_stride if out_stride is None else out_stride
        w = width
        self.out_stride = out_stride
        self.enc = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1),
            nn.BatchNorm2d(w),
            nn.ReLU(True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(True),
            nn.Conv2d(w, w * 2, 3, 2, 1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(True),
            nn.Conv2d(w * 2, w * 2, 3, padding=1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(True),
        )
        dec = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(w * 2, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(True),
        ]
        head_ch = w
        if out_stride == 1:
            dec += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(w, w // 2, 3, padding=1),
                nn.BatchNorm2d(w // 2),
                nn.ReLU(True),
            ]
            head_ch = w // 2
        self.dec = nn.Sequential(*dec)
        self.head = nn.Conv2d(head_ch, n_classes, 1)
        self.head.bias.data.fill_(-2.19)

    def forward(self, x):
        return self.head(self.dec(self.enc(x)))


def gaussian_heatmap(
    pos: np.ndarray, out_px: int, sigma: float, n_channels: int = 2, *, context
) -> torch.Tensor:
    """positions (10, 2) -> (2, out_px, out_px). NaN or -1 = dead player, skipped."""
    xx, yy = _grid(out_px, context=context)
    hm = np.zeros((n_channels, out_px, out_px), np.float32)
    for p in range(pos.shape[0]):
        q = pos[p]
        if not np.isfinite(q).all() or q[0] < 0:
            continue
        ch = 0 if p < context.config.n_per_team else 1
        ix = int(np.clip(round(float(q[0]) * out_px), 0, out_px - 1))
        iy = int(np.clip(round(float(q[1]) * out_px), 0, out_px - 1))
        g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma**2))
        hm[ch] = np.maximum(hm[ch], g)
        hm[ch, iy, ix] = 1.0
    return torch.from_numpy(hm)


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(0.0001, 1 - 0.0001)
    pos = target.ge(0.999).float()
    neg = 1.0 - pos
    pos_loss = -((1 - p) ** 2) * torch.log(p) * pos
    neg_loss = -((1 - target) ** 4) * p**2 * torch.log(1 - p) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def _on_map(h: int, w: int, *, context) -> torch.Tensor:
    """DETECT_MASK at heatmap resolution, as a float tensor (1 on the map, 0 off it)."""
    m = (
        context.state["DETECT_MASK"]
        if context.state["DETECT_MASK"].shape == (h, w)
        else cv2.resize(
            context.state["DETECT_MASK"].astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    )
    return torch.from_numpy(m.astype(np.float32))


@torch.no_grad()
def decode_heatmap(logits: torch.Tensor, thresh: float, max_per_team: int, *, context):
    prob = torch.sigmoid(logits)
    peaks = (F.max_pool2d(prob, 3, 1, 1) == prob).float()
    B, C, h, w = prob.shape
    prob = prob * peaks * _on_map(h, w, context=context).to(prob.device)
    scores, idx = prob.view(B, C, -1).topk(max_per_team, dim=-1)
    ys = (idx // w).float() / h
    xs = (idx % w).float() / w
    out = []
    for b in range(B):
        d = {}
        for c, team in enumerate(context.config.teams[:C]):
            keep = scores[b, c] >= thresh
            d[team] = (
                torch.stack([xs[b, c][keep], ys[b, c][keep], scores[b, c][keep]], -1).cpu().numpy()
            )
        out.append(d)
    return out
