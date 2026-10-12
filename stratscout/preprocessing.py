"""Preprocessing: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimapPreprocessor(nn.Module):
    def __init__(self, out_px: int):
        super().__init__()
        self.out_px = out_px

    def forward(self, frames_hwc) -> torch.Tensor:
        x = torch.as_tensor(np.ascontiguousarray(frames_hwc))
        if x.ndim == 3:
            x = x[None]
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.out_px or x.shape[-2] != self.out_px:
            x = F.interpolate(
                x, size=(self.out_px, self.out_px), mode="bilinear", align_corners=False
            )
        return x
