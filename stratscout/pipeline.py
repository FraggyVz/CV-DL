"""Pipeline: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PHASES
from .features import to_sequence
from .tracking import positions_from_video


class StratScoutPipeline:
    """Pixels in, two answers out. heads: phase -> (classifier, normaliser)."""

    def __init__(
        self,
        detector,
        heads: Dict[str, Tuple[nn.Module, Tuple]],
        repr_mode="features",
        *,
        context,
    ):
        self.context = context
        self.detector = detector.eval()
        self.heads = {ph: (m.eval(), norm) for ph, (m, norm) in heads.items()}
        self.repr_mode = repr_mode

    @torch.no_grad()
    def predict_window(self, frames: np.ndarray, phase: str = "pre") -> Dict:
        model, (mu, sd) = self.heads[phase]
        classes = PHASES[phase]
        pos = positions_from_video(self.detector, frames, context=self.context)
        seq = (
            torch.from_numpy(to_sequence(pos, self.repr_mode, context=self.context))[None] - mu
        ) / sd
        probs = F.softmax(model(seq.to(self.context.device)), -1)[0].cpu().numpy()
        k = int(probs.argmax())
        return {
            "label": classes[k],
            "confidence": float(probs[k]),
            "positions": pos,
            "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        }

    @torch.no_grad()
    def predict_video(self, frames: np.ndarray, stride: int = None, min_confidence: float = 0.0):
        w = self.context.config.seq_len
        stride = stride or w // 2
        out = []
        for s in range(0, max(len(frames) - w + 1, 1), stride):
            win = frames[s : s + w]
            if len(win) < w:
                break
            r = self.predict_window(win)
            if r["confidence"] >= min_confidence:
                out.append(
                    {
                        "start_time": s / self.context.config.fps_sampled,
                        "end_time": (s + w) / self.context.config.fps_sampled,
                        "label": r["label"],
                        "confidence": r["confidence"],
                    }
                )
        return out
