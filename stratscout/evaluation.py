"""Evaluation: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .detector import decode_heatmap


@torch.no_grad()
def detector_metrics(model: nn.Module, loader: DataLoader, *, context) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    errs: List[float] = []
    for x, _, pos in loader:
        dets = decode_heatmap(
            model(x.to(context.device)),
            context.config.det_peak_thresh,
            context.config.n_per_team,
            context=context,
        )
        for b in range(x.shape[0]):
            for c, team in enumerate(context.config.teams):
                gt = pos[
                    b,
                    c * context.config.n_per_team : (c + 1) * context.config.n_per_team,
                ].numpy()
                gt = gt[np.isfinite(gt[:, 0]) & (gt[:, 0] >= 0)]
                pred = dets[b][team]
                if len(gt) == 0:
                    fp += len(pred)
                    continue
                used: set = set()
                order = np.argsort(-pred[:, 2]) if len(pred) else []
                for pi in order:
                    d = np.linalg.norm(gt - pred[pi, :2], axis=1)
                    d[list(used)] = np.inf
                    j = int(np.argmin(d))
                    if d[j] <= context.config.det_match_radius:
                        tp += 1
                        used.add(j)
                        errs.append(float(d[j]))
                    else:
                        fp += 1
                fn += len(gt) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-09),
        "mean_pos_err_norm": float(np.mean(errs)) if errs else float("nan"),
    }
