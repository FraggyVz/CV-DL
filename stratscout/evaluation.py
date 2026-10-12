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


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        pr, rc = (tp / max(tp + fp, 1), tp / max(tp + fn, 1))
        f1s.append(2 * pr * rc / max(pr + rc, 1e-09))
    return (float(np.mean(f1s)), np.array(f1s))


@torch.no_grad()
def evaluate(model, ds, *, context):
    model.eval()
    pred = model(ds.X.to(context.device)).cpu().argmax(-1)
    cm = confusion_matrix(ds.y, pred, ds.n_classes)
    mf1, per = macro_f1(cm)
    return {
        "accuracy": float((pred == ds.y).float().mean()),
        "macro_f1": mf1,
        "per_class_f1": per,
        "confusion": cm,
    }
