"""Training: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import torch

from .config import set_seed
from .detector import MinimapDetector, focal_heatmap_loss
from .evaluation import detector_metrics


def train_detector(epochs: int = None, *, context) -> MinimapDetector:
    epochs = context.config.det_epochs if epochs is None else epochs
    set_seed(context.config.seed)
    model = MinimapDetector(context=context).to(context.device)
    opt = torch.optim.AdamW(model.parameters(), lr=context.config.det_lr, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    context.state["DET_HISTORY"].clear()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y, _ in context.state["det_train_dl"]:
            x, y = (x.to(context.device), y.to(context.device))
            loss = focal_heatmap_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
        sched.step()
        m = detector_metrics(model, context.state["det_val_dl"], context=context)
        context.state["DET_HISTORY"].append(
            {"epoch": ep + 1, "loss": tot / len(context.state["det_train"]), **m}
        )
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(
                f"epoch {ep + 1:3d}/{epochs}  loss {tot / len(context.state['det_train']):.4f}  val P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}  err {m['mean_pos_err_norm'] * context.config.minimap_px:.2f}px"
            )
    return model
