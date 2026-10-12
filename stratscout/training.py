"""Training: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import set_seed
from .detector import MinimapDetector, focal_heatmap_loss
from .evaluation import detector_metrics, evaluate
from .models import MODEL_ZOO


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


def train_classifier(name, train_ds, val_ds, verbose=True, *, context):
    set_seed(context.config.seed)
    model = MODEL_ZOO[name](
        train_ds.X.shape[-1],
        context.config.cls_hidden,
        train_ds.n_classes,
        context=context,
    ).to(context.device)
    opt = torch.optim.AdamW(model.parameters(), lr=context.config.cls_lr, weight_decay=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=context.config.cls_epochs)
    dl = DataLoader(train_ds, batch_size=context.config.cls_batch, shuffle=True)
    best_f1, best_state = (-1.0, None)
    for ep in range(context.config.cls_epochs):
        model.train()
        for xb, yb in dl:
            loss = F.cross_entropy(model(xb.to(context.device)), yb.to(context.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m = evaluate(model, val_ds, context=context)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1:3d}  val acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}")
    model.load_state_dict(best_state)
    return (model, evaluate(model, val_ds, context=context))
