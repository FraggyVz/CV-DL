"""Visualisation: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CLASSES
from .detector import decode_heatmap
from .segmentation import find_icons_teams
from .video import crop_change


def show_pipeline():
    """The pipeline as a diagram: each stage, and the data contract it hands to the next."""
    from matplotlib.patches import FancyBboxPatch

    stages = [
        ("Replay", "recorded .mp4"),
        ("Auto-crop", "find the radar"),
        ("Detector", "CNN, Model 1"),
        ("Tracker", "10 fixed slots"),
        ("Classifier", "Model 2: two\nquestions"),
        ("Report", "tactics, heatmaps,\nwin rates"),
    ]
    contracts = [
        "frames\n(N,H,W,3)",
        "x\n(N,3,128,128)",
        "detections\n{T, CT}",
        "positions\n(N,10,2)",
        "logits\n(B,5) + (B,3)",
    ]
    learned = {"Detector", "Classifier"}
    fig, ax = plt.subplots(figsize=(13, 3.0))
    w, gap = (1.55, 0.62)
    for k, (name, sub) in enumerate(stages):
        x = k * (w + gap)
        edge = "#d9822b" if name in learned else "#5b6b7c"
        ax.add_patch(
            FancyBboxPatch(
                (x, 0),
                w,
                1.0,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc="#eef1f4",
                ec=edge,
                lw=2.2,
            )
        )
        ax.text(x + w / 2, 0.64, name, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x + w / 2, 0.3, sub, ha="center", va="center", fontsize=9.5, color="#44505c")
        if k < len(contracts):
            ax.annotate(
                "",
                xy=(x + w + gap - 0.04, 0.5),
                xytext=(x + w + 0.04, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#33404d", lw=1.6),
            )
            ax.text(
                x + w + gap / 2,
                -0.32,
                contracts[k],
                ha="center",
                va="center",
                fontsize=8.5,
                family="monospace",
                color="#33404d",
            )
    ax.set_xlim(-0.1, len(stages) * (w + gap) - gap + 0.1)
    ax.set_ylim(-0.7, 1.15)
    ax.axis("off")
    ax.set_title(
        "StratScout pipeline: pixels in, tactics out (orange outline = learned model; text under arrows = data contract)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def verify_crop(frames: np.ndarray, box: Tuple[int, int, int, int], n_show: int = 4) -> float:
    x0, y0, x1, y1 = box
    idx = np.linspace(0, len(frames) - 1, n_show).astype(int)
    crops = [frames[i][y0:y1, x0:x1] for i in idx]
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3.1 * (n_show + 1), 3.4))
    for ax, c, i in zip(axes, crops, idx):
        ax.imshow(c)
        ax.set_title(f"probe {i}")
        ax.axis("off")
    diff = np.abs(crops[0].astype(int) - crops[-1].astype(int)).sum(-1)
    axes[-1].imshow(diff, cmap="inferno")
    axes[-1].set_title("difference\n(dots only)")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()
    changed = crop_change(frames, box)
    print(f"{changed * 100:.1f}% of pixels typically differ from the median radar.")
    print("Under ~10% = locked radar (the rest is players and the bomb-radius ring).")
    return changed


def show_crop_on_screen(frame: np.ndarray, box: Tuple[int, int, int, int], name: str = ""):
    """The full replay screen with the auto-detected radar box drawn on it."""
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = box
    fig, ax = plt.subplots(figsize=(10, 10 * frame.shape[0] / frame.shape[1]))
    ax.imshow(frame)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#00e676", lw=2.5))
    ax.text(
        x0,
        y1 + 40,
        f"radar found automatically · box {box}",
        color="#00e676",
        fontsize=11,
        weight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#101722", ec="none", alpha=0.85),
    )
    ax.set_title(f"Replay screen {name} — the only part StratScout reads", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


TEAM_EDGE = {0: "#ff9020", 1: "#40a0ff"}


def show_detections(
    frames: np.ndarray,
    idxs: Sequence[int] = None,
    n: int = 6,
    team_hues=None,
    background=None,
    title: str = "classical icon finder",
    *,
    context,
):
    """Circle every detected icon on a strip of frames (orange = T, blue = CT)."""
    idxs = idxs if idxs is not None else np.linspace(0, len(frames) - 1, n).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3.0 * len(idxs), 3.5))
    for ax, i in zip(np.atleast_1d(axes), idxs):
        f = frames[int(i)]
        pts = find_icons_teams(f, team_hues, background, context=context)
        ax.imshow(f)
        h, w = f.shape[:2]
        for x, y, t in pts:
            ax.plot(x * w, y * h, "o", mfc="none", mec=TEAM_EDGE[int(t)], ms=13, mew=2)
        nT = sum((int(p[2]) == 0 for p in pts))
        ax.set_title(f"f{i} — {nT}T / {len(pts) - nT}CT", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_mask_pipeline(frames: np.ndarray, i: int = 0, team_hues=None, background=None, *, context):
    """Frame -> saturation -> mask -> detections, so threshold problems are visible."""
    f = frames[int(i)]
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= context.config.sat_min) & (hsv[:, :, 2] >= context.config.val_min)
    if background is not None:
        mask &= (
            np.abs(f.astype(np.int16) - background.astype(np.int16)).sum(-1)
            > context.config.bg_thresh
        )
    pts = find_icons_teams(f, team_hues, background, context=context)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(f)
    axes[0].set_title(f"frame {i}")
    axes[1].imshow(hsv[:, :, 1], cmap="viridis")
    axes[1].set_title("saturation")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(
        f"mask (S>={context.config.sat_min}, V>={context.config.val_min}{(', bg-sub' if background is not None else '')})"
    )
    axes[3].imshow(f)
    axes[3].set_title(f"found {len(pts)}")
    h, w = f.shape[:2]
    for x, y, t in pts:
        axes[3].plot(x * w, y * h, "o", mfc="none", mec=TEAM_EDGE[int(t)], ms=13, mew=2)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print("too many blobs -> raise sat_min or narrow icon_area | missing icons -> lower them")


def show_counts(frames: np.ndarray, team_hues=None, background=None, step: int = 1, *, context):
    idx = range(0, len(frames), step)
    nT, nC = ([], [])
    for i in idx:
        pts = find_icons_teams(frames[i], team_hues, background, context=context)
        nT.append(sum((int(p[2]) == 0 for p in pts)))
        nC.append(len(pts) - nT[-1])
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(list(idx), nT, lw=1.2, color=TEAM_EDGE[0], label="T (orange)")
    ax.plot(list(idx), nC, lw=1.2, color=TEAM_EDGE[1], label="CT (blue)")
    ax.axhline(5, ls="--", c="grey", lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("icons found")
    ax.set_ylim(-0.3, 6.5)
    ax.legend(fontsize=8)
    ax.set_title("icons per frame — should hover near 5 and dip on deaths")
    plt.tight_layout()
    plt.show()
    print(
        f"mean  T {np.mean(nT):.2f}   CT {np.mean(nC):.2f}   frames with 5T+5CT: {sum((a == 5 and b == 5 for a, b in zip(nT, nC)))}/{len(nT)}"
    )


def show_training_curve(history: List[Dict[str, float]], *, context):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    a1.plot(ep, [h["loss"] for h in history], color="#33404d", lw=2)
    a1.set_title("training loss (lower is better)")
    a1.set_xlabel("epoch")
    a1.grid(alpha=0.3)
    a2.plot(
        ep,
        [h["recall"] * 100 for h in history],
        color="#d9822b",
        lw=2.2,
        label="recall: share of players found",
    )
    a2.plot(
        ep,
        [h["precision"] * 100 for h in history],
        color="#2f7fd0",
        lw=2.2,
        label="precision: share of detections that are real",
    )
    a2.set_ylim(0, 102)
    a2.set_ylabel("%")
    a2.set_xlabel("epoch")
    a2.grid(alpha=0.3)
    a2.set_title("validation precision and recall")
    a2.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"Model 1 training ({('real' if context.state['USING_REAL_ICONS'] else 'synthetic')} icon labels)"
    )
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_detector_output(
    model, frames: np.ndarray, idxs=None, n: int = 5, show_heatmap=True, *, context
):
    model.eval()
    idxs = idxs if idxs is not None else np.linspace(0, len(frames) - 1, n).astype(int)
    x = context.state["PRE"](frames[idxs]).to(context.device)
    logits = model(x)
    prob = torch.sigmoid(logits).cpu().numpy()
    dets = decode_heatmap(
        logits,
        context.config.det_peak_thresh,
        context.config.n_per_team,
        context=context,
    )
    rows = 2 if show_heatmap else 1
    fig, axes = plt.subplots(rows, len(idxs), figsize=(3.0 * len(idxs), 3.3 * rows), squeeze=False)
    for k, i in enumerate(idxs):
        ax = axes[0][k]
        ax.imshow(frames[i])
        h, w = frames[i].shape[:2]
        nT = 0
        for c, team in enumerate(context.config.teams):
            for x_, y_, s_ in dets[k][team]:
                ax.plot(x_ * w, y_ * h, "o", mfc="none", mec=TEAM_EDGE[c], ms=13, mew=2)
                nT += c == 0
        ax.set_title(
            f"f{i} — {nT}T / {len(dets[k]['T']) + len(dets[k]['CT']) - nT}CT",
            fontsize=10,
        )
        ax.axis("off")
        if show_heatmap:
            axes[1][k].imshow(prob[k].max(0), cmap="hot", vmin=0, vmax=1)
            axes[1][k].set_title("heatmap (max of both teams)", fontsize=8)
            axes[1][k].axis("off")
    fig.suptitle("Model 1 — CNN detections (orange = T, blue = CT)")
    plt.tight_layout()
    plt.show()


def plot_confusion(cm, title, classes: Sequence[str] = CLASSES):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(5.2, 4.4) if n > 3 else (4.2, 3.7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
