"""Visualisation: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import TEAM_COLOUR


def show_round(
    r: Dict,
    ax=None,
    bg: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    *,
    context,
):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 5))
    bg = context.state["PLOT_BG"] if bg is None else bg
    if bg is not None:
        ax.imshow(bg, extent=[0, 1, 1, 0])
    else:
        ax.set_facecolor("#0c1018")
    for team, sl in (
        ("T", slice(0, context.config.n_per_team)),
        ("CT", slice(context.config.n_per_team, None)),
    ):
        for p in r["positions"][:, sl].transpose(1, 0, 2):
            p = p[np.isfinite(p).all(1)]
            if len(p) < 2:
                continue
            ax.plot(p[:, 0], p[:, 1], color=TEAM_COLOUR[team], lw=1.1, alpha=0.8)
            ax.plot(p[0, 0], p[0, 1], "o", color=TEAM_COLOUR[team], ms=2.5)
    if r.get("plant_xy"):
        ax.plot(*r["plant_xy"], "x", color="red", ms=9, mew=2.2)
    lab = label if label is not None else r.get("label")
    ax.set_title(
        f"R{r['round_idx']}  {r['duration_s']:.0f}s  win {r['winner']}  plant {r['plant_site'] or '-'}"
        + (f"\n{lab}" if lab else ""),
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    if own:
        plt.show()


def show_match(demo: str, cols: int = 6, labels: Optional[Dict[int, str]] = None, *, context):
    rs = [r for r in context.state["DEM_ROUNDS"] if r["demo"] == demo]
    rows = int(np.ceil(len(rs) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, 2.8 * rows),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rs):
        ax.axis("on")
        show_round(r, ax, label=(labels or {}).get(r["round_idx"]), context=context)
    fig.suptitle(demo)
    plt.show()
