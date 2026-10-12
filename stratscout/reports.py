"""Reports: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import CT_CLASSES, T_CLASSES
from .contracts import Round
from .synthetic import plot_background
from .tracking import positions_from_video
from .visualisation import TEAM_EDGE


def scouting_report(pipeline, rounds: List[Round], min_confidence: float = 0.0) -> Dict:
    rows = []
    for r in rounds:
        pre = pipeline.predict_window(r.frames, "pre")
        row = {
            "match_id": r.match_id,
            "round": r.round_idx,
            "true": r.label,
            "label": pre["label"],
            "confidence": pre["confidence"],
            "site": r.plant_site,
            "ct_true": r.ct_label,
            "ct_label": None,
        }
        if r.planted and r.post_frames is not None:
            row["ct_label"] = pipeline.predict_window(r.post_frames, "post")["label"]
        rows.append(row)
    kept = [r for r in rows if r["confidence"] >= min_confidence]
    planted = [r for r in rows if r["ct_label"]]
    return {
        "rows": rows,
        "n_used": len(kept),
        "n_planted": len(planted),
        "tendencies": {
            c: sum((r["label"] == c for r in kept)) / max(len(kept), 1) for c in T_CLASSES
        },
        "ct_tendencies": {
            c: sum((r["ct_label"] == c for r in planted)) / max(len(planted), 1) for c in CT_CLASSES
        },
        "agreement": sum((r["label"] == r["true"] for r in rows)) / max(len(rows), 1),
        "ct_agreement": sum((r["ct_label"] == r["ct_true"] for r in planted))
        / max(len(planted), 1),
    }


def plot_routes(rounds: List[Round], label: str, use_vision: bool = False, *, context):
    """Movement trails for one tactic — the view an analyst actually wants."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(plot_background(context=context), extent=[0, 1, 1, 0])
    for r in [x for x in rounds if x.label == label][:12]:
        pos = (
            positions_from_video(context.state["detector"], r.frames, context=context)
            if use_vision
            else r.positions
        )
        for p in range(context.config.n_per_team):
            ax.plot(pos[:, p, 0], pos[:, p, 1], color="#ff9020", alpha=0.35, lw=1.2)
    ax.set_title(f"{label} — attacker routes ({('vision' if use_vision else 'oracle')})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def round_positions(
    r: Round, use_vision: bool = False, detector=None, phase: str = "pre", *, context
) -> Optional[np.ndarray]:
    """(T, 10, 2) for one phase of one round. Vision positions are cached: one detector pass."""
    if not use_vision:
        return r.positions_of(phase)
    frames = r.frames_of(phase)
    if frames is None:
        return None
    key = (id(r), phase)
    if key not in context.state["_POS_CACHE"]:
        context.state["_POS_CACHE"][key] = positions_from_video(detector, frames, context=context)
    return context.state["_POS_CACHE"][key]


def team_slice(team: str, *, context) -> slice:
    return (
        slice(0, context.config.n_per_team)
        if team == "T"
        else slice(context.config.n_per_team, context.state["N_PLAYERS"])
    )


def side_of(label: str) -> str:
    """Defender responses belong to CT; attacker tactics to T."""
    return "CT" if label in CT_CLASSES else "T"


def phase_of(label: str) -> str:
    """Attacker tactics are read before the plant, defender responses after it."""
    return "post" if label in CT_CLASSES else "pre"


def radar_bg(*, context) -> np.ndarray:
    return plot_background(context=context)


def occupancy_map(
    rounds: List[Round],
    team: str = "T",
    label: Optional[str] = None,
    bins: int = 48,
    sigma: float = 1.2,
    use_vision: bool = False,
    detector=None,
    phase: str = "pre",
    *,
    context,
) -> np.ndarray:
    """-> (bins, bins) normalised occupancy density for one side in one phase of the round."""
    sel = [r for r in rounds if label is None or r.label_of(phase) == label]
    sl = team_slice(team, context=context)
    xs, ys = ([], [])
    for r in sel:
        pos = round_positions(r, use_vision, detector, phase, context=context)
        if pos is None:
            continue
        p = pos[:, sl].reshape(-1, 2)
        p = p[np.isfinite(p).all(1)]
        xs.append(p[:, 0])
        ys.append(p[:, 1])
    if not xs:
        return np.zeros((bins, bins), np.float32)
    H, _, _ = np.histogram2d(
        np.concatenate(ys), np.concatenate(xs), bins=bins, range=[[0, 1], [0, 1]]
    )
    if cv2 is not None and sigma > 0:
        H = cv2.GaussianBlur(H.astype(np.float32), (0, 0), sigma)
    return (H / H.max()).astype(np.float32) if H.max() > 0 else H.astype(np.float32)


def _heat_panel(ax, bg, H, team: str, title: str):
    ax.imshow(bg, extent=[0, 1, 1, 0])
    ax.imshow(
        H,
        extent=[0, 1, 1, 0],
        cmap="inferno" if team == "T" else "viridis",
        alpha=0.62,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_side_occupancy(
    rounds: List[Round],
    use_vision: bool = False,
    detector=None,
    bins: int = 48,
    *,
    context,
):
    """T (attackers) and CT (defenders) occupancy over all rounds."""
    bg = radar_bg(context=context)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), layout="constrained")
    for ax, team in zip(axes, ("T", "CT")):
        H = occupancy_map(
            rounds,
            team,
            None,
            bins,
            use_vision=use_vision,
            detector=detector,
            context=context,
        )
        _heat_panel(
            ax,
            bg,
            H,
            team,
            f"{('attackers (T)' if team == 'T' else 'defenders (CT)')}\nn={len(rounds)} rounds",
        )
    fig.suptitle(
        f"18.1 — side occupancy before the plant ({('vision' if use_vision else 'oracle')} positions)"
    )
    plt.show()


def show_tactic_occupancy(
    rounds: List[Round],
    labels: Sequence[str],
    phase: str,
    use_vision: bool = False,
    detector=None,
    bins: int = 48,
    *,
    context,
):
    """2 x len(labels) grid: rows T / CT, columns labels. The side running it is marked with *."""
    bg = radar_bg(context=context)
    fig, axes = plt.subplots(
        2,
        len(labels),
        figsize=(3.0 * len(labels), 6.6),
        squeeze=False,
        layout="constrained",
    )
    for col, lab in enumerate(labels):
        n = sum((r.label_of(phase) == lab for r in rounds))
        for row, team in enumerate(("T", "CT")):
            H = occupancy_map(
                rounds,
                team,
                lab,
                bins,
                use_vision=use_vision,
                detector=detector,
                phase=phase,
                context=context,
            )
            star = " *" if team == side_of(lab) else ""
            _heat_panel(
                axes[row, col],
                bg,
                H,
                team,
                f"{lab} — {team}{star}\nn={n}" + ("  (no rounds)" if n == 0 else ""),
            )
    what = (
        "attacker tactic, before the plant"
        if phase == "pre"
        else "defender response, after the plant"
    )
    fig.suptitle(
        f"18.2 — occupancy per {what}; * = side running it ({('vision' if use_vision else 'oracle')} positions)"
    )
    plt.show()


def _draw_trails(
    ax,
    bg,
    rounds: List[Round],
    label: str,
    team: str,
    max_rounds: int,
    use_vision: bool,
    detector,
    *,
    context,
):
    phase = phase_of(label)
    sl = team_slice(team, context=context)
    colour = TEAM_EDGE[0] if team == "T" else TEAM_EDGE[1]
    sel = [r for r in rounds if r.label_of(phase) == label][:max_rounds]
    ax.imshow(bg, extent=[0, 1, 1, 0])
    for r in sel:
        pos = round_positions(r, use_vision, detector, phase, context=context)
        if pos is None:
            continue
        for pi in range(sl.start, sl.stop):
            p = pos[:, pi]
            p = p[np.isfinite(p).all(1)]
            if len(p) < 2:
                continue
            ax.plot(p[:, 0], p[:, 1], color=colour, alpha=0.35, lw=1.1)
            ax.plot(p[0, 0], p[0, 1], "o", color=colour, ms=2.5, alpha=0.6)
    total = sum((r.label_of(phase) == label for r in rounds))
    shown = f"n={len(sel)}" + (f" of {total}" if total > len(sel) else "")
    when = "before the plant" if phase == "pre" else "after the plant"
    ax.set_title(
        f"{label} — {('attackers' if team == 'T' else 'defenders')} {when}\n{shown}"
        + ("  (no rounds)" if total == 0 else ""),
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def show_trails(
    rounds: List[Round],
    label: str,
    team: Optional[str] = None,
    max_rounds: int = 15,
    use_vision: bool = False,
    detector=None,
    *,
    context,
):
    """One tactic or response, large. team=None -> the side running it."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    _draw_trails(
        ax,
        radar_bg(context=context),
        rounds,
        label,
        team or side_of(label),
        max_rounds,
        use_vision,
        detector,
        context=context,
    )
    plt.tight_layout()
    plt.show()


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval — behaves properly at small n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(centre - half, 0.0), min(centre + half, 1.0))


def win_rate_report(rounds: List[Round], min_n: int = 20) -> Dict:
    scored = [r for r in rounds if r.winner is not None]
    if not scored:
        print("no rounds carry a winner — label them in RoundLabeller (t / c keys)")
        return {}
    t_wins = sum((r.winner == "T" for r in scored))
    p, lo, hi = wilson_interval(t_wins, len(scored))
    print(f"OVERALL  ({len(scored)} rounds)")
    print(f"  T side wins   {p * 100:5.1f}%   95% CI [{lo * 100:.1f}, {hi * 100:.1f}]")
    print(
        f"  CT side wins  {(1 - p) * 100:5.1f}%   95% CI [{(1 - hi) * 100:.1f}, {(1 - lo) * 100:.1f}]"
    )
    verdict = (
        "too few rounds to call"
        if len(scored) < min_n
        else "T-sided"
        if lo > 0.5
        else "CT-sided"
        if hi < 0.5
        else "no significant side advantage — the interval spans 50%"
    )
    print(f"  verdict: {verdict}\n")
    out = {
        "overall": {
            "n": len(scored),
            "t_win_rate": p,
            "ci": (lo, hi),
            "verdict": verdict,
        },
        "by_tactic": {},
        "by_response": {},
    }
    groups = (
        (
            "by_tactic",
            "BY ATTACKER TACTIC - did the attackers win the round?",
            T_CLASSES,
            lambda r: r.label,
            lambda r: r.tactic_worked,
        ),
        (
            "by_response",
            "BY DEFENDER RESPONSE - after a plant, did the defenders win?",
            CT_CLASSES,
            lambda r: r.ct_label,
            lambda r: r.response_worked,
        ),
    )
    for key, head, labels, get, worked in groups:
        print(head)
        print(f"  {'label':<10}{'n':>5}{'success':>10}{'95% CI':>18}")
        for c in labels:
            sub = [r for r in scored if get(r) == c]
            if not sub:
                continue
            w = sum((bool(worked(r)) for r in sub))
            pp, lower, upper = wilson_interval(w, len(sub))
            flag = "" if len(sub) >= min_n else "  <- too few to quote"
            print(
                f"  {c:<10}{len(sub):>5}{pp * 100:>9.1f}%   [{lower * 100:5.1f}, {upper * 100:5.1f}]   {'#' * int(pp * 22)}{flag}"
            )
            out[key][c] = {"n": len(sub), "success_rate": pp, "ci": (lower, upper)}
        print()
    return out


def plot_win_rates(report: Dict, min_n: int = 20):
    panels = [
        ("by_tactic", "attacker tactics: did T win?", "#d9822b"),
        ("by_response", "defender responses after a plant: did CT win?", "#2f7fd0"),
    ]
    panels = [p for p in panels if report.get(p[0])]
    if not panels:
        return
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.4 * len(panels), 3.6),
        squeeze=False,
        gridspec_kw={"width_ratios": [len(report[p[0]]) for p in panels]},
    )
    for ax, (key, title, colour) in zip(axes[0], panels):
        bt = report[key]
        names = list(bt)
        vals = [bt[c]["success_rate"] for c in names]
        los = [vals[i] - bt[c]["ci"][0] for i, c in enumerate(names)]
        his = [bt[c]["ci"][1] - vals[i] for i, c in enumerate(names)]
        cols = [colour if bt[c]["n"] >= min_n else "#c8c8c8" for c in names]
        ax.bar(names, vals, yerr=[los, his], capsize=5, color=cols)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        for k, c in enumerate(names):
            ax.text(k, 0.04, f"n={bt[c]['n']}", ha="center", fontsize=8, color="white")
        ax.tick_params(axis="x", labelrotation=20)
    axes[0][0].set_ylabel("success rate")
    fig.suptitle(
        "Objective 3 — success with 95% Wilson intervals (grey = too few rounds to quote)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


def rounds_needed(target_halfwidth: float = 0.1, p: float = 0.5, z: float = 1.96) -> int:
    return int(math.ceil(z * z * p * (1 - p) / target_halfwidth**2))
