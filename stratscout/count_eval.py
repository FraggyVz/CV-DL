"""Count evaluation: how often each team shows all five players when all five are alive.

No labels are needed: during freeze time and the first seconds after it nobody has died,
so a correct detector shows 5 T and 5 CT. A hidden bomb carrier counts for the T side when
the badge shows a carrier at its rim. Split parameters are fitted on training matches and
scored on held-out matches. This is a proxy for recall in crowded early-round frames, not
labelled precision/recall, and later-round deaths are not covered.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .bomb import SPECTATOR_CT_HUE, SPECTATOR_T_HUE, carrier_rim, carrier_visible, find_bomb
from .config import Config
from .context import RunContext
from .icon_split import SplitParams, icons_in_blob, team_blobs
from .rounds import cached_rounds, video_background
from .segmentation import find_icons_teams
from .storage import write_new_text

HUES = {"T": SPECTATOR_T_HUE, "CT": SPECTATOR_CT_HUE}
PHASES = ("freeze", "early")


def all_alive_windows(rounds: List[Dict], early_seconds: float = 8.0) -> List[tuple]:
    """(phase, start, end) seconds: each detected freeze, then early_seconds after it.
    A partial first round has no observed freeze and is skipped."""
    out = []
    for r in rounds:
        if r["partial"] or r["freeze"] <= 0:
            continue
        end = r["start"] + r["freeze"]
        out += [("freeze", r["start"], end), ("early", end, min(end + early_seconds, r["end"]))]
    return out


def collect(video: Path, early_seconds: float = 8.0) -> List[Dict]:
    """Per-frame records inside the all-alive windows of one prepared video: same-team blob
    areas, whether the bomb shows a carrier, and the original baseline's team counts."""
    baseline_ctx = RunContext(Config(icon_area=(120, 450), hue_tol=8.0))
    ctx = RunContext()
    windows = all_alive_windows(cached_rounds(video), early_seconds)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    background = cv2.cvtColor(
        video_background(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), cv2.COLOR_BGR2RGB
    )
    records = []
    try:
        for phase, a, b in windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a * fps)))
            for i in range(int(round(a * fps)), int(round(b * fps))):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bomb = find_bomb(rgb, context=ctx)
                carried = bomb is not None and carrier_visible(
                    carrier_rim(rgb, bomb, background=background, context=ctx)
                )
                base = find_icons_teams(rgb, HUES, background, context=baseline_ctx)
                records.append(
                    {
                        "frame": i,
                        "phase": phase,
                        "carried": bool(carried),
                        "areas": [
                            (b["team"], b["area"])
                            for b in team_blobs(rgb, HUES, background, context=ctx)
                        ],
                        "baseline": [sum(p[2] == t for p in base) for t in (0, 1)],
                    }
                )
    finally:
        cap.release()
    return records


def counts(record: Dict, params: SplitParams = None) -> List[int]:
    """[T, CT] for one record: the baseline's counts when params is None, else the split
    estimate. Either way a carrier hidden under the badge adds one T."""
    if params is None:
        t, ct = record["baseline"]
    else:
        t = sum(icons_in_blob(a, params) for team, a in record["areas"] if team == 0)
        ct = sum(icons_in_blob(a, params) for team, a in record["areas"] if team == 1)
    return [t + int(record["carried"]), ct]


def score(records: Sequence[Dict], params: SplitParams = None) -> Dict:
    """Share of frames showing exactly 5, mean count and mean absolute error, per team and
    phase, plus 'mean_exact5' over the four team-phase cells."""
    out, cells = {}, []
    for phase in PHASES:
        rows = np.array([counts(r, params) for r in records if r["phase"] == phase])
        if not len(rows):
            continue
        for ti, team in enumerate(("T", "CT")):
            c = rows[:, ti]
            exact = float(np.mean(c == 5))
            out[f"{phase}_{team}"] = {
                "frames": int(len(c)),
                "exact5": round(exact, 4),
                "mean": round(float(c.mean()), 3),
                "mae": round(float(np.abs(c - 5).mean()), 3),
            }
            cells.append(exact)
    out["mean_exact5"] = round(float(np.mean(cells)), 4) if cells else 0.0
    return out


def fit(records: Sequence[Dict]) -> SplitParams:
    """single_area from early-play blobs (mostly single icons); split_ratio and extra_area
    by grid search for the highest mean exact-5 rate."""
    areas = [a for r in records if r["phase"] == "early" for _, a in r["areas"]]
    single = float(np.median(areas))
    grid = itertools.product(
        (1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    )
    best = max(
        (SplitParams(single, r, e) for r, e in grid),
        key=lambda p: score(records, p)["mean_exact5"],
    )
    return best


def run_count_eval(train: List[Path], test: List[Path], output: Path) -> Dict:
    """Fit on train videos, score baseline and split counts on train and test, and write
    report.json into a new output folder."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new folder: {output}")
    data = {}
    for video in [*train, *test]:
        print(f"Collecting all-alive frames: {video}")
        data[str(video)] = collect(video)
    train_records = [r for v in train for r in data[str(v)]]
    params = fit(train_records)
    report = {
        "claims": "Proxy: share of all-alive frames showing 5 per team. Not labelled accuracy.",
        "fitted_on": [str(v) for v in train],
        "params": params.__dict__,
        "results": {},
    }
    for split, videos in (("train", train), ("test", test)):
        for v in videos:
            recs = data[str(v)]
            report["results"][str(v)] = {
                "split": split,
                "baseline": score(recs),
                "split_icons": score(recs, params),
            }
    write_new_text(output / "report.json", json.dumps(report, indent=2))
    return report
