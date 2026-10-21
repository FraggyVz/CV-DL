"""Count evaluation: label-free checks of how many players each detector version finds.

During freeze time and the first seconds after it nobody has died, so a correct detector
shows 5 T and 5 CT. Across a whole match, a team count above five is always wrong. A
hidden bomb carrier counts for the T side when the badge shows a carrier at its rim.
Parameters are fitted on training matches and scored on held-out matches. These are
proxies, not labelled precision/recall: right totals can hide a miss plus a false
detection, and late-round over-counting below five is not measured.
All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .bomb import SPECTATOR_CT_HUE, SPECTATOR_T_HUE, carrier_rim, carrier_visible, find_bomb
from .config import Config
from .context import RunContext
from .icon_split import SplitParams, allocate, team_blobs
from .rounds import cached_rounds, round_at, video_background
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


def _contexts():
    return RunContext(Config(icon_area=(120, 450), hue_tol=8.0)), RunContext()


def _background(cap: cv2.VideoCapture) -> np.ndarray:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cv2.cvtColor(video_background(cap, total), cv2.COLOR_BGR2RGB)


def frame_record(rgb: np.ndarray, background: np.ndarray, *, contexts) -> Dict:
    """Everything needed to count one frame under any version: same-team blobs (area,
    attached white, disc fits), whether the bomb shows a carrier, and baseline counts."""
    baseline_ctx, ctx = contexts
    bomb = find_bomb(rgb, context=ctx)
    carried = bomb is not None and carrier_visible(
        carrier_rim(rgb, bomb, background=background, context=ctx)
    )
    base = find_icons_teams(rgb, HUES, background, context=baseline_ctx)
    blobs = [
        {k: b[k] for k in ("team", "area", "white", "fits")}
        for b in team_blobs(rgb, HUES, background, context=ctx)
    ]
    return {
        "carried": bool(carried),
        "blobs": blobs,
        "baseline": [sum(p[2] == t for p in base) for t in (0, 1)],
    }


def collect(video: Path, early_seconds: float = 8.0) -> List[Dict]:
    """Frame records inside the all-alive windows of one prepared video."""
    contexts = _contexts()
    windows = all_alive_windows(cached_rounds(video), early_seconds)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    background = _background(cap)
    records = []
    try:
        for phase, a, b in windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a * fps)))
            for i in range(int(round(a * fps)), int(round(b * fps))):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                records.append(
                    {"frame": i, "phase": phase, **frame_record(rgb, background, contexts=contexts)}
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
        cap = params.max_per_team
        t_cap = None if cap is None else cap - int(record["carried"])
        t = sum(allocate([b for b in record["blobs"] if b["team"] == 0], params, t_cap))
        ct = sum(allocate([b for b in record["blobs"] if b["team"] == 1], params))
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


def single_area(records: Sequence[Dict]) -> float:
    """Median early-play blob area: most of those blobs are single icons."""
    areas = [b["area"] for r in records if r["phase"] == "early" for b in r["blobs"]]
    return float(np.median(areas))


def fit_area(records: Sequence[Dict], max_per_team: Optional[int]) -> SplitParams:
    """The area rule: split_ratio and extra_area by grid search for the highest mean
    exact-5 rate, with or without the cap. Death markers are not removed."""
    single = single_area(records)
    grid = itertools.product(
        (1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    )
    candidates = (
        SplitParams(single, r, e, max_per_team=max_per_team, method="area", min_arrow_px=0)
        for r, e in grid
    )
    return max(candidates, key=lambda p: score(records, p)["mean_exact5"])


def versions(train_records: Sequence[Dict]) -> Dict[str, SplitParams]:
    """The compared detectors, all with single_area from the training matches. Only the
    area rules are grid-searched; the disc fit has no tuned threshold."""
    return {
        "area_split": fit_area(train_records, None),
        "area_split_capped": fit_area(train_records, 5),
        "discs_markers_capped": SplitParams(single_area(train_records)),
    }


def match_counts(video: Path, params: Dict[str, SplitParams], step: int = 2) -> List[Dict]:
    """[T, CT] per sampled frame of a whole match for the baseline and each version, with
    the round and the seconds since that round's freeze ended."""
    rounds = cached_rounds(video)
    contexts = _contexts()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    background = _background(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows, i = [], 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rec = frame_record(rgb, background, contexts=contexts)
                    r = round_at(rounds, i / fps)
                    row = {"frame": i, "seconds": round(i / fps, 2), "round": r["round"]}
                    row["live_seconds"] = round(i / fps - r["start"] - r["freeze"], 2)
                    row["baseline"] = counts(rec)
                    row.update({name: counts(rec, p) for name, p in params.items()})
                    rows.append(row)
            i += 1
    finally:
        cap.release()
    return rows


def impossible_rate(rows: Sequence[Dict], name: str) -> float:
    """Share of frames where a team count exceeds five (always wrong in 5v5)."""
    return round(float(np.mean([max(r[name]) > 5 for r in rows])), 4)


def run_count_eval(train: List[Path], test: List[Path], output: Path) -> Dict:
    """Fit on train videos; score every version on train and test all-alive frames, count
    impossible totals over the whole test matches, and write report.json and
    test-match-counts.json into a new output folder."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new folder: {output}")
    data = {}
    for video in [*train, *test]:
        print(f"Collecting all-alive frames: {video}")
        data[str(video)] = collect(video)
    variants = versions([r for v in train for r in data[str(v)]])
    report = {
        "claims": "Proxies: share of all-alive frames showing 5 per team, and share of all "
        "frames with an impossible (>5) team count. Not labelled accuracy.",
        "fitted_on": [str(v) for v in train],
        "params": {name: p.__dict__ for name, p in variants.items()},
        "results": {},
    }
    for split, videos in (("train", train), ("test", test)):
        for v in videos:
            recs = data[str(v)]
            report["results"][str(v)] = {
                "split": split,
                "baseline": score(recs),
                **{name: score(recs, p) for name, p in variants.items()},
            }
    full = {}
    for v in test:
        print(f"Counting every other frame of the whole match: {v}")
        rows = match_counts(v, variants)
        full[str(v)] = rows
        report["results"][str(v)]["impossible_rate"] = {
            name: impossible_rate(rows, name) for name in ["baseline", *variants]
        }
    write_new_text(output / "report.json", json.dumps(report, indent=2))
    write_new_text(output / "test-match-counts.json", json.dumps(full))
    return report
