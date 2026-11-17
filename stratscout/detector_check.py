"""Detector check: score the colour detector's player positions against demo positions.

Demo positions are evaluation evidence only. They are never passed to the detector or to
any model. Held-out matches are refused before any of their files are opened. The time
alignment and the demo-to-radar mapping are fitted on odd rounds; the detector is scored
on even rounds, so calibration error and detector error are reported separately. Player
identity over time (the tracker) is not evaluated. Importing this module reads nothing.
"""

from __future__ import annotations

import csv
import io
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .storage import write_new_text

# Test matches. Use the full Vitality-Falcons names: plain "vitality" would also block the
# Aurora-Vitality training match. The demo file is called vitality-vs-falcons-....
HELD_OUT = ("navi", "natus", "vitality-falcons", "vitality-vs-falcons")
TEAMS = ("T", "CT")  # detector team 0 = T (yellow), 1 = CT (cyan)
DEMO_TEAM = {2: 0, 3: 1}  # demo team_num -> detector team
TICKRATE = 64
SAMPLE_TICKS = 4  # demo positions every 4 ticks (16 per second), interpolated between


@dataclass(frozen=True)
class PassMark:
    """Agreed thresholds in native radar pixels. A detection within match_px finds a player;
    errors use the wider error_gate_px so their 90th percentile is not cut at match_px."""

    median_px: float = 6.0
    p90_px: float = 12.0
    min_found: float = 0.90
    max_extra: float = 0.10
    match_px: float = 12.0
    error_gate_px: float = 36.0


@dataclass(frozen=True)
class CheckConfig:
    every_s: float = 1.0
    fps: int = 10
    spawn_s: float = 5.0  # first seconds after freeze end count as the spawn stack
    overlap_px: float = 18.0  # one icon diameter
    still_speed: float = 20.0  # world units per second
    moving_speed: float = 150.0
    offset_range_s: float = 3.0
    offset_step_s: float = 0.05
    pair_px: float = 20.0  # calibration pairs used to fit the mapping
    band: int = 4  # carrier rim band, as in bomb-preview


def refuse_held_out(*names) -> None:
    for name in names:
        if any(h in str(name).lower() for h in HELD_OUT):
            raise ValueError(f"{name} is a held-out test match; the detector check never reads it")


def demo_rounds(freeze_ends, round_ends) -> List[tuple]:
    """Completed rounds as (freeze-end tick, end tick). Keeps the last end event per round,
    drops warm-up ends (no reason) and starts each round at its last freeze end."""
    ends = round_ends.sort_values("tick").drop_duplicates("total_rounds_played", keep="last")
    ends = ends[ends["reason"].notna()]
    freeze = np.sort(freeze_ends["tick"].to_numpy(np.int64))
    rounds, previous = [], -1
    for end in ends["tick"].to_numpy(np.int64):
        before = freeze[(freeze < end) & (freeze > previous)]
        if len(before):
            rounds.append((int(before[-1]), int(end)))
        previous = int(end)
    return rounds


def fit_time_map(demo_s, video_s):
    """Least squares video = a + b * demo over round anchors -> (a, b, residuals in s)."""
    demo_s, video_s = np.asarray(demo_s, float), np.asarray(video_s, float)
    if len(demo_s) < 2:
        raise ValueError("need at least two round anchors to align demo and video time")
    b, a = np.polyfit(demo_s, video_s, 1)
    return float(a), float(b), video_s - (a + b * demo_s)


def overview_uv(world: np.ndarray) -> np.ndarray:
    """World X, Y -> the de_mirage overview's normalised x right, y down (no crop fit)."""
    from types import SimpleNamespace

    from .ingest.config import DemCfg
    from .ingest.coordinates import world_to_radar

    identity = {"s": 1.0, "ox": 0.0, "oy": 0.0}
    return world_to_radar(world, identity, context=SimpleNamespace(config=DemCfg()))


@dataclass(frozen=True)
class RadarMap:
    """Overview coordinates -> radar crop pixels: x = sx * u + tx, y = sy * v + ty."""

    sx: float
    sy: float
    tx: float
    ty: float

    def apply(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        return np.stack([self.sx * uv[..., 0] + self.tx, self.sy * uv[..., 1] + self.ty], -1)


def fit_radar_map(uv, px) -> RadarMap:
    uv, px = np.asarray(uv, float), np.asarray(px, float)
    if len(uv) < 3:
        raise ValueError("too few calibration pairs to fit the radar mapping")
    sx, tx = np.polyfit(uv[:, 0], px[:, 0], 1)
    sy, ty = np.polyfit(uv[:, 1], px[:, 1], 1)
    return RadarMap(float(sx), float(sy), float(tx), float(ty))


def initial_radar_map(frames: Sequence[tuple]) -> RadarMap:
    """(truth uv, detected px) of frames whose counts agree -> spread and centre ratios.
    Needs no pairing, so it starts the iterative fit without guessing which icon is whom."""
    scales, shifts = [], []
    for uv, px in frames:
        spread = uv.std(0)
        if (spread <= 0).any():
            continue
        s = px.std(0) / spread
        scales.append(s)
        shifts.append(px.mean(0) - s * uv.mean(0))
    if not scales:
        raise ValueError("no frames with matching team counts to start the radar mapping")
    (sx, sy), (tx, ty) = np.median(scales, 0), np.median(shifts, 0)
    return RadarMap(float(sx), float(sy), float(tx), float(ty))


_PERMS: Dict[tuple, np.ndarray] = {}


def _perms(m: int, n: int) -> np.ndarray:
    if (m, n) not in _PERMS:
        _PERMS[(m, n)] = np.array(list(permutations(range(m), n)), dtype=np.int64)
    return _PERMS[(m, n)]


def assign(truth: np.ndarray, dets: np.ndarray, gate: float) -> List[tuple]:
    """Optimal one-to-one pairs (truth i, detection j, distance) minimising the summed
    min(distance, gate); pairs beyond the gate are dropped. Exact brute force, which is
    fast for the at most six icons per team on the radar."""
    n, m = len(truth), len(dets)
    if not n or not m:
        return []
    if max(n, m) > 7:
        raise ValueError("more than seven positions for one team; check the inputs")
    d = np.linalg.norm(truth[:, None, :] - dets[None, :, :], axis=-1)
    capped = np.minimum(d, gate)
    if n <= m:
        perms = _perms(m, n)
        best = perms[capped[np.arange(n), perms].sum(1).argmin()]
        pairs = list(enumerate(best.tolist()))
    else:
        perms = _perms(n, m)
        best = perms[capped[perms, np.arange(m)].sum(1).argmin()]
        pairs = [(i, j) for j, i in enumerate(best.tolist())]
    return [(i, j, float(d[i, j])) for i, j in pairs if d[i, j] <= gate]


class DemoPositions:
    """Alive player positions sampled from a demo and interpolated between samples.

    ticks (T,), steamids (P,), xy (T, P, 2) world units, alive (T, P), team (T, P) with
    0 = T, 1 = CT and -1 = not playing. A player counts as alive only when alive at both
    neighbouring samples, so the moment of death is left out."""

    def __init__(self, ticks, steamids, xy, alive, team):
        self.ticks = np.asarray(ticks, np.int64)
        self.steamids = [str(s) for s in steamids]
        self.xy = np.asarray(xy, float)
        self.uv = overview_uv(self.xy).astype(float)
        self.alive = np.asarray(alive, bool)
        self.team = np.asarray(team, np.int64)

    @classmethod
    def from_frame(cls, df) -> DemoPositions:
        df = df.copy()
        df["steamid"] = df["steamid"].astype(str)
        ticks = np.sort(df["tick"].unique())
        steamids = sorted(df["steamid"].unique())
        index = {s: k for k, s in enumerate(steamids)}
        rows = np.searchsorted(ticks, df["tick"].to_numpy())
        cols = df["steamid"].map(index).to_numpy()
        xy = np.full((len(ticks), len(steamids), 2), np.nan)
        alive = np.zeros((len(ticks), len(steamids)), bool)
        team = np.full((len(ticks), len(steamids)), -1, np.int64)
        xy[rows, cols] = df[["X", "Y"]].to_numpy(float)
        alive[rows, cols] = df["is_alive"].to_numpy(bool)
        team[rows, cols] = [DEMO_TEAM.get(int(t), -1) for t in df["team_num"]]
        return cls(ticks, steamids, xy, alive, team)

    def at(self, seconds: float) -> Optional[Dict]:
        """Alive players at demo time seconds: steamid, team, world xy, overview uv and
        speed (world units per second)."""
        tick = seconds * TICKRATE
        hi = int(np.searchsorted(self.ticks, tick))
        if hi == 0 or hi >= len(self.ticks):
            return None
        lo = hi - 1
        span = self.ticks[hi] - self.ticks[lo]
        if span > 4 * SAMPLE_TICKS:
            return None  # outside the sampled rounds
        w = (tick - self.ticks[lo]) / span
        keep = (
            self.alive[lo]
            & self.alive[hi]
            & (self.team[lo] >= 0)
            & (self.team[lo] == self.team[hi])
            & np.isfinite(self.xy[lo]).all(-1)
            & np.isfinite(self.xy[hi]).all(-1)
        )
        idx = np.nonzero(keep)[0]
        xy = (1 - w) * self.xy[lo, idx] + w * self.xy[hi, idx]
        uv = (1 - w) * self.uv[lo, idx] + w * self.uv[hi, idx]
        speed = np.linalg.norm(self.xy[hi, idx] - self.xy[lo, idx], axis=-1) / (span / TICKRATE)
        return {
            "steamid": [self.steamids[k] for k in idx],
            "team": self.team[lo, idx],
            "xy": xy,
            "uv": uv,
            "speed": speed,
        }


def carrier_timeline(pickups, drops, plants) -> List[tuple]:
    """(tick, steamid or None) events: who carries the bomb from that tick on."""
    events = [(int(t), str(s)) for t, s in zip(pickups["tick"], pickups["user_steamid"])]
    events += [(int(t), None) for t in drops["tick"]] + [(int(t), None) for t in plants["tick"]]
    return sorted(events, key=lambda e: (e[0], e[1] is not None))


def carrier_at(timeline: List[tuple], tick: float) -> Optional[str]:
    carrier = None
    for t, steamid in timeline:
        if t > tick:
            break
        carrier = steamid
    return carrier


def truth_frame(frame: Dict, positions: DemoPositions, time_map, offset: float, radar):
    """Truth for one frame: demo time, per-team pixel positions and the alive players."""
    a, b = time_map
    demo_s = (frame["video_s"] - a) / b + offset
    players = positions.at(demo_s)
    if players is None:
        return demo_s, None
    players["px"] = radar.apply(players["uv"])
    return demo_s, players


def _team_arrays(players, frame, team: int):
    if players is None or not len(players["team"]):
        truth = np.zeros((0, 2))
    else:
        truth = players["px"][players["team"] == team]
    dets = frame["dets"][team]
    return truth, dets


def collect_pairs(frames, positions, time_map, offset, radar, gate, equal_counts=True):
    """Calibration pairs (uv, px, distance) from frames, optionally only where the team's
    detected count equals its alive count (fewer wrong pairings)."""
    uv, px, dist = [], [], []
    for frame in frames:
        _, players = truth_frame(frame, positions, time_map, offset, radar)
        if players is None:
            continue
        for team in (0, 1):
            truth, dets = _team_arrays(players, frame, team)
            if equal_counts and len(truth) != len(dets):
                continue
            team_uv = players["uv"][players["team"] == team]
            for i, j, d in assign(truth, dets, gate):
                uv.append(team_uv[i])
                px.append(dets[j])
                dist.append(d)
    return np.array(uv).reshape(-1, 2), np.array(px).reshape(-1, 2), np.array(dist)


def search_offset(frames, positions, time_map, radar, config: CheckConfig, gate: float):
    """Offset (s) minimising the median paired distance; returns it and the whole curve."""
    steps = np.arange(-config.offset_range_s, config.offset_range_s + 1e-9, config.offset_step_s)
    curve = []
    for offset in steps:
        _, _, dist = collect_pairs(frames, positions, time_map, offset, radar, gate)
        curve.append(float(np.median(dist)) if len(dist) else float("inf"))
    best = int(np.argmin(curve))
    return float(steps[best]), [(float(s), c) for s, c in zip(steps, curve)]


def calibrate(frames, positions, time_map, config: CheckConfig, mark: PassMark) -> Dict:
    """Fit the radar mapping and a time offset on calibration frames (odd rounds only).

    Freeze-time frames (players still in spawn) start the mapping without depending on
    time; live frames then refine the offset and mapping in turn. The start needs frames
    where all ten players are detected. Stacked spawns can hide one icon in every freeze
    frame (MOUZ-BetBoom); then live frames start the mapping instead, at offset zero."""
    freeze = [f for f in frames if f["phase"] == "freeze"]
    live = [f for f in frames if f["phase"] == "live"]

    def full_frames(candidates):
        starts = []
        for frame in candidates:
            a, b = time_map
            players = positions.at((frame["video_s"] - a) / b)
            if players is None or len(players["team"]) != 10:
                continue
            dets = frame["dets"]
            if len(dets[0]) == 5 and len(dets[1]) == 5:
                starts.append((players["uv"], np.concatenate(dets)))
        return starts

    start, seed = "freeze", freeze
    starts = full_frames(freeze)
    if not starts:
        start, seed = "live", live
        starts = full_frames(live)
    radar = initial_radar_map(starts)
    for gate in (mark.error_gate_px, config.pair_px, config.pair_px):
        uv, px, _ = collect_pairs(seed, positions, time_map, 0.0, radar, gate)
        radar = fit_radar_map(uv, px)
    offset, curve = 0.0, []
    # A live start fits the map at offset zero, so part of the true offset leaks into it;
    # keep alternating until the offset settles. The freeze start keeps its two rounds.
    for step in range(2 if start == "freeze" else 10):
        previous = offset
        offset, curve = search_offset(live, positions, time_map, radar, config, mark.error_gate_px)
        uv, px, _ = collect_pairs(live + freeze, positions, time_map, offset, radar, config.pair_px)
        radar = fit_radar_map(uv, px)
        if start == "live" and step and abs(offset - previous) < config.offset_step_s / 2:
            break
    _, _, dist = collect_pairs(live, positions, time_map, offset, radar, mark.error_gate_px)
    return {
        "radar": radar,
        "offset_s": offset,
        "offset_curve": curve,
        "pairs": len(dist),
        "mapping_start": start,
    }


def score_frames(frames, positions, time_map, offset, radar, rounds, timeline, config, mark):
    """One row per alive true player and one per detection, for live frames of scored rounds."""
    players_out, dets_out = [], []
    for frame in frames:
        demo_s, players = truth_frame(frame, positions, time_map, offset, radar)
        start_tick, end_tick = rounds[frame["round"] - 1]
        tick = demo_s * TICKRATE
        if players is None or not start_tick <= tick <= end_tick:
            continue
        since_start = (tick - start_tick) / TICKRATE
        carrier = carrier_at(timeline, tick)
        all_px = players["px"]
        for team in (0, 1):
            truth, dets = _team_arrays(players, frame, team)
            in_team = np.nonzero(players["team"] == team)[0]
            pairs = {i: (j, d) for i, j, d in assign(truth, dets, mark.error_gate_px)}
            matched = {j: d for i, (j, d) in pairs.items()}
            for i, k in enumerate(in_team):
                others = np.delete(all_px, k, axis=0)
                same = np.delete(truth, i, axis=0)
                nearest_any = (
                    float(np.linalg.norm(others - all_px[k], axis=-1).min())
                    if len(others)
                    else np.inf
                )
                nearest_same = (
                    float(np.linalg.norm(same - truth[i], axis=-1).min()) if len(same) else np.inf
                )
                distance = pairs[i][1] if i in pairs else float("nan")
                players_out.append(
                    {
                        "match": frame["match"],
                        "round": frame["round"],
                        "video_s": round(frame["video_s"], 3),
                        "since_start_s": round(since_start, 2),
                        "team": TEAMS[team],
                        "steamid": players["steamid"][k],
                        "x": round(float(truth[i, 0]), 2),
                        "y": round(float(truth[i, 1]), 2),
                        "speed": round(float(players["speed"][k]), 1),
                        "nearest_same_px": round(nearest_same, 2),
                        "nearest_any_px": round(nearest_any, 2),
                        "carrier": players["steamid"][k] == carrier,
                        "distance_px": distance,
                        "found": bool(distance <= mark.match_px),
                    }
                )
            for j, (x, y) in enumerate(dets):
                distance = matched.get(j, float("nan"))
                dets_out.append(
                    {
                        "match": frame["match"],
                        "round": frame["round"],
                        "video_s": round(frame["video_s"], 3),
                        "since_start_s": round(since_start, 2),
                        "team": TEAMS[team],
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "inferred_carrier": bool(frame["inferred"][team][j]),
                        "distance_px": distance,
                        "extra": not distance <= mark.match_px,
                    }
                )
    return players_out, dets_out


def team_scores(players, detections, mark: PassMark) -> Dict:
    """Found rate, extra rate and position error (median, 90th percentile) for one group."""
    distances = np.array([p["distance_px"] for p in players], float)
    errors = distances[np.isfinite(distances)]
    found = int(sum(p["found"] for p in players))
    extra = int(sum(d["extra"] for d in detections))
    out = {
        "players": len(players),
        "found": found,
        "missed": len(players) - found,
        "found_rate": found / len(players) if players else float("nan"),
        "detections": len(detections),
        "extra": extra,
        "extra_rate": extra / len(detections) if detections else float("nan"),
        "median_px": float(np.median(errors)) if len(errors) else float("nan"),
        "p90_px": float(np.percentile(errors, 90)) if len(errors) else float("nan"),
    }
    out["passes"] = {
        "median": out["median_px"] <= mark.median_px,
        "p90": out["p90_px"] <= mark.p90_px,
        "found": out["found_rate"] >= mark.min_found,
        "extra": out["extra_rate"] <= mark.max_extra,
    }
    out["pass"] = all(out["passes"].values())
    return out


def strata(players, detections, config: CheckConfig) -> Dict[str, tuple]:
    """Named subsets: crowded moments (spawn stacks, overlaps), the carrier and speed."""

    def spawn(row):
        return row["since_start_s"] < config.spawn_s

    def overlap(row):
        return row["nearest_any_px"] < config.overlap_px

    return {
        "all": (players, detections),
        "spawn stack": ([p for p in players if spawn(p)], [d for d in detections if spawn(d)]),
        "after spawn": (
            [p for p in players if not spawn(p)],
            [d for d in detections if not spawn(d)],
        ),
        "overlapping": ([p for p in players if overlap(p) and not spawn(p)], []),
        "isolated": ([p for p in players if not overlap(p) and not spawn(p)], []),
        "bomb carrier": ([p for p in players if p["carrier"]], []),
        "still": ([p for p in players if p["speed"] < config.still_speed], []),
        "moving": ([p for p in players if p["speed"] > config.moving_speed], []),
    }


def calibration_flags(summary: Dict, config: CheckConfig, mark: PassMark) -> List[str]:
    """Reasons the calibration is too uncertain to judge the detector."""
    flags = []
    if abs(summary["offset_s"]) >= config.offset_range_s - config.offset_step_s:
        flags.append("time offset search reached its limit")
    if summary["anchor_max_residual_s"] > 1.5:
        flags.append("round anchors disagree by more than 1.5 s (pauses or round mismatch?)")
    if abs(summary["time_scale"] - 1) > 0.01:
        flags.append("demo and video run at different speeds (more than 1%)")
    floor = summary["floor_median_px"]
    if not np.isfinite(floor) or floor > mark.median_px / 2:
        flags.append(
            "still, isolated calibration players are off by more than half the pass median"
        )
    if summary["pairs"] < 200:
        flags.append("fewer than 200 calibration pairs")
    drift = [abs(v) for v in summary["scored_round_offset_spread_s"].values()]
    if drift and max(drift) > 0.5:
        flags.append("best time offset differs by more than 0.5 s between rounds")
    return flags


def find_demo(demos_dir: Path, fragment: str) -> Path:
    """The single demo whose file name starts with the match fragment (names only)."""
    hits = [
        p
        for p in sorted(Path(demos_dir).glob("*.dem"))
        if p.stem.lower().startswith(fragment.lower())
    ]
    if len(hits) != 1:
        raise ValueError(
            f"expected one demo starting with {fragment!r} in {demos_dir}, found {len(hits)}"
        )
    return hits[0]


def sample_times(video_rounds, rounds, time_map, every_s: float) -> List[Dict]:
    """Freeze and live frame times for every complete round (partial rounds are skipped)."""
    a, b = time_map
    out = []
    for number, (r, (_, end_tick)) in enumerate(zip(video_rounds, rounds), start=1):
        if r["partial"]:
            continue
        play = r["start"] + r["freeze"]
        end = min(r["end"], a + b * end_tick / TICKRATE)
        for phase, t0, t1 in (("freeze", r["start"] + every_s, play), ("live", play, end)):
            for t in np.arange(t0, t1, every_s):
                out.append({"round": number, "phase": phase, "video_s": float(t)})
    return out


def detect_frames(video: Path, times: List[Dict], match: str, band: int) -> List[Dict]:
    """Run bomb-preview's detector on the frame nearest each time, decoding front to back."""
    import cv2

    from .bomb import SPECTATOR_T_HUE
    from .live_preview import detect, spectator_context
    from .rounds import video_background

    context = spectator_context()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        background = cv2.cvtColor(video_background(cap, total), cv2.COLOR_BGR2RGB)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        wanted: Dict[int, List[Dict]] = {}
        for item in times:
            index = int(round(item["video_s"] * fps))
            if index < total:
                wanted.setdefault(index, []).append(item)
        frames, last = [], max(wanted, default=-1)
        for index in range(last + 1):
            if index not in wanted:
                if not cap.grab():
                    break
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            found = detect(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                band,
                SPECTATOR_T_HUE,
                background,
                context=context,
            )
            dets, inferred = [[], []], [[], []]
            for x, y, team in found["players"]:
                dets[TEAMS.index(team)].append((x, y))
                inferred[TEAMS.index(team)].append(False)
            if found["carrier"] is not None:
                dets[0].append(found["carrier"])
                inferred[0].append(True)
            for item in wanted[index]:
                frames.append(
                    {
                        **item,
                        "match": match,
                        "video_s": index / fps,
                        "dets": [np.array(d, float).reshape(-1, 2) for d in dets],
                        "inferred": inferred,
                    }
                )
    finally:
        cap.release()
    return frames


def load_demo(path: Path):
    from demoparser2 import DemoParser

    parser = DemoParser(str(path))
    if parser.parse_header().get("map_name") != "de_mirage":
        raise ValueError(f"{path.name} is not a de_mirage demo")
    rounds = demo_rounds(
        parser.parse_event("round_freeze_end"),
        parser.parse_event("round_end", other=["total_rounds_played"]),
    )
    return parser, rounds


def check_match(folder: Path, demo: Path, config: CheckConfig, mark: PassMark) -> Dict:
    from .live_preview import match_video
    from .rounds import cached_rounds

    print(f"{folder.name}: estimating rounds and reading the demo", flush=True)
    video_rounds = cached_rounds(match_video(folder, 5))
    parser, rounds = load_demo(demo)
    if len(rounds) != len(video_rounds):
        raise ValueError(
            f"{folder.name}: video has {len(video_rounds)} estimated rounds but the demo has "
            f"{len(rounds)}; cannot pair rounds"
        )
    complete = [k for k, r in enumerate(video_rounds) if not r["partial"]]
    a, b, residuals = fit_time_map(
        [rounds[k][0] / TICKRATE for k in complete],
        [video_rounds[k]["start"] + video_rounds[k]["freeze"] for k in complete],
    )
    times = sample_times(video_rounds, rounds, (a, b), config.every_s)
    ticks = set()
    for k in complete:
        first = int(((video_rounds[k]["start"] - a) / b - config.offset_range_s - 2) * TICKRATE)
        last = rounds[k][1] + int((config.offset_range_s + 2) * TICKRATE)
        ticks.update(range(max(first, 0) // SAMPLE_TICKS * SAMPLE_TICKS, last, SAMPLE_TICKS))
    df = parser.parse_ticks(["X", "Y", "team_num", "is_alive"], ticks=sorted(ticks))
    positions = DemoPositions.from_frame(df)
    timeline = carrier_timeline(
        parser.parse_event("bomb_pickup"),
        parser.parse_event("bomb_dropped"),
        parser.parse_event("bomb_planted"),
    )
    video = match_video(folder, config.fps)
    print(f"{folder.name}: running the detector on {len(times)} frames of {video.name}", flush=True)
    frames = detect_frames(video, times, folder.name, config.band)
    odd = [f for f in frames if f["round"] % 2 == 1]
    even = [f for f in frames if f["round"] % 2 == 0 and f["phase"] == "live"]
    fit = calibrate(odd, positions, (a, b), config, mark)
    radar, offset = fit["radar"], fit["offset_s"]
    calib_players, _ = score_frames(
        [f for f in odd if f["phase"] == "live"],
        positions,
        (a, b),
        offset,
        radar,
        rounds,
        timeline,
        config,
        mark,
    )
    floor = [
        p["distance_px"]
        for p in calib_players
        if p["speed"] < config.still_speed
        and p["nearest_any_px"] >= config.overlap_px
        and p["since_start_s"] >= config.spawn_s
    ]
    floor = np.array(floor, float)
    floor = floor[np.isfinite(floor)]
    calib_errors = np.array([p["distance_px"] for p in calib_players], float)
    calib_errors = calib_errors[np.isfinite(calib_errors)]
    # Diagnostic only: the best offset of each scored round; scores keep the single offset.
    round_offsets = {}
    for number in sorted({f["round"] for f in even}):
        subset = [f for f in even if f["round"] == number]
        steps = np.arange(offset - 1.0, offset + 1.0 + 1e-9, 0.1)
        medians = []
        for step in steps:
            _, _, dist = collect_pairs(subset, positions, (a, b), step, radar, mark.error_gate_px)
            medians.append(float(np.median(dist)) if len(dist) else float("inf"))
        round_offsets[number] = round(float(steps[int(np.argmin(medians))]) - offset, 2)
    calibration = {
        "match": folder.name,
        "demo": demo.name,
        "time_offset_s": a,
        "time_scale": b,
        "anchor_residuals_s": [round(float(r), 3) for r in residuals],
        "anchor_max_residual_s": float(np.abs(residuals).max()),
        "offset_s": offset,
        "offset_curve": fit["offset_curve"],
        "radar_map": asdict(radar),
        "pairs": fit["pairs"],
        "mapping_start": fit["mapping_start"],
        "calibration_median_px": float(np.median(calib_errors))
        if len(calib_errors)
        else float("nan"),
        "calibration_p90_px": float(np.percentile(calib_errors, 90))
        if len(calib_errors)
        else float("nan"),
        "floor_players": int(len(floor)),
        "floor_median_px": float(np.median(floor)) if len(floor) else float("nan"),
        "floor_p90_px": float(np.percentile(floor, 90)) if len(floor) else float("nan"),
        "scored_round_offset_spread_s": round_offsets,
        "frames": {"calibration": len(odd), "scored": len(even)},
    }
    calibration["flags"] = calibration_flags(calibration, config, mark)
    players, detections = score_frames(
        even, positions, (a, b), offset, radar, rounds, timeline, config, mark
    )
    return {"calibration": calibration, "players": players, "detections": detections}


def _csv(rows: List[Dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _json(value) -> str:
    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    return json.dumps(clean(value), indent=2) + "\n"


def summarise(results: List[Dict], config: CheckConfig, mark: PassMark) -> Dict:
    players = [p for r in results for p in r["players"]]
    detections = [d for r in results for d in r["detections"]]
    groups = {"pooled": (players, detections)}
    for r in results:
        groups[r["calibration"]["match"]] = (r["players"], r["detections"])
    summary = {"groups": {}}
    for name, (ps, ds) in groups.items():
        summary["groups"][name] = {}
        for team in TEAMS:
            team_p = [p for p in ps if p["team"] == team]
            team_d = [d for d in ds if d["team"] == team]
            summary["groups"][name][team] = {
                stratum: team_scores(sp, sd, mark)
                for stratum, (sp, sd) in strata(team_p, team_d, config).items()
            }
    flags = [
        f"{r['calibration']['match']}: {f}" for r in results for f in r["calibration"]["flags"]
    ]
    passed = all(summary["groups"]["pooled"][team]["all"]["pass"] for team in TEAMS)
    summary["calibration_flags"] = flags
    summary["verdict"] = "INCONCLUSIVE" if flags else ("PASS" if passed else "FAIL")
    summary["inferred_carriers"] = int(sum(d["inferred_carrier"] for d in detections))
    return summary


def _fmt(value, pattern="{:.1f}") -> str:
    return "n/a" if value is None or not np.isfinite(value) else pattern.format(value)


def report(results: List[Dict], summary: Dict, mark: PassMark) -> str:
    pooled = summary["groups"]["pooled"]
    lines = [
        "# Detector check: colour detector vs demo positions",
        "",
        f"**Verdict: {summary['verdict']}** (Furia/Spirit even rounds; NaVi not read).",
        "",
        f"Pass mark per team: median <= {mark.median_px:g} px, 90th percentile <= {mark.p90_px:g} px,"
        f" found >= {mark.min_found:.0%}, extra <= {mark.max_extra:.0%}. A player is found by a"
        f" detection within {mark.match_px:g} px; errors use pairs within {mark.error_gate_px:g} px.",
        "",
        "## Result per team (both matches, scored rounds)",
        "",
        "| Team | Players | Found | Extra | Median px | 90th pct px | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for team in TEAMS:
        s = pooled[team]["all"]
        lines.append(
            f"| {team} | {s['players']} | {_fmt(s['found_rate'], '{:.1%}')} | "
            f"{_fmt(s['extra_rate'], '{:.1%}')} | {_fmt(s['median_px'])} | {_fmt(s['p90_px'])} | "
            f"{'yes' if s['pass'] else 'no'} |"
        )
    lines += [
        "",
        "## Crowded and difficult moments (not pass/fail)",
        "",
        "| Team | Moment | Players | Found | Extra | Median px | 90th pct px |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for team in TEAMS:
        for stratum, s in pooled[team].items():
            if stratum == "all" or not s["players"]:
                continue
            lines.append(
                f"| {team} | {stratum} | {s['players']} | {_fmt(s['found_rate'], '{:.1%}')} | "
                f"{_fmt(s['extra_rate'], '{:.1%}')} | {_fmt(s['median_px'])} | {_fmt(s['p90_px'])} |"
            )
    lines += [
        "",
        "## Calibration error (kept separate from detector error)",
        "",
        "Fitted on odd rounds only. The floor is still, isolated players after spawn: it",
        "bounds how much of any error comes from the mapping and centroids, not the detector.",
        "",
        "| Match | Anchor max residual s | Offset s | Speed ratio | Pairs | Calibration median px | Floor median px | Floor 90th px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        c = r["calibration"]
        lines.append(
            f"| {c['match']} | {c['anchor_max_residual_s']:.2f} | {c['offset_s']:+.2f} | "
            f"{c['time_scale']:.4f} | {c['pairs']} | {_fmt(c['calibration_median_px'])} | "
            f"{_fmt(c['floor_median_px'])} | {_fmt(c['floor_p90_px'])} |"
        )
    flags = summary["calibration_flags"] or ["none"]
    lines += ["", "Calibration flags: " + "; ".join(flags) + ".", ""]
    lines += [
        "## Not covered",
        "",
        "- The tracker (player identity over time), listed as unrepaired in the README.",
        "- NaVi (held-out test match) and the CNN detector.",
        "- Freeze time and post-round frames; death-marker timing beyond counting extras.",
        f"- Inferred hidden carriers are counted as T detections ({summary['inferred_carriers']} here).",
        "- Calibration uses detector centroids on odd rounds; demo positions are evaluation evidence only.",
        "",
    ]
    return "\n".join(lines)


def run_detector_check(
    fragments: Sequence[str],
    output,
    *,
    demos_dir=Path("demos"),
    recordings_dir=Path("data/processed/recordings"),
    config: CheckConfig = CheckConfig(),
    mark: PassMark = PassMark(),
) -> Dict:
    """Score the colour detector on the given training matches and save to a new folder."""
    from .live_preview import choose_match, list_matches

    refuse_held_out(*fragments)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    pairs = []
    for fragment in fragments:
        folder = choose_match(list_matches(Path(recordings_dir)), fragment)
        demo = find_demo(Path(demos_dir), fragment)
        refuse_held_out(folder.name, demo.name)
        pairs.append((folder, demo))
    output.mkdir(parents=True)
    write_new_text(
        output / "config.json",
        _json(
            {
                "detector": "colour + icon split (live_preview.detect, spectator settings)",
                "matches": [{"folder": f.name, "demo": d.name} for f, d in pairs],
                "check": asdict(config),
                "pass_mark": asdict(mark),
                "split": "calibrate on odd rounds, score even rounds",
                "versions": {n: version(n) for n in ("numpy", "opencv-python", "demoparser2")},
                "python": platform.python_version(),
            }
        ),
    )
    results = [check_match(folder, demo, config, mark) for folder, demo in pairs]
    for r in results:
        name = r["calibration"]["match"]
        write_new_text(output / f"calibration-{name}.json", _json(r["calibration"]))
    write_new_text(output / "players.csv", _csv([p for r in results for p in r["players"]]))
    write_new_text(output / "detections.csv", _csv([d for r in results for d in r["detections"]]))
    summary = summarise(results, config, mark)
    write_new_text(output / "summary.json", _json(summary))
    write_new_text(output / "report.md", report(results, summary, mark))
    print(f"Verdict: {summary['verdict']}. Saved {output / 'report.md'}", flush=True)
    return summary
