"""Demos: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from demoparser2 import DemoParser


def list_demos(demo_dir: str = None, *, context) -> List[Path]:
    d = Path(demo_dir or context.config.demo_dir)
    d.mkdir(parents=True, exist_ok=True)
    return sorted((p for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".dem"))


def match_group(stem: str) -> str:
    """'vitality-vs-falcons-m3-mirage' and 'vitality_vs_falcons' -> 'vitality_vs_falcons'.
    Used to pair a demo with its rendered video, which must land in the same split."""
    s = re.sub("[\\s\\-]+", "_", stem.lower())
    s = re.sub("_(m\\d+_)?(de_)?mirage$", "", s)
    return re.sub("_m\\d+$", "", s)


def _event(parser, name: str):
    import pandas as pd

    try:
        df = parser.parse_event(name)
        return df if isinstance(df, pd.DataFrame) and len(df) else pd.DataFrame(columns=["tick"])
    except Exception:
        return pd.DataFrame(columns=["tick"])


def _winner(w) -> str:
    return {"2": "T", "3": "CT"}.get(str(w), str(w))


def round_windows(parser) -> List[Dict]:
    """[{start_tick, end_tick, winner, reason}] for every completed round after match start."""
    starts = np.sort(_event(parser, "round_freeze_end")["tick"].to_numpy(dtype=np.int64))
    ends = _event(parser, "round_end").sort_values("tick")
    live = _event(parser, "round_announce_match_start")["tick"]
    if len(live):
        starts = starts[starts >= int(live.max())]
    out = []
    for k, a in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else np.iinfo(np.int64).max
        e = ends[(ends["tick"] > a) & (ends["tick"] <= nxt)]
        if len(e) == 0:
            continue
        r = e.iloc[0]
        out.append(
            {
                "start_tick": int(a),
                "end_tick": int(r["tick"]),
                "winner": _winner(r.get("winner")),
                "reason": str(r.get("reason", "")),
            }
        )
    return out


def parse_demo(path: Path, *, context) -> List[Dict]:
    """One demo -> list of round dicts, positions in WORLD units (mapped to the crop later)."""
    parser = DemoParser(str(path))
    header = parser.parse_header()
    if header.get("map_name") != context.config.map_name:
        print(f"{path.name}: map {header.get('map_name')} — skipped")
        return []
    wins = round_windows(parser)
    plants = _event(parser, "bomb_planted")
    sample = [
        np.linspace(w["start_tick"], w["end_tick"], context.config.seq_len).round().astype(int)
        for w in wins
    ]
    plant_ticks = [int(t) for t in plants["tick"]] if len(plants) else []
    ticks = sorted(set(np.concatenate(sample).tolist()) | set(plant_ticks))
    df = parser.parse_ticks(["X", "Y", "team_num", "is_alive"], ticks=ticks)
    df = df[df["team_num"].isin([2, 3])]
    by_tick = {t: g for t, g in df.groupby("tick")}
    rounds = []
    for ri, (w, st) in enumerate(zip(wins, sample)):
        first = by_tick.get(int(st[0]))
        if first is None:
            continue
        slots = []
        for team in (2, 3):
            ids = sorted(first.loc[first["team_num"] == team, "steamid"].tolist())[
                : context.config.n_per_team
            ]
            slots += ids + [None] * (context.config.n_per_team - len(ids))
        world = np.full(
            (context.config.seq_len, 2 * context.config.n_per_team, 2),
            np.nan,
            np.float32,
        )
        for j, t in enumerate(st):
            g = by_tick.get(int(t))
            if g is None:
                continue
            g = g.set_index("steamid")
            for s, sid in enumerate(slots):
                if sid is not None and sid in g.index and bool(g.at[sid, "is_alive"]):
                    world[j, s] = (g.at[sid, "X"], g.at[sid, "Y"])
        plant_tick, plant_xy = (None, None)
        for _, p in plants.iterrows():
            if w["start_tick"] <= p["tick"] <= w["end_tick"]:
                g = by_tick.get(int(p["tick"]))
                if g is not None:
                    row = g[g["steamid"].astype(str) == str(p.get("user_steamid"))]
                    if len(row):
                        plant_tick = int(p["tick"])
                        plant_xy = (float(row["X"].iloc[0]), float(row["Y"].iloc[0]))
                break
        rounds.append(
            {
                "demo": path.stem,
                "match_group": match_group(path.stem),
                "round_idx": ri + 1,
                **w,
                "duration_s": (w["end_tick"] - w["start_tick"]) / context.config.tickrate,
                "plant_tick": plant_tick,
                "plant_world": plant_xy,
                "world": world,
            }
        )
    return rounds
