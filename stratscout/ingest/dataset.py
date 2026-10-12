"""Dataset: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .annotations import load_demo_labels
from .config import CLASSES
from .demos import match_group


def pairing_table(*, context) -> None:
    videos = sorted(
        (
            p
            for d in (context.config.data_dir, context.config.radar_dir)
            for p in Path(d).glob("*.mp4")
            if not p.name.startswith(("~", "."))
        )
    )
    vid_groups: Dict[str, List[str]] = {}
    for v in videos:
        vid_groups.setdefault(match_group(v.stem), []).append(f"{Path(v.parent.name) / v.name}")
    print(f"{'demo':<40} {'match_group':<24} rendered video")
    for d in sorted({r["demo"] for r in context.state["DEM_ROUNDS"]}):
        g = match_group(d)
        print(f"{d:<40} {g:<24} {', '.join(vid_groups.get(g, [])) or '-'}")
    orphans = [
        g for g in vid_groups if g not in {r["match_group"] for r in context.state["DEM_ROUNDS"]}
    ]
    if orphans:
        print("videos with no demo:", orphans)


def build_dem_dataset(out_npz: str = None, *, context) -> int:
    rows = []
    for r in context.state["DEM_ROUNDS"]:
        lab = load_demo_labels(r["demo"], context=context).get(r["round_idx"])
        if lab in CLASSES:
            rows.append((r, lab))
    if not rows:
        print("no labelled rounds yet — use DemoRoundLabeller")
        return 0
    rs, labs = zip(*rows)
    np.savez_compressed(
        out_npz or context.config.dem_npz,
        positions=np.stack([r["positions"] for r in rs]),
        world=np.stack([r["world"] for r in rs]),
        labels=np.array(labs),
        winners=np.array([r["winner"] for r in rs]),
        match_ids=np.array([r["demo"] for r in rs]),
        match_groups=np.array([r["match_group"] for r in rs]),
        round_idx=np.array([r["round_idx"] for r in rs]),
        plant_site=np.array([r["plant_site"] or "" for r in rs]),
        duration_s=np.array([r["duration_s"] for r in rs], np.float32),
    )
    print(
        f"{len(rs)} rounds from {len({r['demo'] for r in rs})} demo(s) -> {out_npz or context.config.dem_npz}"
    )
    print("per class:", {c: labs.count(c) for c in CLASSES})
    return len(rs)
