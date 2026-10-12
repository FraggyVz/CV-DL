"""Workflow: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from demoparser2 import DemoParser

from .annotations import load_demo_labels
from .background import load_plot_bg, radar_background
from .config import SITE_ANCHORS
from .coordinates import fit_radar, show_fit, world_to_radar
from .dataset import build_dem_dataset, pairing_table
from .demos import list_demos, parse_demo
from .video import shrink_all
from .visualisation import show_match


def prepare_radar_videos(context):
    """Explicit workflow step migrated from original notebook cell 4."""
    shrink_all(context=context)


def prepare_background(context):
    """Explicit workflow step migrated from original notebook cell 6."""
    context.state["BG"] = radar_background(context=context)
    context.state["PLOT_BG"] = load_plot_bg(context=context)
    if context.state["PLOT_BG"] is None:
        context.state["PLOT_BG"] = context.state["BG"]
        print(f"{context.config.map_image} not found — plots use the footage median")
    print(
        "fit reference:",
        "footage median" if context.state["BG"] is not None else "none (no footage yet)",
        "| plot background:",
        context.config.map_image
        if context.state["PLOT_BG"] is not None
        and context.state["PLOT_BG"] is not context.state["BG"]
        else "footage median"
        if context.state["BG"] is not None
        else "dark panel",
    )


def parse_demos(context):
    """Explicit workflow step migrated from original notebook cell 8."""
    DEMOS = list_demos(context=context)
    print(f"{context.config.demo_dir}/ -> {len(DEMOS)} demo(s)")
    context.state["DEM_ROUNDS"]: List[Dict] = []
    if DemoParser is not None:
        for d in DEMOS:
            t = time.time()
            r = parse_demo(d, context=context)
            context.state["DEM_ROUNDS"] += r
            if r:
                tw = sum((x["winner"] == "T" for x in r))
                print(
                    f"   {d.name:<40} {len(r):>3} rounds   T {tw} - {len(r) - tw} CT   {sum((x['plant_tick'] is not None for x in r))} plants   [{time.time() - t:.0f}s]"
                )
    print(f"total: {len(context.state['DEM_ROUNDS'])} rounds")


def prepare_mapping(context):
    """Explicit workflow step migrated from original notebook cell 10."""
    fit_path = Path(context.config.radar_fit_json)
    if fit_path.exists():
        RADAR_FIT = json.loads(fit_path.read_text())
        print("loaded", fit_path, RADAR_FIT)
    elif context.state["DEM_ROUNDS"] and context.state["BG"] is not None:
        RADAR_FIT = fit_radar(context.state["DEM_ROUNDS"], context.state["BG"], context=context)
        fit_path.write_text(json.dumps(RADAR_FIT, indent=1))
    else:
        RADAR_FIT = {"s": 1.0, "ox": 0.0, "oy": 0.0}
        print("no footage to fit against — using the overview mapping alone")
    for r in context.state["DEM_ROUNDS"]:
        r["positions"] = world_to_radar(r["world"], RADAR_FIT, context=context)
        if r["plant_world"] is not None:
            pxy = world_to_radar(np.array(r["plant_world"], np.float32), RADAR_FIT, context=context)
            r["plant_xy"] = tuple((float(v) for v in pxy))
            r["plant_site"] = min(
                SITE_ANCHORS,
                key=lambda k: np.hypot(pxy[0] - SITE_ANCHORS[k][0], pxy[1] - SITE_ANCHORS[k][1]),
            )
        else:
            r["plant_xy"], r["plant_site"] = (None, None)
    if context.state["DEM_ROUNDS"]:
        show_fit(
            context.state["DEM_ROUNDS"],
            context.state["PLOT_BG"],
            RADAR_FIT,
            context=context,
        )


def preview_match(context):
    """Explicit workflow step migrated from original notebook cell 12."""
    if context.state["DEM_ROUNDS"]:
        show_match(context.state["DEM_ROUNDS"][0]["demo"], context=context)


def describe_labelling(context):
    """Explicit workflow step migrated from original notebook cell 14."""
    print("Import DemoRoundLabeller from stratscout.ingest.annotations.")
    print("DemoRoundLabeller(context.state['DEMOS'][k].stem, context=context).run()")
    for d in sorted({r["demo"] for r in context.state["DEM_ROUNDS"]}):
        n = sum((r["demo"] == d for r in context.state["DEM_ROUNDS"]))
        print(f"   {d:<40} {len(load_demo_labels(d, context=context)):>3}/{n} labelled")


def export_dataset(context):
    """Explicit workflow step migrated from original notebook cell 16."""
    pairing_table(context=context)
    build_dem_dataset(context=context)
