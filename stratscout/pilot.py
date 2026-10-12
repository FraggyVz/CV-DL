"""Pilot: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from .video import (
    extract_clip,
    match_id_of,
    radar_hidden,
    stack_resized,
    video_duration,
)

PILOT_TOTAL = 400


def _pilot_signature(*, context) -> List[Dict]:
    return [{"video": v.name, "bytes": v.stat().st_size} for v in context.state["VIDEOS"]]


def _pilot_sources_match(*, context) -> bool:
    p = Path(context.config.frames_src_json)
    if not p.exists():
        return False
    saved = json.loads(p.read_text())
    return [{"video": s["video"], "bytes": s.get("bytes")} for s in saved] == _pilot_signature(
        context=context
    )


def extract_pilot_frames(*, context) -> np.ndarray:
    per_video = math.ceil(PILOT_TOTAL / len(context.state["VIDEOS"]))
    clips, sources, row = ([], [], 0)
    for v in context.state["VIDEOS"]:
        dur = video_duration(str(v))
        end = context.config.end_s if context.config.end_s is not None else dur - 20
        try:
            clip = extract_clip(
                str(v),
                context.config.start_s,
                end,
                context.state["CROPS"][v.name],
                n_frames=per_video,
            )
        except RuntimeError as e:
            print(f"skipped {v.name}: {e}")
            continue
        hidden = radar_hidden(clip)
        clip = clip[~hidden]
        sources.append(
            {
                "match_id": match_id_of(v),
                "video": v.name,
                "bytes": v.stat().st_size,
                "rows": [row, row + len(clip)],
            }
        )
        clips.append(clip)
        row += len(clip)
        print(
            f"{v.name:<40} {dur:5.0f}s  ->  {len(clip)} frames from {context.config.start_s:.0f}s to {end:.0f}s"
            + (f"   ({hidden.sum()} dropped: radar hidden)" if hidden.any() else "")
        )
    frames = stack_resized(clips)
    np.save(context.config.frames_npy, frames)
    Path(context.config.frames_src_json).write_text(json.dumps(sources, indent=1))
    print(
        "saved",
        context.config.frames_npy,
        frames.shape,
        frames.dtype,
        "| row ranges per match ->",
        context.config.frames_src_json,
    )
    return frames
