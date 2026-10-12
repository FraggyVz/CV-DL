"""Video: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _open_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """H.264 if this OpenCV build has it, else MPEG-4 Part 2 (still ~31x smaller)."""
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if vw.isOpened():
            return vw
    raise RuntimeError("OpenCV cannot write .mp4 on this machine")


def radar_only_clip(
    video: Path, box: Sequence[int], out: Path, fps: float = None, *, context
) -> Path:
    """Full-screen render -> radar-only clip at `fps`. Decodes once, front to back."""
    fps = fps or context.config.radar_fps
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = box
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("~" + out.name)
    vw = _open_writer(tmp, fps, (x1 - x0, y1 - y0))
    step, keep_at, written = (src_fps / fps, 0.0, 0)
    for i in range(total):
        if i < keep_at:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(np.ascontiguousarray(frame[y0:y1, x0:x1]))
        written += 1
        keep_at += step
    cap.release()
    vw.release()
    tmp.replace(out)
    return out


def crop_for(video: Path, crops: Dict) -> Optional[List[int]]:
    """Crop box by file name, else by byte size: renaming a video does not change its size,
    so a renamed file still finds its box (crops.json is keyed by the old name)."""
    if video.name in crops:
        return crops[video.name]["crop_box"]
    size = video.stat().st_size
    hits = [c["crop_box"] for c in crops.values() if c.get("bytes") == size]
    return hits[0] if hits else None


def shrink_all(force: bool = False, *, context) -> None:
    crops_path = Path(context.config.crops_json)
    if not crops_path.exists():
        print(f"{crops_path} missing — run stratscout_prototype.ipynb §1-3 first")
        return
    crops = json.loads(crops_path.read_text())
    videos = sorted(
        (
            p
            for p in Path(context.config.data_dir).glob("*")
            if p.suffix.lower() == ".mp4" and (not p.name.startswith((".", "~")))
        )
    )
    for v in videos:
        out = Path(context.config.radar_dir) / f"{v.stem}.mp4"
        box = crop_for(v, crops)
        if box is None:
            print(f"{v.name:<36} no crop box in {crops_path.name} — run the main notebook §3")
            continue
        if out.exists() and out.stat().st_mtime > v.stat().st_mtime and (not force):
            print(f"{v.name:<36} already done ({out.stat().st_size / 1000000.0:.0f} MB)")
            continue
        t = time.time()
        radar_only_clip(v, box, out, context=context)
        print(
            f"{v.name:<36} {v.stat().st_size / 1000000.0:7.0f} MB -> {out.stat().st_size / 1000000.0:5.1f} MB   [{time.time() - t:.0f}s]"
        )
