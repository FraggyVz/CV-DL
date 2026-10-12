"""Background: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .video import crop_for


def radar_background(n: int = 120, *, context) -> Optional[np.ndarray]:
    """RGB median radar from the first radar-only clip, else from a full render + its crop."""
    clips = sorted(Path(context.config.radar_dir).glob("*.mp4"))
    crops = (
        json.loads(Path(context.config.crops_json).read_text())
        if Path(context.config.crops_json).exists()
        else {}
    )
    if clips:
        src, box = (clips[0], None)
    else:
        vids = [
            p for p in sorted(Path(context.config.data_dir).glob("*.mp4")) if crop_for(p, crops)
        ]
        if not vids:
            return None
        src, box = (vids[0], crop_for(vids[0], crops))
    cap = cv2.VideoCapture(str(src))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in np.linspace(total * 0.05, total * 0.9, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f if box is None else f[box[1] : box[3], box[0] : box[2]])
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def load_plot_bg(*, context) -> Optional[np.ndarray]:
    raw = cv2.imread(context.config.map_image, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        a = raw[..., 3:4].astype(np.float32) / 255
        rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        return (rgb * a + np.array([12, 16, 24], np.float32) * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
