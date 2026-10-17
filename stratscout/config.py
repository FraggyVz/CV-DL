"""Config: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class Config:
    seed: int = 0
    data_dir: str = "data"
    video_exts: Tuple[str, ...] = (".mp4",)
    video_path: Optional[str] = None
    crop_box: Optional[Tuple[int, int, int, int]] = None
    probe_frames: int = 20
    search_frac: float = 0.45
    start_s: float = 60.0
    end_s: Optional[float] = None
    minimap_px: int = 128
    n_per_team: int = 5
    teams: Tuple[str, str] = ("T", "CT")
    seq_len: int = 32
    fps_sampled: float = 4.0
    sat_min: int = 90
    val_min: int = 90
    icon_area: Tuple[int, int] = (6, 250)
    hue_tol: float = 22.0
    bg_thresh: int = 40
    # C4 badge shape rule, measured on 494x500 spectator-radar crops (docs/bomb-detection.md).
    bomb_area: Tuple[int, int] = (120, 450)
    bomb_min_aspect: float = 1.15
    bomb_min_extent: float = 0.80
    bomb_min_corner_fill: float = 0.30
    det_stride: int = 1
    det_sigma: float = 1.0
    det_epochs: int = 25
    det_lr: float = 0.002
    det_batch: int = 32
    det_peak_thresh: float = 0.2
    det_match_radius: float = 0.035
    track_gate: float = 0.09
    track_max_miss: int = 8
    cls_epochs: int = 40
    cls_lr: float = 0.002
    cls_batch: int = 32
    cls_hidden: int = 96
    map_image: str = "mirage_radar.png"
    frames_npy: str = "pilot_frames.npy"
    icon_labels_json: str = "pilot_labels.json"
    frames_src_json: str = "pilot_frames_sources.json"
    crops_json: str = "crops.json"
    labels_dir: str = "labels"
    rounds_npz: str = "rounds_data.npz"
    out_dir: str = "runs/modular-experiment"
    # Experiment sizes are configurable so smoke checks exercise the real workflow.
    train_matches: int = 12
    val_matches: int = 3
    test_matches: int = 3
    detector_matches: int = 3
    rounds_per_match: int = 20
    detector_train_frames: int = 400
    detector_val_frames: int = 120


T_CLASSES = ["A_RUSH", "B_RUSH", "A_SPLIT", "B_SPLIT", "DEFAULT"]

CT_CLASSES = ["RETAKE", "TRICKLE", "SAVE"]

PHASES = {"pre": T_CLASSES, "post": CT_CLASSES}

CLASSES = T_CLASSES

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

N_CLASSES = len(CLASSES)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
