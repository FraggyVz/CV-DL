"""Config: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

CLASSES = ["A_RUSH", "B_RUSH", "A_SPLIT", "B_SPLIT", "RETAKE"]

EXCLUDED = "EXCLUDED"


@dataclass
class DemCfg:
    data_dir: str = "data"
    radar_dir: str = "data_radar"
    crops_json: str = "crops.json"
    radar_fps: float = 8.0
    demo_dir: str = "demos"
    map_name: str = "de_mirage"
    tickrate: int = 64
    seq_len: int = 32
    n_per_team: int = 5
    overview_pos_x: float = -3230.0
    overview_pos_y: float = 1713.0
    overview_scale: float = 5.0
    overview_px: int = 1024
    radar_fit_json: str = "radar_fit.json"
    labels_dir: str = "labels"
    dem_npz: str = "dem_rounds.npz"
    map_image: str = "mirage-radar.png"


SITE_ANCHORS = {"A": (0.545, 0.77), "B": (0.23, 0.276)}

TEAM_COLOUR = {"T": "#ff9020", "CT": "#40a0ff"}
