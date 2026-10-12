"""Contracts: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import T_CLASSES

TEAM_IDX = {"T": 0, "CT": 1}


def check_contract(name: str, arr, expected: Tuple) -> None:
    got = tuple(arr.shape)
    if len(got) != len(expected) or any((e is not None and e != g for e, g in zip(expected, got))):
        raise ValueError(f"[contract] {name}: expected {expected}, got {got}")
    print(f"[contract] {name}: {got} OK")


@dataclass
class Round:
    match_id: str
    round_idx: int
    label: str
    positions: Optional[np.ndarray] = None
    frames: Optional[np.ndarray] = None
    winner: Optional[str] = None
    plant_site: Optional[str] = None
    ct_label: Optional[str] = None
    post_positions: Optional[np.ndarray] = None
    post_frames: Optional[np.ndarray] = None

    @property
    def y(self) -> int:
        return T_CLASSES.index(self.label)

    @property
    def planted(self) -> bool:
        return self.plant_site is not None

    def label_of(self, phase: str) -> Optional[str]:
        return self.label if phase == "pre" else self.ct_label

    def positions_of(self, phase: str) -> Optional[np.ndarray]:
        return self.positions if phase == "pre" else self.post_positions

    def frames_of(self, phase: str) -> Optional[np.ndarray]:
        return self.frames if phase == "pre" else self.post_frames

    @property
    def tactic_worked(self) -> Optional[bool]:
        """The attackers' tactic worked if T won the round."""
        return None if self.winner is None else self.winner == "T"

    @property
    def response_worked(self) -> Optional[bool]:
        """The defenders' response worked if CT won a round in which the bomb was planted."""
        return None if self.winner is None or not self.planted else self.winner == "CT"
