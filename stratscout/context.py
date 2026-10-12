"""Explicit, isolated state for one experiment; no work is performed on construction."""

from dataclasses import dataclass, field
from typing import Any

from .config import Config


@dataclass
class RunContext:
    config: Config = field(default_factory=Config)
    device: str = "cpu"
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state.setdefault("N_PLAYERS", self.config.n_per_team * 2)
        self.state.setdefault("_GRID_CACHE", {})
        self.state.setdefault("_POS_CACHE", {})

    def require(self, *names):
        missing = [name for name in names if name not in self.state]
        if missing:
            raise RuntimeError("Missing workflow inputs: " + ", ".join(missing))
