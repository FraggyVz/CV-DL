"""Independent state for demo-development workflows."""

from dataclasses import dataclass, field
from typing import Any

from .config import DemCfg


@dataclass
class IngestContext:
    config: DemCfg = field(default_factory=DemCfg)
    state: dict[str, Any] = field(default_factory=dict)
