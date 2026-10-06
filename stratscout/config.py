"""Versioned, validated calibration profiles. Coordinates use x-right, y-down."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

TEAMS = ("T", "CT")


@dataclass(frozen=True)
class TeamSettings:
    hue: int
    hue_tolerance: int = 10
    saturation_min: int = 100
    value_min: int = 90
    area_min: int = 5
    area_max: int = 200
    aspect_max: float = 3.0
    fill_min: float = 0.35

    def validate(self) -> None:
        for name, low, high in (
            ("hue", 0, 179), ("hue_tolerance", 0, 89),
            ("saturation_min", 1, 255), ("value_min", 1, 255),
            ("area_min", 1, 1_000_000), ("area_max", 1, 1_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.area_min > self.area_max:
            raise ValueError("area_min must not exceed area_max")
        for name, low, high in (("aspect_max", 1, 100), ("fill_min", 0, 1)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


@dataclass(frozen=True)
class Profile:
    source_resolution: tuple[int, int]  # width, height
    crop: tuple[int, int, int, int]  # x0, y0, exclusive x1, y1
    teams: dict[str, TeamSettings]
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported profile schema_version (expected 1)")
        if len(self.source_resolution) != 2 or len(self.crop) != 4:
            raise ValueError("source_resolution needs two integers; crop needs four")
        if any(type(v) is not int for v in (*self.source_resolution, *self.crop)):
            raise ValueError("Resolution and crop coordinates must be integers")
        width, height = self.source_resolution
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Crop must have positive area and lie inside the source frame")
        if set(self.teams) != set(TEAMS):
            raise ValueError("Profile must contain exactly T and CT settings")
        for settings in self.teams.values():
            settings.validate()
        t, ct = (self.teams[team] for team in TEAMS)
        difference = abs(t.hue - ct.hue)
        distance = min(difference, 180 - difference)
        if distance <= t.hue_tolerance + ct.hue_tolerance:
            raise ValueError("T and CT hue ranges overlap; narrow tolerances or resample colours")

    def to_dict(self) -> dict:
        return asdict(self)


def load_profile(path: str | Path) -> Profile:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = Profile(
            source_resolution=tuple(data["source_resolution"]),
            crop=tuple(data["crop"]),
            teams={name: TeamSettings(**values) for name, values in data["teams"].items()},
            schema_version=data["schema_version"],
        )
        profile.validate()
        return profile
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid profile {path}: {exc}") from exc


def save_profile(profile: Profile, path: str | Path) -> None:
    profile.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(profile.to_dict(), stream, indent=2, allow_nan=False)
        stream.write("\n")
