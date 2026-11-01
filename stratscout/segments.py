"""Tactical segment labels: the five-class record format, validation and safe saving.

Follows docs/tactic-definitions.md. One JSON file per match and reviewer, so two people
can label the same clips independently. Every change is written atomically, so closing
the window or a crash cannot lose saved segments. Importing this module reads nothing.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import TACTIC_CLASSES

SCHEMA = "stratscout-tactic-segments"
SCHEMA_VERSION = 1
# Bump when docs/tactic-definitions.md changes a class or boundary rule.
DEFINITION_VERSION = "2026-10-30-pilot"
STATUSES = ("pending", "reviewed", "excluded")
EXCLUSION_REASONS = (
    "fake: outside current tactic scope",
    "no retake attempt (save)",
    "unusable footage",
    "other: see notes",
)
# The E key cycles through these in order: E = fake, E E = save, E E E = unusable, E E E E = other.
KEY_EXCLUSION_REASONS = EXCLUSION_REASONS
SITE_OF = {"A_DIRECT": "A", "A_SPLIT": "A", "B_DIRECT": "B", "B_SPLIT": "B"}


def segments_json(match_folder, reviewer: str, labels_dir="labels") -> Path:
    """labels/<match folder>.<reviewer>.segments.json"""
    return Path(labels_dir) / f"{Path(match_folder).name}.{reviewer_name(reviewer)}.segments.json"


def reviewer_name(name: str) -> str:
    """A short file-safe reviewer name, e.g. 'gin' or 'jx'."""
    name = (name or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name):
        raise ValueError("reviewer must be 1-32 letters, digits, '-' or '_', e.g. 'gin'")
    return name


def match_identity(match_folder) -> str:
    """The source hash recorded in the prepared match's manifest.json."""
    manifest = Path(match_folder) / "manifest.json"
    if not manifest.exists():
        raise ValueError(f"no manifest.json in {match_folder}; label prepared match folders")
    match_id = json.loads(manifest.read_text(encoding="utf-8")).get("match_id")
    if not match_id:
        raise ValueError(f"{manifest} has no match_id")
    return match_id


def validate_segment(seg: Dict) -> List[str]:
    """Problems that stop a segment being saved; an empty list means it is valid."""
    problems = []
    start, end = seg.get("start_s"), seg.get("end_s")
    if start is None:
        problems.append("mark the segment start (i)")
    if end is None:
        problems.append("mark the segment end (o)")
    if start is not None and end is not None and end <= start:
        problems.append("the end must come after the start")
    status, label = seg.get("status"), seg.get("label")
    if status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    elif status == "excluded":
        if seg.get("exclusion_reason") not in EXCLUSION_REASONS:
            problems.append("an excluded segment needs a reason (e)")
        if label is not None:
            problems.append("an excluded segment has no class")
    elif label is None:
        if status == "reviewed":
            problems.append("a reviewed segment needs one of the five classes")
        elif not seg.get("undecided"):
            problems.append("choose a class (1-5), 0 = undecided, or e = exclude")
    elif label not in TACTIC_CLASSES:
        problems.append(f"unknown class {label!r}")
    if label == "RETAKE" and seg.get("site") not in ("A", "B"):
        problems.append("a RETAKE needs the planted site (a / b)")
    if label in SITE_OF and seg.get("site") not in (None, SITE_OF[label]):
        problems.append(f"{label} is a site-{SITE_OF[label]} class")
    return problems


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SegmentFile:
    """One reviewer's segment labels for one match, saved after every change.

    Opening an existing file resumes it; a file for a different match is refused.
    Reviewed segments are never removed by undo."""

    def __init__(
        self, path, *, match_id: str, match_folder: str, video: str, video_fps: float, reviewer
    ):
        self.path = Path(path)
        self.reviewer = reviewer_name(reviewer)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != SCHEMA:
                raise ValueError(f"{self.path} is not a segment label file")
            if self.data.get("match_id") != match_id:
                raise ValueError(f"{self.path} belongs to a different match; refusing to mix")
        else:
            self.data = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "definition_version": DEFINITION_VERSION,
                "classes": TACTIC_CLASSES,
                "match_id": match_id,
                "match_folder": Path(match_folder).name,
                "video": Path(video).name,
                "video_fps": video_fps,
                "time_base": "seconds in the prepared normal-speed video",
                "reviewer": self.reviewer,
                "checked_rounds": [],
                "bookmarks": [],
                "segments": [],
            }

    @property
    def segments(self) -> List[Dict]:
        return self.data["segments"]

    def add(self, seg: Dict) -> Dict:
        problems = validate_segment(seg)
        if problems:
            raise ValueError("; ".join(problems))
        record = {
            "id": max((s["id"] for s in self.segments), default=0) + 1,
            **seg,
            "site": SITE_OF.get(seg.get("label"), seg.get("site")),
            "reviewer": self.reviewer,
            "definition_version": DEFINITION_VERSION,
            "created_at": _now(),
        }
        record["start_s"], record["end_s"] = round(seg["start_s"], 2), round(seg["end_s"], 2)
        self.segments.append(record)
        self.save()
        return record

    def undo_last(self) -> Optional[Dict]:
        """Remove the most recently added segment unless it has been reviewed."""
        if not self.segments:
            return None
        last = max(self.segments, key=lambda s: s["id"])
        if last.get("status") == "reviewed":
            raise ValueError("the last segment is reviewed; undo only removes unreviewed work")
        self.segments.remove(last)
        self.save()
        return last

    def remove(self, segment_id: int) -> Dict:
        """Remove one segment by id unless it has been reviewed."""
        seg = next((s for s in self.segments if s["id"] == segment_id), None)
        if seg is None:
            raise ValueError(f"no segment #{segment_id}")
        if seg.get("status") == "reviewed":
            raise ValueError(f"#{segment_id} is reviewed; only unreviewed work can be deleted")
        self.segments.remove(seg)
        self.save()
        return seg

    def review(self, segment_ids, reviewed_by: str) -> List[Dict]:
        """Mark labelled pending segments reviewed after team agreement; all or nothing."""
        wanted = set(segment_ids)
        found = [s for s in self.segments if s["id"] in wanted]
        missing = wanted - {s["id"] for s in found}
        if missing:
            raise ValueError(f"no segment {sorted(missing)}")
        for s in found:
            if s["status"] != "pending" or s.get("label") is None:
                kind = "undecided" if s["status"] == "pending" else s["status"]
                raise ValueError(f"#{s['id']} is {kind}; only labelled pending segments")
        stamp, by = _now(), reviewer_name(reviewed_by)
        for s in found:
            s.update(status="reviewed", reviewed_by=by, reviewed_at=stamp)
        self.save()
        return found

    def toggle_checked(self, round_number: int) -> bool:
        """Mark an estimated round as watched (or unmark it); returns the new state."""
        checked = set(self.data["checked_rounds"]) ^ {round_number}
        self.data["checked_rounds"] = sorted(checked)
        self.save()
        return round_number in checked

    def add_bookmark(self, seconds: float) -> None:
        self.data["bookmarks"] = sorted(set(self.data["bookmarks"]) | {round(seconds, 2)})
        self.save()

    def summary(self) -> Dict[str, int]:
        counts = {c: 0 for c in TACTIC_CLASSES} | {"undecided": 0, "excluded": 0}
        for s in self.segments:
            key = "excluded" if s["status"] == "excluded" else s["label"] or "undecided"
            counts[key] += 1
        return counts

    def save(self) -> None:
        """Write a temporary file, then replace the old one in a single step."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
