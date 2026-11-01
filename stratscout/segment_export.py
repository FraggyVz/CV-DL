"""Export reviewed tactical segments as a clip dataset, split by whole match.

Reads labels/<match>.<reviewer>.segments.json, cuts each usable segment out of the
match's prepared minimap video and writes train/ and test/ folders, each with its own
manifest.csv, plus a shared export.json. Every clip of a test match goes to test/, so no
match is ever shared between splits and training code can be given train/ alone. Clip
frames use the labeller's timing (frame index = seconds x fps). Importing this module
reads nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import TACTIC_CLASSES
from .segments import SCHEMA, match_identity

COLUMNS = [
    "clip_id",
    "split",
    "label",
    "site",
    "match_folder",
    "match_id",
    "reviewer",
    "segment_id",
    "estimated_round",
    "start_s",
    "end_s",
    "duration_s",
    "status",
    "boundary_uncertain",
    "definition_version",
    "source_video",
    "clip",
    "frames",
]


def load_label_files(labels_dir, reviewer: Optional[str] = None) -> List[Dict]:
    """Segment label files, at most one per match (pick a reviewer when there are more)."""
    pattern = f"*.{reviewer}.segments.json" if reviewer else "*.segments.json"
    files = []
    for path in sorted(Path(labels_dir).glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a segment label file")
        files.append({"path": path, "data": data})
    folders = [f["data"]["match_folder"] for f in files]
    doubled = sorted({m for m in folders if folders.count(m) > 1})
    if doubled:
        raise ValueError(f"several reviewers labelled {doubled}; pass --reviewer to choose one")
    if not files:
        raise ValueError(f"no {pattern} files in {labels_dir}")
    return files


def usable(seg: Dict, include_pending: bool = False) -> bool:
    """Reviewed segments with a class; pending ones only when asked. Never undecided/excluded."""
    allowed = ("reviewed", "pending") if include_pending else ("reviewed",)
    return seg.get("label") in TACTIC_CLASSES and seg.get("status") in allowed


def split_of(match_folder: str, test_matches: Sequence[str]) -> str:
    name = match_folder.lower()
    return "test" if any(t.lower() in name for t in test_matches) else "train"


def clip_rows(files: List[Dict], test_matches, include_pending: bool = False) -> List[Dict]:
    """One manifest row per usable segment, before any video is read."""
    rows = []
    for f in files:
        data = f["data"]
        split = split_of(data["match_folder"], test_matches)
        for seg in sorted(data["segments"], key=lambda s: s["start_s"]):
            if not usable(seg, include_pending):
                continue
            clip_id = f"{data['match_folder']}.{data['reviewer']}.s{seg['id']:03d}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "label": seg["label"],
                    "site": seg.get("site") or "",
                    "match_folder": data["match_folder"],
                    "match_id": data["match_id"],
                    "reviewer": data["reviewer"],
                    "segment_id": seg["id"],
                    "estimated_round": seg.get("estimated_round", ""),
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "duration_s": round(seg["end_s"] - seg["start_s"], 2),
                    "status": seg["status"],
                    "boundary_uncertain": bool(seg.get("boundary_uncertain")),
                    "definition_version": seg.get("definition_version", ""),
                    "source_video": "",
                    "clip": "",
                    "frames": "",
                }
            )
    return rows


def skipped_counts(files: List[Dict], include_pending: bool = False) -> Dict[str, int]:
    counts = {"undecided": 0, "excluded": 0, "pending": 0}
    for f in files:
        for seg in f["data"]["segments"]:
            if usable(seg, include_pending):
                continue
            if seg["status"] == "excluded":
                counts["excluded"] += 1
            elif seg.get("label") is None:
                counts["undecided"] += 1
            else:
                counts["pending"] += 1
    return counts


def cut_clip(video, start_s: float, end_s: float, out) -> int:
    """Copy the frames from start_s to end_s into a new MP4; returns the frame count."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    writer, written = None, 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        first, last = int(round(start_s * fps)), int(round(end_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for _ in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not written:
        raise ValueError(f"{video} has no frames between {start_s} and {end_s} s")
    return written


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_segments(
    labels_dir,
    recordings_dir,
    output,
    test_matches: Sequence[str],
    *,
    reviewer: Optional[str] = None,
    include_pending: bool = False,
    fps: int = 10,
    clips: bool = True,
) -> Dict:
    """Write <split>/manifest.csv, <split>/clips/<label>/<clip_id>.mp4 and export.json."""
    from .live_preview import match_video

    files = load_label_files(labels_dir, reviewer)
    folders = [f["data"]["match_folder"] for f in files]
    unmatched = [t for t in test_matches if not any(t.lower() in m.lower() for m in folders)]
    if unmatched:
        raise ValueError(f"--test {unmatched} matched no labelled match; check the spelling")
    rows = clip_rows(files, test_matches, include_pending)
    if not rows:
        raise ValueError(
            "no usable segments: mark them reviewed (review-segments) or pass --include-pending"
        )
    videos = {}
    for f in files:
        folder = Path(recordings_dir) / f["data"]["match_folder"]
        if match_identity(folder) != f["data"]["match_id"]:
            raise ValueError(f"{f['path']} was labelled on a different video than {folder}")
        videos[f["data"]["match_folder"]] = match_video(folder, fps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        video = videos[row["match_folder"]]
        row["source_video"] = video.as_posix()
        if clips:
            clip = Path("clips") / row["label"] / f"{row['clip_id']}.mp4"
            out = output / row["split"] / clip
            row["frames"] = cut_clip(video, row["start_s"], row["end_s"], out)
            row["clip"] = clip.as_posix()  # relative to the split folder
    for split in ("train", "test"):
        (output / split).mkdir(exist_ok=True)
        with (output / split / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(r for r in rows if r["split"] == split)
    counts = {
        split: {
            c: sum(r["split"] == split and r["label"] == c for r in rows) for c in TACTIC_CLASSES
        }
        for split in ("train", "test")
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_matches": list(test_matches),
        "include_pending": include_pending,
        "fps_variant": fps,
        "label_files": {f["path"].name: _sha256(f["path"]) for f in files},
        "splits": {r["match_folder"]: r["split"] for r in rows},
        "counts": counts,
        "skipped": skipped_counts(files, include_pending),
        "clips": len(rows),
    }
    (output / "export.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
