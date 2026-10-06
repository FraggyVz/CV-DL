"""Evaluate reviewed point labels with one-to-one matching in native crop pixels."""

import json
import math
from pathlib import Path

import numpy as np

from .config import TEAMS


def validate_points(points: list) -> None:
    if not isinstance(points, list):
        raise ValueError("Each team must contain a list of x/y point objects")
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            raise ValueError("Each point needs x and y")
        for axis in ("x", "y"):
            value = point[axis]
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Point coordinates must be finite normalised values in [0, 1]")


def match_points(predicted: list, truth: list, width: int, height: int) -> tuple[int, int, int]:
    """Maximum-cardinality one-to-one matches within 2% of the crop diagonal."""
    validate_points(predicted)
    validate_points(truth)
    if not predicted or not truth:
        return 0, len(predicted), len(truth)
    p = np.array([[p["x"], p["y"]] for p in predicted]) * [width, height]
    g = np.array([[p["x"], p["y"]] for p in truth]) * [width, height]
    distances = np.linalg.norm(p[:, None] - g[None], axis=-1)
    radius = 0.02 * math.hypot(width, height)
    edges = [sorted(np.flatnonzero(row <= radius), key=lambda j: row[j]) for row in distances]
    owners = {}

    def assign(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or assign(owners[target], seen):
                owners[target] = index
                return True
        return False

    matches = sum(assign(index, set()) for index in range(len(predicted)))
    return matches, len(predicted) - matches, len(truth) - matches


def evaluate_run(run: str | Path, annotations: str | Path, output: str | Path) -> dict:
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    gold = json.loads(Path(annotations).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(gold, dict):
        raise ValueError("Run metadata and annotations must be JSON objects")
    if metadata.get("status") != "complete":
        raise ValueError("Only complete detection runs can be evaluated")
    if gold.get("schema_version") != 1:
        raise ValueError("Annotation schema_version must be 1")
    if gold.get("video_sha256") != metadata["input"]["sha256"]:
        raise ValueError("Annotations refer to a different video SHA-256")
    for key in ("crop", "source_resolution"):
        if gold.get(key) != metadata["profile"][key]:
            raise ValueError(f"Annotation {key} does not match the detection run")
    frames = gold.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Annotations need a non-empty frames list")
    selected, seen, rounds = {}, set(), {"calibration": set(), "evaluation": set()}
    counts = {"calibration": 0, "evaluation": 0}
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("Each annotated frame must be a JSON object")
        index = row.get("frame_index")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("Annotation frame indices must be unique non-negative integers")
        if index >= metadata["frames_processed"]:
            raise ValueError(f"Annotation frame {index} is outside the processed video")
        seen.add(index)
        split = row.get("split")
        round_id = row.get("round_id")
        if split not in rounds or not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("Every annotation needs a round_id and calibration/evaluation split")
        if row.get("review_status") != "reviewed":
            raise ValueError(f"Frame {index} has not been manually reviewed")
        for team in TEAMS:
            if team not in row:
                raise ValueError(f"Frame {index} is missing the {team} list (use [] for none)")
            validate_points(row[team])
        counts[split] += 1
        rounds[split].add(round_id)
        if split == "evaluation":
            selected[index] = row
    if rounds["calibration"] & rounds["evaluation"]:
        raise ValueError("Calibration and evaluation frames must come from separate rounds")
    if not selected:
        raise ValueError("No reviewed evaluation frames")
    totals = {team: np.zeros(3, dtype=int) for team in TEAMS}
    x0, y0, x1, y1 = gold["crop"]
    found = set()
    with (run / "detections.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            index = record["frame_index"]
            if index not in selected:
                continue
            if index in found:
                raise ValueError(f"Duplicate detection record for frame {index}")
            found.add(index)
            for team in TEAMS:
                totals[team] += match_points(record[team], selected[index][team], x1 - x0, y1 - y0)
    if found != set(selected):
        raise ValueError("Detection file is missing annotated evaluation frames")
    metrics = {}
    for team, (tp, fp, fn) in totals.items():
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        metrics[team] = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                         "precision": precision, "recall": recall,
                         "meets_target": bool(tp + fn > 0 and precision >= .90 and recall >= .90)}
    protocol = (counts["calibration"] >= 10 and counts["evaluation"] >= 20
                and len(rounds["calibration"] | rounds["evaluation"]) >= 3)
    report = {"video_sha256": gold["video_sha256"], "reviewed_frames": counts,
              "matching_radius_fraction_of_crop_diagonal": .02, "teams": metrics,
              "sample_protocol_met": protocol,
              "quantitative_target_met": protocol and all(m["meets_target"] for m in metrics.values()),
              "scope": "This recording only; inspect difficult cases before declaring the milestone complete."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
