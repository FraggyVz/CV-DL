"""Prepare and classify round recordings for the local web app."""

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .media import probe
from .video_preparation import prepare_video

TCN_MODELS = Path(__file__).resolve().parents[1] / "runs" / "tactic-tcn-clipdata-final" / "models"
GRU_MODELS = (
    Path(__file__).resolve().parents[1] / "runs" / "tactic-gru-clipdata-final-20261121" / "models"
)
VIDEO_PROFILE = Path(os.environ.get("STRATSCOUT_VIDEO_PROFILE", "configs/recording-video.json"))
MODEL_SIZE = (494, 500)
MODEL_FPS = 10.0
ACTIVE_ANALYSES: set[str] = set()
ACTIVE_LOCK = threading.Lock()


class AnalysisInProgress(RuntimeError):
    """Raised when another analysis is already running for the recording."""


def _analysis_is_active(upload_dir: Path, video_id: str) -> bool:
    path = _state_path(upload_dir, video_id)
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    return (
        state["status"] in ("queued", "preparing", "detecting", "classifying")
        and state["analysis_id"] in ACTIVE_ANALYSES
    )


def _video_metadata_path(upload_dir: Path, video_id: str) -> Path:
    return upload_dir / f"{video_id}.video.json"


def _write_updates(updates: dict[Path, dict]) -> None:
    temporary_paths = []
    try:
        for path, data in updates.items():
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary_paths.append((temporary, path))
        for temporary, path in temporary_paths:
            os.replace(temporary, path)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def _remove_video_files(upload_dir: Path, video_id: str) -> None:
    for suffix in (".mp4", ".video.json", ".segments.json", ".analysis.json", ".browser.mp4"):
        (upload_dir / f"{video_id}{suffix}").unlink(missing_ok=True)
    analysis_directory = upload_dir / "analysis" / video_id
    if analysis_directory.is_dir():
        shutil.rmtree(analysis_directory)


def delete_recording(upload_dir: Path, video_id: str) -> dict:
    """Delete one recording and compact later round numbers in its match."""
    upload_dir = Path(upload_dir)
    with ACTIVE_LOCK:
        metadata_path = _video_metadata_path(upload_dir, video_id)
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        if not (upload_dir / f"{video_id}.mp4").is_file() and not metadata_path.is_file():
            raise FileNotFoundError(video_id)

        match_id = metadata.get("match_id")
        deleted_round = metadata.get("round_number")
        later_videos = []
        if match_id and isinstance(deleted_round, int):
            for path in upload_dir.glob("*.video.json"):
                candidate_id = path.name.removesuffix(".video.json")
                if candidate_id == video_id:
                    continue
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    candidate.get("match_id") == match_id
                    and isinstance(candidate.get("round_number"), int)
                    and candidate["round_number"] > deleted_round
                ):
                    later_videos.append((candidate_id, path, candidate))

        affected_ids = [video_id, *(candidate_id for candidate_id, _, _ in later_videos)]
        if any(_analysis_is_active(upload_dir, candidate_id) for candidate_id in affected_ids):
            raise AnalysisInProgress(
                "Wait for this round and later rounds to finish analysing before removing it."
            )

        updates: dict[Path, dict] = {}
        for candidate_id, path, candidate in later_videos:
            candidate["round_number"] -= 1
            updates[path] = candidate
            segments_path = upload_dir / f"{candidate_id}.segments.json"
            if segments_path.is_file():
                segments = json.loads(segments_path.read_text(encoding="utf-8"))
                for segment in segments:
                    if segment["round_number"] > deleted_round:
                        segment["round_number"] -= 1
                updates[segments_path] = segments
            analysis_path = _state_path(upload_dir, candidate_id)
            if analysis_path.is_file():
                state = json.loads(analysis_path.read_text(encoding="utf-8"))
                for result in state.get("results", []):
                    if result["round_number"] > deleted_round:
                        result["round_number"] -= 1
                updates[analysis_path] = state
        _write_updates(updates)
        _remove_video_files(upload_dir, video_id)
        return {
            "deleted_id": video_id,
            "renumbered": {
                candidate_id: candidate["round_number"]
                for candidate_id, _, candidate in later_videos
            },
        }


def reset_recordings(upload_dir: Path) -> int:
    """Delete all app-managed recording data while preserving unrelated files."""
    upload_dir = Path(upload_dir)
    with ACTIVE_LOCK:
        video_ids = set()
        for suffix in (
            ".browser.mp4",
            ".video.json",
            ".segments.json",
            ".analysis.json",
            ".mp4",
        ):
            for path in upload_dir.glob(f"*{suffix}"):
                try:
                    video_ids.add(str(uuid.UUID(path.name[: -len(suffix)])))
                except ValueError:
                    continue
        analysis_root = upload_dir / "analysis"
        if analysis_root.is_dir():
            for path in analysis_root.iterdir():
                if path.is_dir():
                    try:
                        video_ids.add(str(uuid.UUID(path.name)))
                    except ValueError:
                        continue
        if any(_analysis_is_active(upload_dir, video_id) for video_id in video_ids):
            raise AnalysisInProgress("Wait for all round analyses to finish before resetting.")
        for video_id in video_ids:
            _remove_video_files(upload_dir, video_id)
        return len(video_ids)


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_path(upload_dir: Path, video_id: str) -> Path:
    return upload_dir / f"{video_id}.analysis.json"


def _analysis_directory(upload_dir: Path, video_id: str, analysis_id: str) -> Path:
    return upload_dir / "analysis" / video_id / analysis_id


def queue_analysis(upload_dir: Path, video_id: str) -> dict:
    """Create a durable queued record and reject duplicate in-flight work."""
    upload_dir = Path(upload_dir)
    state_path = _state_path(upload_dir, video_id)
    with ACTIVE_LOCK:
        if not (upload_dir / f"{video_id}.mp4").is_file():
            raise FileNotFoundError(video_id)
        if state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous["status"] in ("queued", "preparing", "detecting", "classifying"):
                if previous["analysis_id"] in ACTIVE_ANALYSES:
                    raise AnalysisInProgress("An analysis is already running for this recording.")
                previous.update(
                    status="failed",
                    stage="failed",
                    message="The app stopped before this analysis finished. Run it again.",
                    error="Analysis interrupted by an application restart.",
                    updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                _write_json(state_path, previous)

        analysis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = {
            "analysis_id": analysis_id,
            "video_id": video_id,
            "status": "queued",
            "stage": "queued",
            "message": "Analysis is queued.",
            "progress": 0,
            "error": None,
            "results": [],
            "created_at": now,
            "updated_at": now,
        }
        _analysis_directory(upload_dir, video_id, analysis_id).mkdir(parents=True, exist_ok=False)
        _write_json(state_path, state)
        ACTIVE_ANALYSES.add(analysis_id)
        return state


def get_analysis(upload_dir: Path, video_id: str) -> dict:
    path = _state_path(Path(upload_dir), video_id)
    if not path.is_file():
        return {
            "video_id": video_id,
            "status": "idle",
            "stage": "idle",
            "message": "Save at least one segment, then run analysis.",
            "progress": 0,
            "error": None,
            "results": [],
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] in ("queued", "preparing", "detecting", "classifying"):
        with ACTIVE_LOCK:
            if state["analysis_id"] not in ACTIVE_ANALYSES:
                state.update(
                    status="failed",
                    stage="failed",
                    message="The app stopped before this analysis finished. Run it again.",
                    error="Analysis interrupted by an application restart.",
                    updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                _write_json(path, state)
    return state


def invalidate_analysis(upload_dir: Path, video_id: str) -> None:
    path = _state_path(Path(upload_dir), video_id)
    if not path.is_file():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] in ("queued", "preparing", "detecting", "classifying"):
        raise AnalysisInProgress("Cannot change segment boundaries during analysis.")
    if state["status"] == "complete":
        state.update(
            status="stale",
            stage="stale",
            message="Segment boundaries changed. Run analysis again for the updated segments.",
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _write_json(path, state)


def _set_progress(
    upload_dir: Path,
    video_id: str,
    state: dict,
    *,
    status: str,
    message: str,
    progress: int,
) -> None:
    state.update(
        status=status,
        stage=status,
        message=message,
        progress=progress,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _write_json(_state_path(Path(upload_dir), video_id), state)


def _profile_for(
    source: Path,
    output: Path,
    info: dict,
    speed: float,
    prepared_clip: bool,
) -> tuple[Path, float]:
    from .video_profiles import load_profile

    size = (info["width"], info["height"])
    if prepared_clip:
        if size != MODEL_SIZE or abs(info["fps"] - MODEL_FPS) > 0.5:
            raise ValueError(
                f"Already-prepared clips must be 494x500 at 10 fps; "
                f"{source.name} is {size[0]}x{size[1]} at {info['fps']:g} fps."
            )
        profile_path = output / "model-ready-profile.json"
        _write_json(
            profile_path,
            {
                "schema_version": 1,
                "source_resolution": list(size),
                "crop": [0, 0, *size],
            },
        )
        return profile_path, 1.0
    if not VIDEO_PROFILE.is_file():
        raise ValueError(
            f"{source.name} is {size[0]}x{size[1]} at {info['fps']:g} fps. "
            "Create configs/recording-video.json with the recording guide's crop workflow, "
            "then restart the app. Already-prepared model clips must be 494x500 at 10 fps."
        )
    load_profile(VIDEO_PROFILE, size)
    return VIDEO_PROFILE, speed


def _prepare_segments(
    upload_dir: Path,
    video_id: str,
    analysis_id: str,
    segments: list[dict],
    playback_speed: float,
    prepared_clip: bool,
    progress: Callable[[int, str], None],
) -> list[tuple[dict, Path]]:
    source = Path(upload_dir) / f"{video_id}.mp4"
    info = probe(source)
    directory = _analysis_directory(Path(upload_dir), video_id, analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    profile, speed = _profile_for(source, directory, info, playback_speed, prepared_clip)
    prepared = []
    for index, segment in enumerate(segments, 1):
        progress(
            round(5 + (index - 1) / len(segments) * 35),
            f"Preparing segment {index} of {len(segments)} (round {segment['round_number']}).",
        )
        output = directory / "prepared" / f"{index:03d}-{segment['id']}"
        manifest = prepare_video(
            source,
            profile,
            output,
            rates=(10,),
            start=segment["start_s"],
            duration=segment["end_s"] - segment["start_s"],
            match_id=video_id,
            playback_speed=speed,
        )
        path = output / f"{video_id}.segment-{segment['id']}.mp4"
        (output / "minimap-10fps.mp4").replace(path)
        manifest["variants"][0]["file"] = path.name
        _write_json(output / "manifest.json", manifest)
        prepared.append((segment, path))
    return prepared


def _predictor():
    from . import gru_predict

    return gru_predict


def _detect_and_predict(
    prepared: list[tuple[dict, Path]],
    min_confidence: float,
    progress: Callable[[str, int, str], None],
) -> list[dict]:
    gru_predict = _predictor()
    paths = [path for _, path in prepared]
    progress("detecting", 45, "Detecting player and bomb icons in the prepared segments.")
    tcn_members = gru_predict.load_models("tcn", device="cpu")
    gru_members = gru_predict.load_models("gru", device="cpu")
    steps = {
        tcn_members[0][2].step,
        int(gru_members[0][1]["settings"]["step"]),
    }
    if len(steps) != 1:
        raise ValueError(f"TCN and GRU checkpoints use different frame steps: {sorted(steps)}")
    step = steps.pop()
    detections, warnings = gru_predict.detect(
        paths,
        background_from=paths,
        log=lambda message: progress("detecting", 55, message),
    )
    progress("detecting", 65, "Extracting the model's detector and ResNet features.")
    features = gru_predict.clip_features(
        paths,
        detections,
        step=step,
        device="cpu",
        log=lambda message: progress("detecting", 75, message),
    )
    progress("classifying", 85, "Running the saved TCN and GRU ensembles.")
    predictions = gru_predict.predict_clips(
        paths,
        models=("tcn", "gru"),
        min_confidence=min_confidence,
        device="cpu",
        features=features,
    )
    by_name = {result["clip"]: result for result in predictions}
    results = []
    for segment, path in prepared:
        result = by_name[path.stem]
        detection = detections[path.stem]
        result_warnings = list(dict.fromkeys(result["warnings"] + warnings[path.stem]))
        results.append(
            {
                "segment_id": segment["id"],
                "round_number": segment["round_number"],
                "phase": segment["phase"],
                "start_s": segment["start_s"],
                "end_s": segment["end_s"],
                "detection": {
                    "frames": int(detection["n_frames"]),
                    "player_icon_observations": int(len(detection["players"])),
                    "bomb_visible_frames": sum(bool(frame[0]) for frame in detection["bomb"]),
                },
                "warnings": result_warnings,
                "predictions": result["predictions"],
            }
        )
    return results


def run_analysis(
    upload_dir: Path,
    video_id: str,
    analysis_id: str,
    segments: list[dict],
    *,
    playback_speed: float,
    prepared_clip: bool,
    min_confidence: float,
) -> None:
    """Run preparation, detection and both model ensembles as a durable background job."""
    upload_dir = Path(upload_dir)
    state_path = _state_path(upload_dir, video_id)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def prepare_progress(progress: int, message: str) -> None:
        _set_progress(
            upload_dir,
            video_id,
            state,
            status="preparing",
            message=message,
            progress=progress,
        )

    def predict_progress(stage: str, progress: int, message: str) -> None:
        _set_progress(
            upload_dir,
            video_id,
            state,
            status=stage,
            message=message,
            progress=progress,
        )

    try:
        prepared = _prepare_segments(
            upload_dir,
            video_id,
            analysis_id,
            segments,
            playback_speed,
            prepared_clip,
            prepare_progress,
        )
        results = _detect_and_predict(prepared, min_confidence, predict_progress)
        state.update(results=results, error=None)
        _set_progress(
            upload_dir,
            video_id,
            state,
            status="complete",
            message="Detection and both model predictions are ready.",
            progress=100,
        )
    except Exception as exc:
        state.update(error=f"{type(exc).__name__}: {exc}")
        _set_progress(
            upload_dir,
            video_id,
            state,
            status="failed",
            message="Analysis failed. See the error below and correct the input before retrying.",
            progress=state["progress"],
        )
        raise
    finally:
        with ACTIVE_LOCK:
            ACTIVE_ANALYSES.discard(analysis_id)
