"""Process a snapshot of a recording inbox without controlling OBS or the game."""

import json
import math
import subprocess
from pathlib import Path

from .fps_comparison import compare_fps
from .media import identity, probe, versions
from .video_preparation import ENCODING, prepare_video
from .video_profiles import create_profile, load_profile

VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi"}


def comparison_windows(manifest):
    """Choose bounded, evenly spread samples; these do not identify gameplay events."""
    duration = min(v["duration_seconds"] for v in manifest["variants"])
    frames = math.floor(duration * 30 + 1e-6)
    length = min(300, frames)
    if length < 1:
        raise ValueError("Recording is too short for a comparison")
    starts = sorted({int((frames - length) * fraction) for fraction in (0.2, 0.5, 0.8)})
    return [(start / 30, length / 30) for start in starts]


def checked_file(directory, name, size=None):
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise ValueError(f"Missing or unsafe output file: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Output size changed: {path}")


def reuse_bundle(directory, source, profile):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or manifest["source"]["sha256"] != source["sha256"]
        or manifest["source"]["bytes"] != source["bytes"]
        or manifest["source_offset_seconds"] != 0
        or manifest["encoding"] != ENCODING
        or manifest["sampling"]["rates"] != [5, 10, 30]
        or any(manifest["profile"][key] != profile[key] for key in ("crop", "source_resolution"))
        or sorted(v["fps"] for v in manifest["variants"]) != [5, 10, 30]
    ):
        raise ValueError("Existing bundle is incomplete or uses different settings")
    for variant in manifest["variants"]:
        checked_file(directory, variant["file"], variant["bytes"])
    return manifest


def check_comparison(directory, manifest, windows):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if (
        report["status"] != "complete"
        or report["source"]["sha256"] != manifest["source"]["sha256"]
        or [(w["start_seconds"], w["duration_seconds"]) for w in report["windows"]] != windows
    ):
        raise ValueError("Existing comparison is incomplete or uses different windows")
    checked_file(directory, "report.html")
    for window in report["windows"]:
        checked_file(directory, window["video"], window["validated"]["bytes"])
        for still in window["stills"]:
            checked_file(directory, still["file"])


def process_recordings(
    inbox="data/raw/inbox",
    profile_path="configs/recording-video.json",
    output="data/processed/recordings",
    *,
    at=10,
    emit=print,
):
    """One scan, one shared layout, one bundle per source hash; preserve all inputs."""
    inbox, profile_path, output = map(lambda p: Path(p).resolve(), (inbox, profile_path, output))
    if not math.isfinite(at) or at < 0:
        raise ValueError("Preview time must be non-negative and finite")
    if output == inbox or output.is_relative_to(inbox):
        raise ValueError("Keep the output directory outside the recording inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        (p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if not videos:
        emit(f"No recordings found. Put finished MKV/MP4/MOV/AVI files directly in {inbox}")
        return []
    versions()  # Fail before asking for a crop if the executables are unavailable.
    profile = load_profile(profile_path) if profile_path.exists() else None
    results, failures = [], []
    for video in videos:
        emit(f"Checking {video.name} ...", flush=True)
        try:
            info = probe(video)
            if info["fps"] + 1e-6 < 30:
                raise ValueError("Recording must be at least 30 fps for the 5/10/30 comparison")
            if info["duration_seconds"] < 0.2:
                raise ValueError("Recording must be at least 0.2 seconds long")
            if profile is None:
                emit("Select the entire minimap once; this crop will be reused for this layout.")
                create_profile(video, profile_path, at=min(at, info["duration_seconds"] / 2))
                profile = load_profile(profile_path)
            load_profile(profile_path, (info["width"], info["height"]))
            source = identity(video)
            directory = output / source["sha256"][:16]
            reused = directory.exists()
            if reused:
                manifest = reuse_bundle(directory, source, profile)
            else:
                emit(f"Encoding and validating 5/10/30 fps: {directory}", flush=True)
                manifest = prepare_video(video, profile_path, directory)
                if manifest["source"]["sha256"] != source["sha256"]:
                    raise RuntimeError("Source changed before encoding; wait until OBS has stopped")
            windows = comparison_windows(manifest)
            comparison = directory / "comparison"
            if comparison.exists():
                check_comparison(comparison, manifest, windows)
                action = "skipped"
            else:
                emit("Generating comparison previews and report ...", flush=True)
                compare_fps(directory, windows)
                action = "resumed" if reused else "processed"
            results.append({"video": str(video), "output": str(directory), "action": action})
            emit(f"{action.capitalize()}: {video.name}\nReport: {comparison / 'report.html'}")
        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            subprocess.SubprocessError,
        ) as exc:
            failures.append(video.name)
            emit(
                f"FAILED {video.name}: {exc}\n"
                "Source and existing outputs preserved. Correct the input/profile; if an output "
                "is incomplete or incompatible, retry with --output set to a new directory.",
                flush=True,
            )
    emit(f"Finished: {len(results)} successful/skipped; {len(failures)} failed.")
    if failures:
        raise RuntimeError("Recordings failed: " + ", ".join(failures))
    return results
