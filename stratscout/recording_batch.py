"""Process a snapshot of a recording inbox without controlling OBS or the game."""

import json
import math
import re
import subprocess
from pathlib import Path

from .fps_comparison import compare_fps
from .media import identity, probe, versions
from .video_preparation import ENCODING, prepare_video, validate_speed
from .video_profiles import create_profile, load_profile, validate_profile

VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi"}


def match_folder_name(video):
    """A short, portable display name; source identity stays in the manifest."""
    name = re.sub(r"[^a-z0-9]+", "-", Path(video).stem.lower()).strip("-")[:100].rstrip("-")
    if not name:
        name = "match"
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
    }
    return f"match-{name}" if name in reserved else name


def bundle_directory(output, video, source):
    """Find renamed/legacy bundles by full hash; use the match name for new ones."""
    output = Path(output)
    named = output / match_folder_name(video)
    matches = []
    manifests = {}
    for directory in sorted(output.iterdir()) if output.exists() else ():
        if not directory.is_dir():
            continue
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            source_hash = manifest["source"]["sha256"]
            if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        manifests[directory] = source_hash
        if source_hash == source["sha256"]:
            matches.append(directory)
    if len(matches) > 1:
        raise ValueError("Multiple bundles match this source; use an output root with one copy")
    if matches:
        return matches[0]
    legacy = output / source["sha256"][:16]
    if legacy.exists():
        return legacy  # Let reuse validation report incomplete or corrupt legacy outputs.
    if named.exists():
        if named not in manifests:
            raise ValueError(f"Existing destination has no valid source identity: {named}")
        return output / f"{named.name}-{source['sha256'][:16]}"
    return named


def comparison_windows(manifest):
    """Choose bounded, evenly spread samples; these do not identify gameplay events."""
    duration = min(v["duration_seconds"] for v in manifest["variants"])
    frames = math.floor(duration * 30 + 1e-6)
    length = min(300, frames)
    if length < 1:
        raise ValueError("Recording is too short for a comparison")
    starts = sorted({int((frames - length) * fraction) for fraction in (0.2, 0.5, 0.8)})
    speed = manifest.get("playback_speed", 1)
    offset = manifest.get("source_offset_seconds", 0)
    return [(offset + start / (30 * speed), length / (30 * speed)) for start in starts]


class SkipRecording(Exception):
    """Explicit user choice to leave a recording untouched."""


def parse_time(value):
    """Read seconds, MM:SS or HH:MM:SS without guessing the time domain."""
    try:
        parts = value.strip().split(":")
        if len(parts) not in (1, 2, 3):
            raise ValueError
        if len(parts) > 1 and (
            any(not p.isdecimal() for p in parts[:-1])
            or not 0 <= float(parts[-1]) < 60
            or (len(parts) == 3 and not 0 <= int(parts[1]) < 60)
        ):
            raise ValueError
        seconds = sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError
        return seconds
    except (ValueError, OverflowError) as exc:
        raise ValueError("Use seconds, MM:SS or HH:MM:SS, e.g. 1431 or 23:51") from exc


def prompt_settings(info, playback_speed=2, *, input_fn=None, emit=print):
    """Three terminal prompts; answers only become metadata inside the output bundle."""
    read = input if input_fn is None else input_fn
    length = info["duration_seconds"]
    emit(
        f"Raw video: {length:.3f} seconds, {info['fps']:g} fps. "
        "Enter times from the raw video player, not the demo clock. "
        "Type s to skip this recording or Ctrl+C to stop."
    )

    def ask(label, default, parse, validate):
        while True:
            answer = read(label).strip()
            if answer.lower() in ("s", "skip"):
                raise SkipRecording
            try:
                result = default if not answer else parse(answer)
                validate(result)
                return result
            except ValueError as exc:
                emit(str(exc))

    def valid_speed(speed):
        validate_speed(speed)
        if info["fps"] / speed + 1e-6 < 30:
            raise ValueError("This speed would leave fewer than 30 source frames per game second")

    speed = ask(
        f"Demo playback speed [1 = normal, 2 = double; Enter = {playback_speed:g}]: ",
        playback_speed,
        lambda s: float(s.lower().removesuffix("x")),
        valid_speed,
    )

    def valid_start(start):
        if start >= length or (length - start) * speed < 0.2:
            raise ValueError("Start must leave at least 0.2 seconds of game time before the end")

    start = ask("Start time [Enter = 0:00]: ", 0, parse_time, valid_start)

    def valid_end(end):
        end = length if end is None else end
        if end > length or end <= start or (end - start) * speed < 0.2:
            raise ValueError(
                f"End must be after start, at most {length:.3f}s, and retain 0.2 game seconds"
            )

    end = ask("End time [Enter = end of video]: ", None, parse_time, valid_end)
    duration = None if end is None else end - start
    emit(
        f"Keeping {start:g} to {length if end is None else end:g} raw seconds; "
        f"normal-speed duration: {((length - start) if duration is None else duration) * speed:g}s."
    )
    return {"playback_speed": speed, "start": start, "duration": duration}


def checked_file(directory, name, size=None):
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise ValueError(f"Missing or unsafe output file: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Output size changed: {path}")


def reuse_bundle(directory, source, profile, resolution):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    validate_profile(manifest["profile"], resolution)
    if (
        manifest["status"] != "complete"
        or manifest["source"]["sha256"] != source["sha256"]
        or manifest["source"]["bytes"] != source["bytes"]
        or manifest["encoding"] != ENCODING
        or manifest["sampling"]["rates"] != [5, 10, 30]
        or (
            profile is not None
            and any(
                manifest["profile"][key] != profile[key] for key in ("crop", "source_resolution")
            )
        )
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
        or report.get("playback_speed", 1) != manifest.get("playback_speed", 1)
        or report["source_offset_seconds"] != manifest["source_offset_seconds"]
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
    playback_speed=2,
    interactive=True,
    input_fn=None,
    emit=print,
):
    """One scan and shared layout; readable bundles identified by source hash."""
    inbox, profile_path, output = map(lambda p: Path(p).resolve(), (inbox, profile_path, output))
    if not math.isfinite(at) or at < 0:
        raise ValueError("Preview time must be non-negative and finite")
    validate_speed(playback_speed)
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
            resolution = (info["width"], info["height"])
            if profile is not None:
                validate_profile(profile, resolution)
                emit(f"Using saved minimap crop: {profile_path}")
            source = identity(video)
            directory = bundle_directory(output, video, source)
            reused = directory.exists()
            if reused:
                manifest = reuse_bundle(directory, source, profile, resolution)
            else:
                if info["fps"] + 1e-6 < 30 or info["duration_seconds"] < 0.2:
                    raise ValueError("Recording must supply at least 30 fps and 0.2 seconds")
                settings = (
                    prompt_settings(info, playback_speed, input_fn=input_fn, emit=emit)
                    if interactive
                    else {"playback_speed": playback_speed, "start": 0, "duration": None}
                )
                if info["fps"] / settings["playback_speed"] + 1e-6 < 30:
                    raise ValueError("Recording must supply at least 30 fps after speed correction")
                duration = settings["duration"]
                if duration is None:
                    duration = info["duration_seconds"] - settings["start"]
                if profile is None:
                    if not interactive:
                        raise ValueError("Non-interactive processing needs an existing --profile")
                    emit(
                        "Select the entire minimap once; this crop will be reused for this layout."
                    )
                    create_profile(
                        video, profile_path, at=settings["start"] + min(at, duration / 2)
                    )
                    profile = load_profile(profile_path)
                emit(f"Encoding and validating 5/10/30 fps: {directory}", flush=True)
                manifest = prepare_video(
                    video,
                    profile_path,
                    directory,
                    start=settings["start"],
                    duration=settings["duration"],
                    playback_speed=settings["playback_speed"],
                )
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
        except SkipRecording:
            results.append({"video": str(video), "action": "skipped-by-user"})
            emit(f"Skipped by request: {video.name}")
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
