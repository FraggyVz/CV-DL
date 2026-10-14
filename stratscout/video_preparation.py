"""One decode/crop, aligned sampling and independently encoded video variants."""

import math
import time
from pathlib import Path

from .media import identity, json_write, probe, run_ffmpeg, validate_video, versions
from .video_profiles import load_profile

ENCODING = {
    "codec": "libx264",
    "crf": 18,
    "preset": "medium",
    "pixel_format": "yuv420p",
    "audio": False,
    "faststart": True,
}


def validate_speed(playback_speed):
    if (
        isinstance(playback_speed, bool)
        or not isinstance(playback_speed, (int, float))
        or not math.isfinite(playback_speed)
        or playback_speed <= 0
    ):
        raise ValueError("Playback speed must be a positive finite number (1 for normal, 2 for 2x)")
    return playback_speed


def sampling_graph(crop, rates, playback_speed=1):
    x0, y0, x1, y1 = crop
    base = max(rates)
    branches = "".join(f"[s{rate}]" for rate in rates)
    graph = (
        f"[0:v:0]setpts=(PTS-STARTPTS)*{playback_speed:g},"
        f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}:exact=1,"
        f"fps={base}:start_time=0:round=near,split={len(rates)}{branches}"
    )
    for rate in rates:
        graph += f";[s{rate}]select='not(mod(n,{base // rate}))',setpts=N/({rate}*TB)[v{rate}]"
    return graph


def prepare_video(
    video,
    profile_path,
    output,
    *,
    rates=(5, 10, 30),
    start=0,
    duration=None,
    match_id=None,
    playback_speed=1,
):
    video, output = Path(video).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output destination already exists: {output}")
    rates = sorted(rates)
    validate_speed(playback_speed)
    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(type(r) is not int or r not in (5, 10, 30) for r in rates)
    ):
        raise ValueError("Choose distinct frame rates from 5, 10 and 30")
    if (
        not math.isfinite(start)
        or start < 0
        or (duration is not None and (not math.isfinite(duration) or duration <= 0))
    ):
        raise ValueError("Start must be non-negative and duration must be positive and finite")
    tools = versions()
    info = probe(video)
    if info["fps"] / playback_speed + 1e-6 < max(rates):
        raise ValueError(
            f"Source is {info['fps']:g} fps at {playback_speed:g}x playback: "
            f"only {info['fps'] / playback_speed:g} fps in game time, below requested {max(rates)}"
        )
    profile = load_profile(profile_path, (info["width"], info["height"]))
    bounded_duration = duration is not None
    remaining = info["duration_seconds"] - start
    if remaining <= 0 or (duration is not None and duration > remaining + 1 / info["fps"]):
        raise ValueError("Requested time interval is outside the source video")
    duration = min(duration, remaining) if duration is not None else remaining
    output_duration = duration * playback_speed
    if output_duration < 1 / min(rates):
        raise ValueError("Interval must contain at least one frame at the lowest requested rate")
    source = identity(video)
    profile_match = (
        profile.get("match_id") if profile.get("source_sha256") == source["sha256"] else None
    )
    match_id = match_id or profile_match or source["sha256"]
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("Match identity must be a non-empty string")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 2,
        "status": "processing",
        "match_id": match_id,
        "source": source,
        "source_video": info,
        "profile": profile,
        "source_offset_seconds": start,
        "requested_duration_seconds": duration,
        "playback_speed": playback_speed,
        "expected_output_duration_seconds": output_duration,
        "timeline": {
            "trim_units": "source-video seconds",
            "output_units": "elapsed game-time seconds from trim start; not absolute demo time",
            "source_mapping": "source_offset_seconds + output_elapsed_seconds / playback_speed",
            "assumption": "constant replay speed; no pauses, seeks or capture stalls",
        },
        "sampling": {
            "base_fps": max(rates),
            "rates": rates,
            "rounding": "near",
            "origin": "first decoded frame after requested source offset",
            "timestamps": "source_offset_seconds + frame_index / (fps * playback_speed)",
        },
        "encoding": ENCODING,
        "tools": tools,
        "variants": [],
    }
    path = output / "manifest.json"
    json_write(path, manifest)
    began = time.perf_counter()
    try:
        graph = sampling_graph(profile["crop"], rates, playback_speed)
        arguments = ["-ss", str(start)]
        # Some containers report the last frame's timestamp as duration. Applying
        # that header as a hard cutoff would discard the final observation.
        if bounded_duration:
            arguments += ["-t", str(duration)]
        arguments += ["-i", str(video), "-filter_complex", graph]
        for rate in rates:
            arguments += [
                "-map",
                f"[v{rate}]",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(rate),
                "-fps_mode",
                "cfr",
                "-movflags",
                "+faststart",
                str(output / f"minimap-{rate}fps.partial.mp4"),
            ]
        manifest["conversion_seconds"] = run_ffmpeg(arguments, output / "encoding.log")
        for rate in rates:
            temporary = output / f"minimap-{rate}fps.partial.mp4"
            result = validate_video(temporary, output / f"decode-{rate}fps.log")
            x0, y0, x1, y1 = profile["crop"]
            if (result["width"], result["height"]) != (x1 - x0, y1 - y0) or abs(
                result["fps"] - rate
            ) > 1e-6:
                raise RuntimeError("Encoded resolution or rate differs from requested settings")
            if (
                abs(result["duration_seconds"] - output_duration)
                > 1 / rate + 1 / max(rates) + 0.001
            ):
                raise RuntimeError("Encoded duration differs by more than one sampling interval")
            if (
                abs(result["start_time"]) > 1e-6
                or abs(result["duration_seconds"] - result["frames"] / rate) > 1 / rate + 1e-6
            ):
                raise RuntimeError(
                    "Encoded timestamps do not start at zero with the expected duration"
                )
            result.update(file=f"minimap-{rate}fps.mp4", match_id=match_id)
            manifest["variants"].append(result)
        base_count = manifest["variants"][-1]["frames"]
        for result in manifest["variants"]:
            if result["frames"] != math.ceil(base_count / (max(rates) // int(result["fps"]))):
                raise RuntimeError("Output frame counts do not share the canonical sampling grid")
        if (
            video.stat().st_size != source["bytes"]
            or video.stat().st_mtime_ns != source["mtime_ns"]
        ):
            raise RuntimeError("Source changed during processing")
        for rate in rates:
            (output / f"minimap-{rate}fps.partial.mp4").rename(output / f"minimap-{rate}fps.mp4")
        manifest.update(status="complete", total_seconds=time.perf_counter() - began)
    except BaseException as exc:
        manifest.update(
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
            total_seconds=time.perf_counter() - began,
        )
        json_write(path, manifest)
        raise
    json_write(path, manifest)
    return manifest
