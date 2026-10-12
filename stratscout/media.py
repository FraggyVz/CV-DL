"""Small, import-safe FFmpeg helpers. No shell interpolation or source writes."""

import hashlib
import json
import math
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path


def tool(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Missing {name}; install FFmpeg and add its bin directory to PATH")
    return path


def json_write(path, value):
    """Atomically update metadata inside a directory already reserved by this run."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def probe(video):
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    result = subprocess.run(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"Unreadable video {video}: {result.stderr[-2000:]}")
    data = json.loads(result.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"No video stream: {video}")
    stream = streams[0]
    try:
        rate = Fraction(stream["avg_frame_rate"])
        duration = float(stream.get("duration", data["format"].get("duration", 0)))
        if rate <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Video needs a valid frame rate and duration: {video}") from exc
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": float(rate),
        "rate": str(rate),
        "duration_seconds": duration,
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None,
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "start_time": float(stream.get("start_time", 0)),
    }


def identity(video):
    path = Path(video).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed while hashing; wait until recording has stopped")
    return {
        "path": str(path),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def versions():
    return {
        name: subprocess.check_output([tool(name), "-version"], text=True).splitlines()[0]
        for name in ("ffmpeg", "ffprobe")
    }


def run_ffmpeg(arguments, log):
    """Log progress to disk, keep memory bounded, kill the child on cancellation."""
    log = Path(log)
    started = time.perf_counter()
    with (
        log.open("w", encoding="utf-8") as errors,
        log.with_suffix(".progress").open("w") as progress,
    ):
        command = [
            tool("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-n",
            "-v",
            "error",
            "-xerror",
            "-progress",
            "pipe:1",
            *map(str, arguments),
        ]
        process = subprocess.Popen(command, stdout=progress, stderr=errors)
        try:
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(
            f"FFmpeg failed ({code}); see {log}: " + log.read_text(encoding="utf-8")[-2000:]
        )
    return time.perf_counter() - started


def validate_video(video, log):
    """Decode every frame, not just the header. Count decoded frames via FFmpeg."""
    info = probe(video)
    elapsed = run_ffmpeg(
        ["-i", video, "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "null", "-"], log
    )
    counts = [
        int(line.partition("=")[2])
        for line in Path(log).with_suffix(".progress").read_text().splitlines()
        if line.startswith("frame=")
    ]
    if not counts or counts[-1] <= 0:
        raise RuntimeError(f"No decoded frames: {video}")
    if info["frames"] is not None and info["frames"] != counts[-1]:
        raise RuntimeError(f"Advertised/decoded frame count differs: {video}")
    info.update(frames=counts[-1], bytes=Path(video).stat().st_size, decode_seconds=elapsed)
    return info
