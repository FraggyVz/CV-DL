"""Visual/timing/storage comparisons only; no detector or tracker dependency."""

import html
import json
import math
import subprocess
from pathlib import Path

from .media import json_write, tool, validate_video


def parse_window(value):
    try:
        start, duration = map(float, value.split(":"))
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            raise ValueError
        return start, duration
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid window {value!r}; use source-seconds:duration-seconds") from exc


def compare_fps(directory, windows, *, output=None):
    import cv2
    import numpy as np

    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory / "comparison"
    if output.exists():
        raise FileExistsError(f"Comparison destination already exists: {output}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Comparison requires a complete, validated preparation bundle")
    variants = sorted(manifest["variants"], key=lambda item: item["fps"])
    if [v["fps"] for v in variants] != [5, 10, 30]:
        raise ValueError("Comparison requires all three rates: 5, 10, 30")
    offset = manifest["source_offset_seconds"]
    speed = manifest.get("playback_speed", 1)
    windows = [parse_window(w) if isinstance(w, str) else w for w in windows]
    common_duration = min(v["duration_seconds"] for v in variants) / speed
    if not windows or any(
        start < offset or start + length > offset + common_duration + 1e-6
        for start, length in windows
    ):
        raise ValueError("Comparison windows must lie within the bundle's common source interval")
    width, height = variants[0]["width"], variants[0]["height"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "processing",
        "match_id": manifest["match_id"],
        "source": manifest["source"],
        "source_offset_seconds": offset,
        "playback_speed": speed,
        "conversion_seconds": manifest["conversion_seconds"],
        "preparation_total_seconds": manifest["total_seconds"],
        "variants": variants,
        "claims": "Visual, timing, storage and decoding cost only. No accuracy or winning rate.",
        "sampling": "30 fps normal-time playback; hold floor((source_time-offset)*speed*rate)",
        "windows": [],
    }
    json_write(output / "report.json", report)
    try:
        for number, (start, duration) in enumerate(windows):
            caps, current, frames = [], [-1] * 3, [None] * 3
            temporary = output / f"window-{number + 1}.partial.mp4"
            count = math.ceil(duration * speed * 30 - 1e-8)
            still_indices = sorted({0, count // 2, count - 1})
            window = {
                "start_seconds": start,
                "duration_seconds": duration,
                "playback_duration_seconds": duration * speed,
                "stills": [],
                "video": f"window-{number + 1}.mp4",
            }
            process = None
            try:
                for variant in variants:
                    path = (directory / variant["file"]).resolve()
                    if path.parent != directory:
                        raise ValueError("Variant path must stay inside the preparation bundle")
                    cap = cv2.VideoCapture(str(path))
                    caps.append(cap)
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open comparison input: {path}")
                with (output / f"preview-{number + 1}.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            tool("ffmpeg"),
                            "-hide_banner",
                            "-nostdin",
                            "-n",
                            "-v",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "bgr24",
                            "-video_size",
                            f"{width * 3}x{height + 64}",
                            "-framerate",
                            "30",
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-crf",
                            "18",
                            "-preset",
                            "medium",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=log,
                    )
                    for index in range(count):
                        timestamp = start + index / (30 * speed)
                        game_elapsed = (timestamp - offset) * speed
                        canvas = np.zeros((height + 64, width * 3, 3), np.uint8)
                        observations = []
                        for column, (variant, cap) in enumerate(zip(variants, caps)):
                            target = int(math.floor(game_elapsed * variant["fps"] + 1e-7))
                            if current[column] == -1:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                                current[column] = target - 1
                            while current[column] < target:
                                ok, frame = cap.read()
                                if not ok:
                                    raise RuntimeError("Unexpected end of a comparison input")
                                current[column] += 1
                                frames[column] = frame
                            x = column * width
                            canvas[64:, x : x + width] = frames[column]
                            observation = offset + target / (variant["fps"] * speed)
                            observations.append(
                                {
                                    "fps": variant["fps"],
                                    "frame_index": target,
                                    "source_grid_seconds": observation,
                                    "game_elapsed_seconds": target / variant["fps"],
                                }
                            )
                            cv2.putText(
                                canvas,
                                f"{variant['fps']:g} fps | game +{game_elapsed:.3f}s",
                                (x + 8, 23),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                canvas,
                                f"Source {timestamp:.3f}s | held {observation:.3f}s",
                                (x + 8, 48),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (200, 200, 200),
                                1,
                            )
                        process.stdin.write(canvas.tobytes())
                        if index in still_indices:
                            filename = f"window-{number + 1}-frame-{index}.png"
                            if not cv2.imwrite(str(output / filename), canvas):
                                raise RuntimeError("Failed to save comparison still")
                            window["stills"].append(
                                {
                                    "file": filename,
                                    "source_seconds": timestamp,
                                    "game_elapsed_seconds": game_elapsed,
                                    "observations": observations,
                                }
                            )
                    process.stdin.close()
                    if process.wait():
                        raise RuntimeError(f"Preview encoding failed; see {log.name}")
                info = validate_video(temporary, output / f"preview-decode-{number + 1}.log")
                if info["frames"] != count or info["fps"] != 30:
                    raise RuntimeError("Preview timing validation failed")
                temporary.rename(output / window["video"])
                window["validated"] = info
                report["windows"].append(window)
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                if process is not None and process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                for cap in caps:
                    cap.release()
        rows = "".join(
            f"<tr><td>{v['fps']:g}</td><td>{v['frames']}</td>"
            f"<td>{v['duration_seconds']:.3f}</td><td>{v['bytes'] / 1e6:.2f}</td>"
            f"<td>{v['decode_seconds']:.3f}</td></tr>"
            for v in variants
        )
        panels = "".join(
            f"<h2>Source {w['start_seconds']:g}s, {w['duration_seconds']:g}s window</h2>"
            f"<video controls preload='metadata' src='{w['video']}'></video>"
            + "".join(
                f"<figure><img src='{s['file']}'><figcaption>Source time "
                f"{s['source_seconds']:.3f}s; lower rates hold the last observation.</figcaption>"
                "</figure>"
                for s in w["stills"]
            )
            for w in report["windows"]
        )
        document = (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>StratScout frame-rate comparison</title><style>"
            "body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:12px}"
            "video,img{width:100%;height:auto}td,th{padding:8px;border:1px solid #ccc}"
            "table{border-collapse:collapse}figure{margin:20px 0}</style>"
            f"<h1>Frame-rate comparison: {html.escape(str(manifest['match_id']))}</h1>"
            f"<p>Capture playback speed: {speed:g}x. Previews play at normal game speed. "
            "Window start/duration and source labels use raw-video seconds; game + time "
            "is elapsed from the retained clip start, not the absolute demo clock.</p>"
            f"<p>{report['claims']}</p><p>Shared conversion: {manifest['conversion_seconds']:.2f}s. "
            "Decode times are complete-file wall times from validation, affected by caching and load; "
            "they exclude the detector. File sizes use decimal MB. Native crop resolution retained.</p>"
            "<table><tr><th>FPS</th><th>Frames</th><th>Duration (s)</th><th>MB</th>"
            f"<th>Decode (s)</th></tr>{rows}</table>{panels}</html>"
        )
        (output / "report.html").write_text(document, encoding="utf-8")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        json_write(output / "report.json", report)
        raise
    json_write(output / "report.json", report)
    return report
