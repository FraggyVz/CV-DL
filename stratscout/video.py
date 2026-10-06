"""Streaming video I/O and reproducible run exports."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import cv2

from . import __version__
from .config import Profile
from .detector import ColourDetector, draw_detections


def open_video(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}. Supply a readable recording, not a .dem.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video has no usable frame rate; re-encode as constant-frame-rate video")
    return capture, float(fps)


def read_frame(capture, index: int):
    if index < 0:
        raise ValueError("Frame index must be non-negative")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError(f"Cannot seek to frame {index}")
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}")
    return frame


def file_identity(path: Path) -> dict:
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def _verify_export(path: Path, expected: int, resolution: tuple[int, int]) -> None:
    capture, _ = open_video(path)
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != resolution:
                raise OSError("Encoded overlay resolution differs from the requested resolution")
            decoded += 1
    finally:
        capture.release()
    if decoded != expected:
        raise OSError(f"Overlay export failed verification: decoded {decoded}/{expected} frames")


def detect_video(video: str | Path, profile: Profile, output: str | Path, *,
                 start_frame: int = 0, max_frames: int | None = None) -> dict:
    """Stream a CFR recording or preview range, retaining source frame indices."""
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    detector = ColourDetector(profile)
    video, output = Path(video), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new run directory.")
    capture, fps = open_video(video)
    writer = None
    metadata = None
    created = False
    count = 0
    try:
        advertised = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        advertised = int(advertised) if math.isfinite(advertised) and advertised > 0 else None
        if advertised is not None and start_frame >= advertised:
            raise ValueError("start_frame is outside the advertised video frame range")
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise ValueError(f"Cannot seek to start frame {start_frame}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video contains no decodable frames")
        detector.crop_frame(frame)  # Validate before creating any output.
        identity = file_identity(video)
        width, height = profile.source_resolution
        # MPEG-4 encoders can truncate odd dimensions; pad only the exported overlay.
        encoded_size = (width + width % 2, height + height % 2)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        metadata = {
            "schema_version": 1, "status": "processing",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": identity, "profile": profile.to_dict(),
            "source_fps": fps, "advertised_frame_count": advertised,
            "frame_range": {"start_frame": start_frame, "requested_max_frames": max_frames},
            "timestamp_convention": "frame_index / source_fps; constant-frame-rate input required",
            "coordinate_convention": "x right, y down; normalised to native minimap crop",
            "overlay_resolution": list(encoded_size), "overlay_codec": "mp4v",
            "overlay_has_audio": False,
            "versions": {"stratscout": __version__, "python": platform.python_version(),
                         "numpy": version("numpy"), "opencv-python": version("opencv-python")},
        }
        _write_metadata(output, metadata)
        writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, encoded_size)
        if not writer.isOpened():
            raise OSError("Cannot create annotated.mp4; check output permissions and MP4 codec support")
        with (output / "detections.jsonl").open("x", encoding="utf-8") as records:
            while ok:
                detections = detector.detect(frame)
                source_index = start_frame + count
                record = {"frame_index": source_index, "timestamp_s": source_index / fps,
                          **detections}
                records.write(json.dumps(record, allow_nan=False) + "\n")
                overlay = draw_detections(frame, profile, detections)
                if encoded_size != (width, height):
                    overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2,
                                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
                writer.write(overlay)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} frames ({count / fps:.1f}s)", flush=True)
                if max_frames is not None and count >= max_frames:
                    break
                ok, frame = capture.read()
        writer.release()
        writer = None
        capture.release()
        if advertised is not None:
            expected = advertised - start_frame
            if max_frames is not None:
                expected = min(expected, max_frames)
            if count < expected:
                raise OSError(f"Source decoding stopped early: {count}/{expected} requested frames")
        _verify_export(output / "annotated.mp4", count, encoded_size)
        current_stat = video.stat()
        if (current_stat.st_size != identity["size_bytes"]
                or current_stat.st_mtime_ns != identity["mtime_ns"]):
            raise OSError("Source video changed during processing; repeat with a stable recording")
        metadata.update(status="complete", frames_processed=count, duration_s=count / fps,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        metadata["frame_range"]["end_frame_exclusive"] = start_frame + count
        _write_metadata(output, metadata)
        return metadata
    except BaseException as exc:
        if created and metadata is not None:
            metadata.update(status="failed", frames_processed=count,
                            error=f"{type(exc).__name__}: {exc}")
            metadata["frame_range"]["end_frame_exclusive"] = start_frame + count
            try:
                _write_metadata(output, metadata)
            except OSError:
                pass  # Preserve the original I/O error when the output disk has failed.
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _write_metadata(output: Path, metadata: dict) -> None:
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
