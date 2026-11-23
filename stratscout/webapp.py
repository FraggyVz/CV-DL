"""Local web application entry point."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"
UPLOAD_DIR = Path(os.environ.get("STRATSCOUT_UPLOAD_DIR", "data/web/uploads"))
MAX_UPLOAD_BYTES = int(os.environ.get("STRATSCOUT_MAX_UPLOAD_BYTES", str(20 * 1024**3)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
UPLOAD_LOCK = threading.RLock()

app = FastAPI(title="StratScout")
app.mount(
    "/assets",
    StaticFiles(directory=str(FRONTEND_DIR / "assets"), check_dir=False),
    name="frontend-assets",
)


class SegmentDraft(BaseModel):
    id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    round_number: int = Field(ge=1)
    phase: Literal["attack", "retake"]

    @model_validator(mode="after")
    def end_after_start(self):
        if self.end_s <= self.start_s:
            raise ValueError("segment end must be after its start")
        return self


class SegmentDraftList(BaseModel):
    segments: list[SegmentDraft]

    @model_validator(mode="after")
    def unique_segment_ids(self):
        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        return self


class AnalysisRequest(BaseModel):
    playback_speed: Literal[1, 2] = 2
    prepared_clip: bool = False
    min_confidence: float = Field(default=0.6, ge=0, le=1)


class AnalysisSettings(BaseModel):
    prepared_clip: bool


def video_id_from_path(video_id: str) -> str:
    try:
        return str(uuid.UUID(video_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Recording not found") from exc


def video_path(video_id: str) -> Path:
    return UPLOAD_DIR / f"{video_id_from_path(video_id)}.mp4"


def metadata_path(video_id: str) -> Path:
    return UPLOAD_DIR / f"{video_id_from_path(video_id)}.video.json"


def segments_path(video_id: str) -> Path:
    return UPLOAD_DIR / f"{video_id_from_path(video_id)}.segments.json"


def browser_copy_path(video_id: str) -> Path:
    return UPLOAD_DIR / f"{video_id_from_path(video_id)}.browser.mp4"


def full_clip_segment(video_id: str) -> dict:
    canonical_id = video_id_from_path(video_id)
    with UPLOAD_LOCK:
        if not video_path(canonical_id).is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        metadata_file = metadata_path(canonical_id)
        metadata = (
            json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.is_file() else {}
        )
        duration = metadata.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            from .media import probe

            duration = probe(video_path(canonical_id))["duration_seconds"]
        round_number = metadata.get("round_number")
        if not isinstance(round_number, int) or round_number < 1:
            round_number = 1
        phase = metadata.get("phase")
        if phase not in ("attack", "retake"):
            phase = "attack"
        segment = {
            "id": str(uuid.uuid4()),
            "start_s": 0.0,
            "end_s": float(duration),
            "round_number": round_number,
            "phase": phase,
        }
        path = segments_path(canonical_id)
        temporary = UPLOAD_DIR / f".{canonical_id}.{uuid.uuid4().hex}.segments.tmp"
        try:
            temporary.write_text(json.dumps([segment], indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return segment


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "application": "StratScout"}


@app.get("/api/videos")
def list_videos() -> dict:
    videos = [
        json.loads(path.read_text(encoding="utf-8")) for path in UPLOAD_DIR.glob("*.video.json")
    ]
    videos.sort(key=lambda video: video["uploaded_at"], reverse=True)
    return {"videos": videos}


@app.delete("/api/videos")
def reset_videos() -> dict:
    from .web_analysis import AnalysisInProgress, reset_recordings

    with UPLOAD_LOCK:
        try:
            deleted_count = reset_recordings(UPLOAD_DIR)
        except AnalysisInProgress as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted_count": deleted_count}


@app.post("/api/videos", status_code=201)
def upload_video(
    file: UploadFile = File(...),
    match_id: str | None = Form(default=None),
    match_name: str | None = Form(default=None),
    round_number: int | None = Form(default=None, ge=1),
    phase: Literal["attack", "retake"] = Form(default="attack"),
) -> dict:
    if match_id is None:
        if match_name is not None or round_number is not None:
            raise HTTPException(
                status_code=422,
                detail="Match name, match ID and round number must be provided together.",
            )
    else:
        if round_number is None or not match_name or not match_name.strip():
            raise HTTPException(
                status_code=422,
                detail="Match name, match ID and round number must be provided together.",
            )
        try:
            match_id = str(uuid.UUID(match_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid match ID.") from exc
        match_name = match_name.strip()[:80]

    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(status_code=415, detail="Choose an MP4 recording.")
    if file.content_type not in (None, "", "video/mp4", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an MP4 recording.")
    if MAX_UPLOAD_BYTES < 1:
        raise RuntimeError("STRATSCOUT_MAX_UPLOAD_BYTES must be greater than zero")

    video_id = str(uuid.uuid4())
    temporary = UPLOAD_DIR / f"{video_id}.uploading"
    temporary_metadata = UPLOAD_DIR / f".{video_id}.video.tmp"
    temporary_segments = UPLOAD_DIR / f".{video_id}.segments.tmp"
    destination = video_path(video_id)
    total_bytes = 0
    UPLOAD_LOCK.acquire()
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as output:
            while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    limit_gib = MAX_UPLOAD_BYTES / 1024**3
                    raise HTTPException(
                        status_code=413,
                        detail=f"Recording exceeds the {limit_gib:g} GiB upload limit.",
                    )
                output.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded recording is empty.")
        os.replace(temporary, destination)
        from .media import probe

        try:
            video_info = probe(destination)
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=f"Unable to read the MP4: {exc}") from exc
        except RuntimeError as exc:
            destination.unlink(missing_ok=True)
            if str(exc).startswith("Missing ffprobe"):
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(status_code=415, detail=f"Unable to read the MP4: {exc}") from exc
        metadata = {
            "id": video_id,
            "filename": filename[:255],
            "size_bytes": total_bytes,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "content_url": f"/api/videos/{video_id}/content",
            "codec": video_info["codec"],
            "pixel_format": video_info["pixel_format"],
            "duration_seconds": video_info["duration_seconds"],
            "prepared_clip": False,
        }
        if match_id is not None:
            metadata.update(
                match_id=match_id,
                match_name=match_name,
                round_number=round_number,
                phase=phase,
            )
            segment = [
                {
                    "id": str(uuid.uuid4()),
                    "start_s": 0.0,
                    "end_s": video_info["duration_seconds"],
                    "round_number": round_number,
                    "phase": phase,
                }
            ]
            temporary_segments.write_text(json.dumps(segment, indent=2), encoding="utf-8")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata, metadata_path(video_id))
        if match_id is not None:
            os.replace(temporary_segments, segments_path(video_id))
        return metadata
    except OSError:
        destination.unlink(missing_ok=True)
        metadata_path(video_id).unlink(missing_ok=True)
        segments_path(video_id).unlink(missing_ok=True)
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            temporary_segments.unlink(missing_ok=True)
            file.file.close()
        finally:
            UPLOAD_LOCK.release()


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    canonical_id = video_id_from_path(video_id)
    from .web_analysis import AnalysisInProgress, delete_recording

    with UPLOAD_LOCK:
        try:
            return delete_recording(UPLOAD_DIR, canonical_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording not found") from exc
        except AnalysisInProgress as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.put("/api/videos/{video_id}/analysis-settings")
def save_analysis_settings(video_id: str, payload: AnalysisSettings) -> dict:
    canonical_id = video_id_from_path(video_id)
    with UPLOAD_LOCK:
        source = video_path(canonical_id)
        path = metadata_path(canonical_id)
        if not source.is_file() or not path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["prepared_clip"] = payload.prepared_clip
        temporary = UPLOAD_DIR / f".{canonical_id}.{uuid.uuid4().hex}.video.tmp"
        try:
            temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return metadata


@app.get("/api/videos/{video_id}/content", response_class=FileResponse)
def get_video_content(video_id: str) -> FileResponse:
    canonical_id = video_id_from_path(video_id)
    source = video_path(canonical_id)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Recording not found")
    metadata_file = metadata_path(canonical_id)
    metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    )
    if metadata.get("codec") == "h264" and metadata.get("pixel_format") == "yuv420p":
        path = source
    else:
        from .video_preparation import browser_compatible_video

        try:
            path = browser_compatible_video(source, browser_copy_path(canonical_id))
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=415,
                detail=f"Could not prepare a browser-compatible video: {exc}",
            ) from exc
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/videos/{video_id}/segments")
def get_video_segments(video_id: str) -> dict:
    with UPLOAD_LOCK:
        path = video_path(video_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        saved_segments = segments_path(video_id)
        segments = (
            json.loads(saved_segments.read_text(encoding="utf-8"))
            if saved_segments.exists()
            else []
        )
        if not segments:
            segments = [full_clip_segment(video_id)]
        return {"video_id": video_id_from_path(video_id), "segments": segments}


@app.put("/api/videos/{video_id}/segments")
def save_video_segments(video_id: str, payload: SegmentDraftList) -> dict:
    canonical_id = video_id_from_path(video_id)
    if not video_path(canonical_id).is_file():
        raise HTTPException(status_code=404, detail="Recording not found")
    from .web_analysis import AnalysisInProgress, invalidate_analysis

    try:
        invalidate_analysis(UPLOAD_DIR, canonical_id)
    except AnalysisInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    path = segments_path(canonical_id)
    temporary = UPLOAD_DIR / f".{canonical_id}.{uuid.uuid4().hex}.segments.tmp"
    data = [segment.model_dump() for segment in payload.segments]
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"video_id": canonical_id, "segments": data}


@app.post("/api/videos/{video_id}/analysis", status_code=202)
def start_video_analysis(
    video_id: str,
    payload: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    canonical_id = video_id_from_path(video_id)
    if not video_path(canonical_id).is_file():
        raise HTTPException(status_code=404, detail="Recording not found")
    path = segments_path(canonical_id)
    if not path.is_file():
        full_clip_segment(canonical_id)
    segments = SegmentDraftList(segments=json.loads(path.read_text(encoding="utf-8"))).segments
    if not segments:
        segments = [SegmentDraft.model_validate(full_clip_segment(canonical_id))]

    from .web_analysis import AnalysisInProgress, queue_analysis, run_analysis

    try:
        state = queue_analysis(UPLOAD_DIR, canonical_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Recording not found") from exc
    except AnalysisInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(
        run_analysis,
        UPLOAD_DIR,
        canonical_id,
        state["analysis_id"],
        [segment.model_dump() for segment in segments],
        playback_speed=payload.playback_speed,
        prepared_clip=payload.prepared_clip,
        min_confidence=payload.min_confidence,
    )
    return state


@app.get("/api/videos/{video_id}/analysis")
def get_video_analysis(video_id: str) -> dict:
    canonical_id = video_id_from_path(video_id)
    if not video_path(canonical_id).is_file():
        raise HTTPException(status_code=404, detail="Recording not found")
    from .web_analysis import get_analysis

    return get_analysis(UPLOAD_DIR, canonical_id)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend assets are missing. Rebuild the Docker image or run the Vite dev server.",
        )
    return FileResponse(index)
