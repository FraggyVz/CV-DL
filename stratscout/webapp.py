"""Local web application entry point."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app = FastAPI(title="StratScout")
app.mount(
    "/assets",
    StaticFiles(directory=str(FRONTEND_DIR / "assets"), check_dir=False),
    name="frontend-assets",
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "application": "StratScout"}


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend assets are missing. Rebuild the Docker image or run the Vite dev server.",
        )
    return FileResponse(index)
