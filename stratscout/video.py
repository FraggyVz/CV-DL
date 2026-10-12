"""Video: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def list_videos(data_dir: Optional[str] = None, *, context) -> List[Path]:
    """Every readable replay in the data folder, sorted so the order is stable between runs.

    Drop-in friendly: hidden / temp files are ignored, and a file OpenCV cannot open (still
    copying, truncated render, wrong codec) is skipped with a warning instead of crashing later.
    """
    d = Path(data_dir or context.config.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    found = sorted(
        (
            p
            for p in d.iterdir()
            if p.is_file()
            and p.suffix.lower() in context.config.video_exts
            and (not p.name.startswith((".", "~")))
        )
    )
    if cv2 is None:
        return found
    ok = []
    for p in found:
        cap = cv2.VideoCapture(str(p))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
        cap.release()
        if n > 0:
            ok.append(p)
        else:
            print(f"WARNING: skipping {p.name} — OpenCV cannot read it (still copying? corrupt?)")
    return ok


def match_id_of(video) -> str:
    """data/faze_vs_navi_m1.mp4 -> 'faze_vs_navi_m1'. One file = one match."""
    return Path(video).stem


def sample_probe_frames(
    video: str, n: int = 20, skip_head: float = 0.05, skip_tail: float = 0.05
) -> np.ndarray:
    """Grab `n` RGB frames spread across the video (skipping intro/outro)."""
    assert cv2 is not None, "pip install opencv-python"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    idx = np.linspace(total * skip_head, total * (1 - skip_tail), n).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < 4:
        raise RuntimeError("could not read enough frames — is the file a valid video?")
    return np.stack(out)


def video_resolution(video: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    wh = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return wh


def _stability_and_detail(frames_rgb: np.ndarray):
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]).astype(np.float32)
    med = np.median(g, axis=0)
    stable = (np.abs(g - med[None]) <= 3.0).mean(axis=0)
    gx = cv2.Sobel(med.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    detail = cv2.blur(np.sqrt(gx * gx + gy * gy), (21, 21))
    return (stable, detail, med)


def find_radar_box(
    frames: np.ndarray,
    search_frac: float = 0.45,
    stable_min: float = 0.97,
    detail_min: float = 6.0,
    pad: int = 2,
    square: bool = True,
    edge_stable_min: float = 0.9,
) -> Tuple[int, int, int, int]:
    """Locate the radar panel in the top-left. Returns (x0, y0, x1, y1) in source pixels."""
    H, W = frames.shape[1:3]
    stable, detail, _ = _stability_and_detail(frames)
    sh, sw = (int(H * search_frac), int(W * search_frac))
    m = ((stable[:sh, :sw] >= stable_min) & (detail[:sh, :sw] >= detail_min)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        raise RuntimeError("no stable, detailed region found — set CFG.crop_box by hand")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.float32)
    rows, cols = (comp.sum(1), comp.sum(0))
    ry = np.where(rows >= 0.35 * rows.max())[0]
    rx = np.where(cols >= 0.35 * cols.max())[0]
    x0, y0, x1, y1 = (int(rx[0]), int(ry[0]), int(rx[-1]) + 1, int(ry[-1]) + 1)
    x0, y0 = (max(x0 - pad, 0), max(y0 - pad, 0))
    x1, y1 = (min(x1 + pad, W), min(y1 + pad, H))
    while x1 - x0 > 16 and stable[y0:y1, x0].mean() < edge_stable_min:
        x0 += 1
    while x1 - x0 > 16 and stable[y0:y1, x1 - 1].mean() < edge_stable_min:
        x1 -= 1
    while y1 - y0 > 16 and stable[y0, x0:x1].mean() < edge_stable_min:
        y0 += 1
    while y1 - y0 > 16 and stable[y1 - 1, x0:x1].mean() < edge_stable_min:
        y1 -= 1
    if square:
        s = min(x1 - x0, y1 - y0)
        cx, cy = ((x0 + x1) // 2, (y0 + y1) // 2)
        x0, y0 = (cx - s // 2, cy - s // 2)
        x1, y1 = (x0 + s, y0 + s)
    return (int(x0), int(y0), int(x1), int(y1))


def crop_change(frames: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Typical fraction of crop pixels that differ from the median crop. Players and the bomb ring
    keep this at a few %; a radar that drifts, rotates or blends with the world pushes it to ~100%.
    Median over frames, so one menu / match-result frame cannot spoil it."""
    x0, y0, x1, y1 = box
    c = frames[:, y0:y1, x0:x1].astype(np.int16)
    med = np.median(c, axis=0)
    return float(np.median((np.abs(c - med).sum(-1) > 30).mean(axis=(1, 2))))


def radar_hidden(frames: np.ndarray, max_changed: float = 0.3) -> np.ndarray:
    """bool (N,) — True where the radar is covered (match-result screen, menus, pauses).
    Compares each cropped frame to the clip's median, which is the bare map."""
    c = frames.astype(np.int16)
    med = np.median(c, axis=0)
    return (np.abs(c - med).sum(-1) > 30).mean(axis=(1, 2)) > max_changed


def detect_crops(
    videos: Sequence[Path], cache_json: Optional[str] = None, *, context
) -> Dict[str, Tuple[int, int, int, int]]:
    """Crop box per video, keyed by file name. Cached, because probing a multi-GB file is slow.

    A cache entry is reused only if the video's resolution and file size still match — replace a
    file with a new render under the same name and its box is detected again, not silently reused.
    Renaming a video keeps its size, so a renamed file finds its old entry; entries for videos no
    longer in data/ are dropped.
    A video whose radar cannot be found is left out of the result (and so out of the dataset).
    """
    cache_path = Path(cache_json or context.config.crops_json)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    crops = {}
    for v in videos:
        res, size = (list(video_resolution(v)), v.stat().st_size)
        hit = cache.get(v.name) or next(
            (e for e in cache.values() if e.get("bytes") == size and e["resolution"] == res),
            None,
        )
        if (
            hit
            and hit["resolution"] == res
            and (hit.get("bytes") == size)
            and (
                context.config.crop_box is None or hit["crop_box"] == list(context.config.crop_box)
            )
        ):
            crops[v.name] = tuple(hit["crop_box"])
            cache[v.name] = hit
            continue
        print(f"   probing {v.name} ...")
        try:
            probe_v = sample_probe_frames(str(v), context.config.probe_frames)
            box = (
                tuple(context.config.crop_box)
                if context.config.crop_box is not None
                else find_radar_box(probe_v, context.config.search_frac)
            )
        except RuntimeError as e:
            print(f"WARNING: {v.name} left out — {e}")
            continue
        crops[v.name] = box
        cache[v.name] = {
            "resolution": res,
            "bytes": size,
            "crop_box": list(box),
            "change_pct": round(crop_change(probe_v, box) * 100, 1),
        }
    cache = {n: cache[n] for n in crops}
    cache_path.write_text(json.dumps(cache, indent=1))
    print(f"\n{'video':<40} {'crop_box':<22} {'size':>9} {'change':>7}")
    for name, box in crops.items():
        ch = cache[name].get("change_pct", float("nan"))
        flag = "" if ch <= 10 else "   <-- check capture settings (CLAUDE.md §4)"
        print(
            f"{name:<40} {str(box):<22} {box[2] - box[0]:>4}x{box[3] - box[1]:<4} {ch:>6.1f}%{flag}"
        )
    resolutions = {tuple(cache[n]["resolution"]) for n in crops}
    if len(resolutions) > 1:
        print(
            f"WARNING: videos were recorded at different resolutions {sorted(resolutions)}. Each gets its own crop box, but icon size relative to the map may differ — re-render at one resolution if detection quality varies between matches."
        )
    return crops


def extract_clip(
    video: str,
    start_s: float,
    end_s: float,
    box: Tuple[int, int, int, int],
    n_frames: int = 32,
    out_px: Optional[int] = None,
) -> np.ndarray:
    """-> uint8 (n_frames, H, W, 3) of cropped radar."""
    assert cv2 is not None, "pip install opencv-python"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    x0, y0, x1, y1 = box
    last = min(end_s * fps, total - 2)
    idx = np.linspace(start_s * fps, last, n_frames).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        if out_px:
            rgb = cv2.resize(rgb, (out_px, out_px), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"got {len(frames)}/{n_frames} frames — check start/end times")
    return np.stack(frames).astype(np.uint8)


def video_duration(video: str) -> float:
    cap = cv2.VideoCapture(video)
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 60.0)
    cap.release()
    return float(d)


def stack_resized(clips: Sequence[np.ndarray]) -> np.ndarray:
    """Stack clips from different videos. Crop boxes can differ by a pixel or two between
    renders, so anything not matching the first clip's size is resized to it (INTER_AREA)."""
    h, w = clips[0].shape[-3:-1]

    def fit(c):
        if c.shape[-3:-1] == (h, w):
            return c
        flat = c.reshape(-1, *c.shape[-3:])
        out = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in flat])
        return out.reshape(*c.shape[:-3], h, w, 3)

    return np.concatenate([fit(c) for c in clips]).astype(np.uint8)
