"""Live preview: play a prepared minimap video with player and bomb detections drawn on top.

Detection runs on each displayed frame; nothing is written to disk. Seeking replays a few
seconds silently so the bomb state (which depends on history) is rebuilt correctly.
Opening the window is explicit; importing this module does not start a GUI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bomb import SPECTATOR_T_HUE, BombStateTracker, find_bomb, t_pixels_near
from .config import Config
from .context import RunContext
from .segmentation import find_icons_teams, learn_background

WINDOW = "StratScout live detection"
SPECTATOR_CT_HUE = 94.0
# BGR display colours.
# Contrasting marker colours so detections stand out from the yellow/cyan icons.
T_COLOUR, CT_COLOUR = (0, 120, 255), (255, 0, 255)
STATE_STYLE = {
    "carried": ((80, 220, 80), "CARRIED - T colour beside the badge"),
    "alone": ((0, 200, 255), "ALONE - no T nearby, waiting to see if it stays still"),
    "dropped": ((60, 60, 255), "DROPPED (candidate) - still and alone"),
    "missing": ((160, 160, 160), "Badge not detected - state unconfirmed"),
}


def spectator_context() -> RunContext:
    """Colour-baseline settings for native 494x500 spectator-radar crops: icons are
    ~220-270 px and the tight hue tolerance keeps the orange badge (~16) out of T (~27)."""
    return RunContext(Config(icon_area=(120, 450), hue_tol=8.0))


def detect(
    rgb: np.ndarray,
    near_px: int,
    t_hue: float,
    background: Optional[np.ndarray] = None,
    *,
    context,
) -> Dict:
    """One RGB frame -> player detections (pixels) and the bomb badge, if found."""
    h, w = rgb.shape[:2]
    hues = {"T": t_hue, "CT": SPECTATOR_CT_HUE}
    players = [
        (x * w, y * h, "T" if team == 0 else "CT")
        for x, y, team in find_icons_teams(rgb, hues, background, context=context)
    ]
    bomb = find_bomb(rgb, context=context)
    t_pix = (
        0
        if bomb is None
        else t_pixels_near(rgb, bomb["x"], bomb["y"], near_px, t_hue, context=context)
    )
    return {"players": players, "bomb": bomb, "t_pixels": t_pix}


def video_background(cap: cv2.VideoCapture, total: int, samples: int = 60) -> np.ndarray:
    """Median of frames spread across the match: the map without players. Subtracting it
    stops static yellow map art (the A/B site letters) being detected as T players."""
    frames = []
    for i in np.linspace(0, max(total - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return learn_background(np.stack(frames), sample=len(frames))


def draw(
    bgr: np.ndarray,
    found: Dict,
    state: str,
    tracker: BombStateTracker,
    status: List[str],
    near_px: int,
    scale: int,
) -> np.ndarray:
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for x, y, team in found["players"]:
        colour = T_COLOUR if team == "T" else CT_COLOUR
        c = (int(x * scale), int(y * scale))
        r = 11 * scale
        cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), colour, 2)
        cv2.putText(img, team, (c[0] - r, c[1] - r - 4), 0, 0.5, colour, 2)
    colour, text = STATE_STYLE[state]
    bomb = found["bomb"]
    if bomb is not None:
        bx, by = int(bomb["x"] * scale), int(bomb["y"] * scale)
        cv2.circle(img, (bx, by), near_px * scale, colour, 1)
        cv2.rectangle(
            img, (bx - 13 * scale, by - 9 * scale), (bx + 13 * scale, by + 9 * scale), colour, 2
        )
    n_t = sum(p[2] == "T" for p in found["players"])
    n_ct = len(found["players"]) - n_t
    lines = [
        (status[0], (230, 230, 230)),
        (text, colour),
        (
            f"Players detected: {n_t} T, {n_ct} CT   |   T pixels near badge: {found['t_pixels']}"
            + (
                f"   |   still {tracker.still_seconds():.1f}s"
                if state in ("alone", "dropped")
                else ""
            ),
            (230, 230, 230),
        ),
        *[(s, (170, 170, 170)) for s in status[1:]],
    ]
    panel = np.zeros((26 * len(lines) + 12, img.shape[1], 3), np.uint8)
    cv2.line(panel, (0, 1), (img.shape[1], 1), (90, 90, 90), 1)
    for k, (s, c) in enumerate(lines):
        cv2.putText(panel, s, (10, 26 + 26 * k), 0, 0.55, c, 1, cv2.LINE_AA)
    return np.vstack([img, panel])


def live_preview(
    video: Path,
    start: float = 0.0,
    near_px: int = 40,
    min_seconds: float = 2.0,
    t_hue: float = SPECTATOR_T_HUE,
    scale: int = 2,
    context: Optional[RunContext] = None,
) -> None:
    """Open an interactive window. Space play/pause, N/B next/previous frame,
    A/D back/forward 5 s, Q or Esc to quit."""
    context = context or spectator_context()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = BombStateTracker(fps, min_seconds)
    background = video_background(cap, total)

    def step(bgr: np.ndarray) -> Tuple[Dict, str]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        found = detect(rgb, near_px, t_hue, background, context=context)
        b = found["bomb"]
        state = tracker.update(
            None if b is None else b["x"], None if b is None else b["y"], found["t_pixels"]
        )
        return found, state

    def advance(index: int):
        ok, bgr = cap.read()
        if not ok:
            return None
        return (index, bgr, *step(bgr))

    def seek(frame: int):
        """Show frame, warming the tracker on the preceding few seconds of history."""
        frame = int(np.clip(frame, 0, max(total - 1, 0)))
        warm = max(frame - int((min_seconds + 1) * fps), 0)
        tracker.reset()
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm)
        for _ in range(frame - warm):
            ok, bgr = cap.read()
            if not ok:
                break
            step(bgr)
        return advance(frame)

    current = seek(int(start * fps))
    if current is None:
        raise ValueError(f"no frame at {start}s in {video}")
    playing = True
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            t0 = time.perf_counter()
            index, bgr, found, state = current
            secs = index / fps
            status = [
                f"{'PLAYING' if playing else 'PAUSED'} | frame {index} | "
                f"{int(secs) // 60:02d}:{secs % 60:05.2f} ({secs:.2f}s)",
                "Space: play/pause | N/B: next/previous frame | A/D: -/+5 seconds | Q/Esc: quit",
            ]
            cv2.imshow(WINDOW, draw(bgr, found, state, tracker, status, near_px, scale))
            wait = max(int(1000 / fps - (time.perf_counter() - t0) * 1000), 1) if playing else 0
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            target = None
            if key == ord(" "):
                playing = not playing
            elif key == ord("n"):
                playing, target = False, "next"
            elif key == ord("b"):
                playing, target = False, index - 1
            elif key == ord("a"):
                target = index - int(5 * fps)
            elif key == ord("d"):
                target = index + int(5 * fps)
            elif playing:
                target = "next"
            if target == "next":
                nxt = advance(index + 1)
                if nxt is None:
                    playing = False  # end of video: stay on the last frame
                else:
                    current = nxt
            elif target is not None:
                current = seek(target) or current
    finally:
        cap.release()
        cv2.destroyAllWindows()
