"""Live preview: play a prepared minimap video with player and bomb detections drawn on top.

Detection runs on each displayed frame; nothing is written to disk. Seeking replays a few
seconds silently so the bomb state (which depends on history) is rebuilt correctly.
Rounds are estimated from spawn occupancy (see rounds.py) and cached under runs/.
Opening the window is explicit; importing this module does not start a GUI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bomb import SPECTATOR_T_HUE, BombStateTracker, carrier_rim, carrier_visible, find_bomb
from .config import Config
from .context import RunContext
from .rounds import cached_rounds, round_at
from .rounds import video_background as bgr_background
from .segmentation import find_icons_teams

WINDOW = "StratScout live detection"
# cv2.waitKeyEx codes for the arrow keys on Windows.
LEFT_ARROW, RIGHT_ARROW = 2424832, 2555904
SPECTATOR_CT_HUE = 94.0
# BGR display colours.
# Contrasting marker colours so detections stand out from the yellow/cyan icons.
T_COLOUR, CT_COLOUR = (0, 120, 255), (255, 0, 255)
STATE_STYLE = {
    "carried": ((80, 220, 80), "CARRIED - carrier's rim/arrow visible at the badge edge"),
    "alone": ((0, 200, 255), "ALONE - no carrier at the badge, waiting to see if it stays still"),
    "dropped": ((60, 60, 255), "DROPPED (candidate) - still and alone"),
    "missing": ((160, 160, 160), "Badge not detected - state unconfirmed"),
    "rewinding": ((160, 160, 160), "Rewinding - bomb state resumes when playback does"),
}
SPEEDS = (2, 4, 8, 16)


def spectator_context() -> RunContext:
    """Colour-baseline settings for native 494x500 spectator-radar crops: icons are
    ~220-270 px and the tight hue tolerance keeps the orange badge (~16) out of T (~27)."""
    return RunContext(Config(icon_area=(120, 450), hue_tol=8.0))


def detect(
    rgb: np.ndarray,
    band: int,
    t_hue: float,
    background: Optional[np.ndarray] = None,
    *,
    context,
) -> Dict:
    """One RGB frame -> player detections (pixels), the bomb badge and whether its carrier
    shows at the badge edge. A carried badge is drawn over its carrier, so the carrier is
    added at the badge centre as an inferred T unless the colour baseline already found it."""
    h, w = rgb.shape[:2]
    hues = {"T": t_hue, "CT": SPECTATOR_CT_HUE}
    players = [
        (x * w, y * h, "T" if team == 0 else "CT")
        for x, y, team in find_icons_teams(rgb, hues, background, context=context)
    ]
    bomb = find_bomb(rgb, context=context)
    rim = {"yellow": 0, "white": 0}
    carrier = None
    if bomb is not None:
        rim = carrier_rim(rgb, bomb, band, t_hue, context=context)
        near = [
            p for p in players if p[2] == "T" and np.hypot(p[0] - bomb["x"], p[1] - bomb["y"]) < 10
        ]
        if carrier_visible(rim) and not near:
            carrier = (bomb["x"], bomb["y"])
    return {"players": players, "bomb": bomb, "rim": rim, "carrier": carrier}


def list_matches(root: Path = Path("data/processed/recordings")) -> List[Path]:
    """Prepared match folders (those containing minimap-*fps.mp4 variants)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("minimap-*fps.mp4")))


def match_video(folder: Path, fps: int) -> Path:
    """The folder's variant at fps, or its highest-rate variant if that one is missing."""
    want = Path(folder) / f"minimap-{fps}fps.mp4"
    if want.exists():
        return want
    variants = sorted(Path(folder).glob("minimap-*fps.mp4"), key=lambda v: int(v.stem[8:-3]))
    if not variants:
        raise ValueError(f"no minimap-*fps.mp4 in {folder}")
    return variants[-1]


def choose_match(matches: List[Path], name: Optional[str] = None) -> Path:
    """Pick a match by name fragment, or ask in the terminal when name is None."""
    if not matches:
        raise ValueError("no prepared matches found under data/processed/recordings")
    if name is not None:
        hits = [m for m in matches if name.lower() in m.name.lower()]
        if len(hits) != 1:
            raise ValueError(f"--match {name!r} matched {len(hits)} folders; be more specific")
        return hits[0]
    for k, m in enumerate(matches, 1):
        print(f"  {k}. {m.name}")
    while True:
        answer = input(f"Choose a match [1-{len(matches)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(matches):
            return matches[int(answer) - 1]
        print("  Enter one of the listed numbers.")


def round_label(r: Dict, n_rounds: int) -> str:
    extra = ", started before recording" if r["partial"] else ""
    return f"Round {r['round']}/{n_rounds} (estimated{extra})"


def draw(
    bgr: np.ndarray,
    found: Dict,
    state: str,
    tracker: BombStateTracker,
    status: List[str],
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
    if found["carrier"] is not None:
        c = (int(found["carrier"][0] * scale), int(found["carrier"][1] * scale))
        r = 15 * scale
        cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), T_COLOUR, 2)
        cv2.putText(img, "T carrier (under badge)", (c[0] - r, c[1] - r - 4), 0, 0.5, T_COLOUR, 2)
    if bomb is not None:
        bx, by = int(bomb["x"] * scale), int(bomb["y"] * scale)
        cv2.rectangle(
            img, (bx - 13 * scale, by - 9 * scale), (bx + 13 * scale, by + 9 * scale), colour, 2
        )
    n_t = sum(p[2] == "T" for p in found["players"])
    n_ct = len(found["players"]) - n_t
    carrier = " + 1 carrier under badge" if found["carrier"] is not None else ""
    rim = found["rim"]
    lines = [
        (status[0], (230, 230, 230)),
        (text, colour),
        (
            f"Players: {n_t} T{carrier}, {n_ct} CT   |   badge rim: {rim['yellow']} yellow, "
            f"{rim['white']} white px"
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
    band: int = 4,
    min_seconds: float = 2.0,
    t_hue: float = SPECTATOR_T_HUE,
    scale: int = 2,
    round_number: Optional[int] = None,
    rounds_video: Optional[Path] = None,
    context: Optional[RunContext] = None,
) -> None:
    """Open an interactive window. Space play/pause (back to normal speed), Right/Left arrows
    fast-forward/rewind (press again for 2x, 4x, 8x, 16x), K/L back/forward 5 s, N/B
    next/previous frame, [ and ] previous/next round, Q or Esc to quit.
    Rounds are estimated on rounds_video (a faster variant of the same match) if given."""
    context = context or spectator_context()
    print("Estimating rounds from spawn occupancy (cached after the first run)...")
    rounds = cached_rounds(rounds_video or video)
    print(f"{len(rounds)} rounds estimated.")
    if round_number is not None:
        if not 1 <= round_number <= len(rounds):
            raise ValueError(f"--round must be 1-{len(rounds)} for this match")
        start = rounds[round_number - 1]["start"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = BombStateTracker(fps, min_seconds)
    background = cv2.cvtColor(bgr_background(cap, total), cv2.COLOR_BGR2RGB)

    def step(bgr: np.ndarray, frames: int = 1) -> Tuple[Dict, str]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        found = detect(rgb, band, t_hue, background, context=context)
        b = found["bomb"]
        state = tracker.update(
            None if b is None else b["x"],
            None if b is None else b["y"],
            b is not None and carrier_visible(found["rim"]),
            frames=frames,
        )
        return found, state

    def advance(index: int, frames: int = 1):
        """Move forward by frames, running detection only on the frame shown; the tracker
        counts the skipped frames so still times stay in real seconds."""
        for _ in range(frames - 1):
            if not cap.grab():
                return None
        ok, bgr = cap.read()
        if not ok:
            return None
        return (index, bgr, *step(bgr, frames))

    def rewind_to(frame: int):
        """Show an earlier frame quickly; the history-dependent bomb state is not rebuilt."""
        frame = int(np.clip(frame, 0, max(total - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (frame, bgr, detect(rgb, band, t_hue, background, context=context), "rewinding")

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
    speed = 1  # 0 paused, 1 normal, >1 fast-forward, <0 rewind
    # Resizable: the frame is scaled to the window, keeping its aspect ratio.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    sized = False
    try:
        while True:
            t0 = time.perf_counter()
            index, bgr, found, state = current
            secs = index / fps
            rnd = round_at(rounds, secs)
            into = secs - rnd["start"]
            mode = {0: "PAUSED", 1: "PLAYING"}.get(
                speed, f"FAST-FORWARD {speed}x" if speed > 1 else f"REWIND {-speed}x"
            )
            status = [
                f"{mode} | {round_label(rnd, len(rounds))} | "
                f"round time {int(into) // 60}:{int(into) % 60:02d} | "
                f"match {int(secs) // 60:02d}:{secs % 60:05.2f} | frame {index}",
                "Space: play/pause | Right/Left arrows: fast-forward/rewind (repeat for faster)",
                "K/L: -/+5 s | N/B: next/previous frame | [ / ]: previous/next round | Q/Esc: quit",
            ]
            img = draw(bgr, found, state, tracker, status, scale)
            if not sized:
                cv2.resizeWindow(WINDOW, img.shape[1], img.shape[0])
                sized = True
            cv2.imshow(WINDOW, img)
            wait = max(int(1000 / fps - (time.perf_counter() - t0) * 1000), 1) if speed else 0
            raw = cv2.waitKeyEx(wait)
            key = -1 if raw == -1 else raw & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            was_rewinding = speed < 0
            target = None
            if raw == RIGHT_ARROW:
                speed = next((v for v in SPEEDS if v > speed), SPEEDS[-1]) if speed > 1 else 2
            elif raw == LEFT_ARROW:
                faster = next((v for v in SPEEDS if v > -speed), SPEEDS[-1])
                speed = -faster if speed < -1 else -2
            elif key == ord(" "):
                speed = 0 if speed == 1 else 1
            elif key == ord("n"):
                speed, target = 0, "next"
            elif key == ord("b"):
                speed, target = 0, index - 1
            elif key == ord("k"):
                target = index - int(5 * fps)
            elif key == ord("l"):
                target = index + int(5 * fps)
            elif key == ord("]"):
                later = [r for r in rounds if r["start"] > secs + 0.5]
                target = int(later[0]["start"] * fps) if later else None
            elif key == ord("["):
                # Back to this round's start, or to the previous round if already there.
                k = rnd["round"] - 1 - (1 if into < 3 and rnd["round"] > 1 else 0)
                target = int(rounds[k]["start"] * fps)
            if target is None and speed < 0:
                if index == 0:
                    speed = 0
                else:
                    current = rewind_to(index + speed) or current
                continue
            if target is None and was_rewinding:
                target = index  # leaving rewind: rebuild the bomb state at this frame
            if target is None and speed > 0:
                target = "next"
            if target == "next":
                step_frames = max(speed, 1)
                nxt = advance(index + step_frames, step_frames)
                if nxt is None:
                    speed = 0  # end of video: stay on the last frame
                else:
                    current = nxt
            elif target is not None:
                current = seek(target) or current
    finally:
        cap.release()
        cv2.destroyAllWindows()
