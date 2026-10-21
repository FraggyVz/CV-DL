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

from .bomb import SPECTATOR_T_HUE, BombStateTracker, find_bomb, t_pixels_near
from .config import Config
from .context import RunContext
from .rounds import cached_rounds, round_at
from .rounds import video_background as bgr_background
from .segmentation import find_icons_teams

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
    round_number: Optional[int] = None,
    rounds_video: Optional[Path] = None,
    context: Optional[RunContext] = None,
) -> None:
    """Open an interactive window. Space play/pause, N/B next/previous frame,
    A/D back/forward 5 s, [ and ] previous/next round, Q or Esc to quit.
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
            status = [
                f"{'PLAYING' if playing else 'PAUSED'} | {round_label(rnd, len(rounds))} | "
                f"round time {int(into) // 60}:{int(into) % 60:02d} | "
                f"match {int(secs) // 60:02d}:{secs % 60:05.2f} | frame {index}",
                "Space: play/pause | N/B: next/previous frame | A/D: -/+5 seconds",
                "[ / ]: previous/next round | Q/Esc: quit",
            ]
            img = draw(bgr, found, state, tracker, status, near_px, scale)
            if not sized:
                cv2.resizeWindow(WINDOW, img.shape[1], img.shape[0])
                sized = True
            cv2.imshow(WINDOW, img)
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
            elif key == ord("]"):
                later = [r for r in rounds if r["start"] > secs + 0.5]
                target = int(later[0]["start"] * fps) if later else None
            elif key == ord("["):
                # Back to this round's start, or to the previous round if already there.
                k = rnd["round"] - 1 - (1 if into < 3 and rnd["round"] > 1 else 0)
                target = int(rounds[k]["start"] * fps)
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
