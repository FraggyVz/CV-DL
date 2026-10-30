"""Keyboard labeller for tactical segments (docs/tactic-definitions.md).

Plays a prepared minimap video. The reviewer marks a segment's start and end, picks one of
the five classes (or undecided / excluded) and presses S; the segment is saved to disk
at once. Rounds are estimated from spawn occupancy (rounds.py) and are navigation aids only.
Opening the window is explicit; importing this module does not start a GUI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import TACTIC_CLASSES
from .live_preview import LEFT_ARROW, RIGHT_ARROW
from .rounds import round_at
from .segments import KEY_EXCLUSION_REASONS, SITE_OF, SegmentFile

WINDOW = "StratScout segment labeller"
ENTER, ESC, BACKSPACE = 13, 27, 8
STEP_SECONDS = 2.0  # Left/Right arrow jump
HELP = [
    "Space play/pause | Left/Right -/+2 s | , . one frame | [ ] previous/next round start",
    "I start  O end (press again to change) | 0 undecided | U boundary uncertain",
    "1 A_DIRECT 2 B_DIRECT 3 A_SPLIT 4 B_SPLIT 5 RETAKE (+ A/B site)",
    "E exclude (repeat: next reason)",
    "S save | X clear the draft | N/P next/previous saved segment | C round checked | Q quit",
    "Fix a saved one: D D inside it (or Z Z for the last saved), then I, O, class, S again",
    "M/J bookmark",
]
STATUS_COLOUR = {"pending": (80, 200, 255), "excluded": (150, 150, 150), "reviewed": (80, 220, 80)}


def clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    return f"{int(seconds) // 60:02d}:{seconds % 60:04.1f}"


def wrap(text: str, fits) -> List[str]:
    """Split a status line at ' | ' (then ', ') breaks until every piece fits."""
    if fits(text):
        return [text]
    for sep in (" | ", ", "):
        parts = text.split(sep)
        if len(parts) > 1:
            out, cur = [], parts[0]
            for p in parts[1:]:
                if fits(cur + sep + p):
                    cur += sep + p
                else:
                    out.append(cur)
                    cur = p
            return [piece for line in out + [cur] for piece in wrap(line, fits)]
    return [text]


def text_panel(lines: List[tuple], width: int, font: float) -> np.ndarray:
    """Draw (text, colour) lines at their real pixel size, wrapping to the width."""
    fits = lambda s: cv2.getTextSize(s, 0, font, 1)[0][0] <= width - 16  # noqa: E731
    rows = [(piece, colour) for text, colour in lines for piece in wrap(text, fits)]
    step = int(round(46 * font))
    panel = np.zeros((step * len(rows) + 12, width, 3), np.uint8)
    for k, (text, colour) in enumerate(rows):
        cv2.putText(panel, text, (8, step * (k + 1)), 0, font, colour, 1, cv2.LINE_AA)
    return panel


def to_height(img: np.ndarray, height: int) -> np.ndarray:
    """Crop or pad (with black at the bottom) to exactly height rows."""
    pad = np.zeros((max(height - len(img), 0), img.shape[1], 3), np.uint8)
    return np.vstack([img, pad])[:height]


def fit_into(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """The frame scaled to fit width x height, centred on black, keeping its ratio."""
    fit = max(min(width / bgr.shape[1], height / bgr.shape[0]), 0.01)
    small = cv2.resize(
        bgr,
        (max(int(bgr.shape[1] * fit), 1), max(int(bgr.shape[0] * fit), 1)),
        interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), np.uint8)
    y0, x0 = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return canvas


def use_real_pixels() -> None:
    """Stop Windows display scaling (e.g. 150%) from stretching the window, which blurs
    text and can push the window past the screen edge. No effect on other systems."""
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def screen_work_area() -> tuple:
    """(left, top, width, height) of the screen area not covered by the taskbar.

    Uses the Windows API when available; otherwise assumes a 1280x720 screen."""
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        # SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except (AttributeError, OSError):
        pass
    return 0, 0, 1280, 720


def play_start(r: Dict) -> float:
    """When players can move: the end of the estimated freeze time."""
    return r["start"] + r["freeze"]


class SegmentLabeller:
    """Labelling keys applied at a video time; the window loop lives in run()."""

    def __init__(self, store: SegmentFile, rounds: List[Dict]):
        self.store, self.rounds = store, rounds
        self.message, self.ok = "Press I at the start of an attack or retake.", True
        self.armed = None  # (key, segment id) waiting for the second Z/D press
        self.clear()

    def clear(self):
        self.draft: Dict = {"start_s": None, "end_s": None, "label": None, "site": None}
        self.undecided, self.reason, self.uncertain = False, None, False

    def round_of(self, seconds: float) -> Dict:
        return round_at(self.rounds, seconds)

    def record(self, seconds: float) -> Dict:
        """The draft as a segment record, its round taken from the start time."""
        r = self.round_of(self.draft["start_s"] if self.draft["start_s"] is not None else seconds)
        seg = {
            **self.draft,
            "status": "excluded" if self.reason else "pending",
            "estimated_round": r["round"],
            "round_play_start_s": round(play_start(r), 2),
            "boundary_uncertain": self.uncertain,
        }
        if self.reason:
            seg["exclusion_reason"] = self.reason
            seg["proposed_label"], seg["label"] = seg["label"], None
        elif self.undecided:
            seg["undecided"] = True
        return seg

    def say(self, text: str, ok: bool = True):
        self.message, self.ok = text, ok

    def press(self, key: str, seconds: float) -> Optional[float]:
        """Apply one key at video time seconds; returns a time to jump to, if any."""
        d = self.draft
        armed, self.armed = self.armed, None  # any other key cancels a pending delete
        if key == "i":
            d["start_s"] = seconds
            self.say(f"Start {clock(seconds)}. Now mark the end with O.")
        elif key == "o":
            d["end_s"] = seconds
            self.say(f"End {clock(seconds)}. Choose a class, then S to save.")
        elif key in "12345" and len(key) == 1:
            d["label"], self.undecided, self.reason = TACTIC_CLASSES[int(key) - 1], False, None
            d["site"] = SITE_OF.get(d["label"], d["site"] if d["label"] == "RETAKE" else None)
            self.say(f"Class {d['label']}." + (" Press A or B for the site." * (key == "5")))
        elif key == "0":
            d["label"], self.undecided, self.reason = None, True, None
            self.say("Undecided: saved as pending for another reviewer.")
        elif key in ("a", "b"):
            if d["label"] in SITE_OF:
                self.say(f"{d['label']} already means site {SITE_OF[d['label']]}.", False)
            else:
                d["site"] = key.upper()
                self.say(f"Site {d['site']}.")
        elif key == "e":
            reasons = KEY_EXCLUSION_REASONS
            k = reasons.index(self.reason) + 1 if self.reason in reasons else 0
            self.reason = reasons[k % len(reasons)]
            self.undecided = False
            self.say(f"Exclude: {self.reason} (E again for the next reason, X to clear).")
        elif key == "u":
            self.uncertain = not self.uncertain
            self.say("Boundary marked uncertain." if self.uncertain else "Boundary certain.")
        elif key == "x":
            self.clear()
            self.say("Draft cleared.")
        elif key in ("s", "enter"):  # S saves; Enter is kept as an alias
            try:
                saved = self.store.add(self.record(seconds))
            except ValueError as exc:
                self.say(f"Not saved: {exc}", False)
            else:
                what = saved["label"] or saved.get("exclusion_reason") or "undecided"
                self.say(
                    f"Saved #{saved['id']} {what} {clock(saved['start_s'])}-"
                    f"{clock(saved['end_s'])} to {self.store.path.name}"
                )
                self.clear()
        elif key in ("z", "d"):
            if key == "z":
                # The last saved segment.
                seg = max(self.store.segments, key=lambda s: s["id"], default=None)
                if seg is None:
                    self.say("Nothing to delete.", False)
                    return None
            else:
                # The saved segment being watched: the latest one started at or before now.
                here = [s for s in self.store.segments if s["start_s"] <= seconds <= s["end_s"]]
                if not here:
                    self.say("D deletes the saved segment you are inside (marked >).", False)
                    return None
                seg = max(here, key=lambda s: s["start_s"])
            what = seg["label"] or seg.get("exclusion_reason") or "undecided"
            span = f"#{seg['id']} {what} {clock(seg['start_s'])}-{clock(seg['end_s'])}"
            if armed != (key, seg["id"]):
                # Deleting saved work needs a second press of the same key.
                self.armed = (key, seg["id"])
                self.say(f"Press {key.upper()} again to delete {span} (any other key cancels).")
                return None
            try:
                self.store.remove(seg["id"])
            except ValueError as exc:
                self.say(str(exc), False)
            else:
                self.say(f"Deleted {span}. Label it again and press S.")
        elif key == "c":
            r = self.round_of(seconds)
            done = self.store.toggle_checked(r["round"])
            self.say(f"Round {r['round']} marked {'checked' if done else 'not checked'}.")
        elif key == "m":
            self.store.add_bookmark(seconds)
            self.say(f"Bookmark at {clock(seconds)}.")
        elif key == "j":
            marks = self.store.data["bookmarks"]
            if not marks:
                self.say("No bookmarks yet (M adds one).", False)
                return None
            return next((m for m in marks if m > seconds + 0.5), marks[0])
        elif key in ("n", "p"):
            saved = sorted(self.store.segments, key=lambda s: s["start_s"])
            if not saved:
                self.say("No saved segments yet.", False)
                return None
            if key == "n":
                seg = next((s for s in saved if s["start_s"] > seconds + 0.5), saved[0])
            else:
                seg = next((s for s in saved[::-1] if s["start_s"] < seconds - 0.5), saved[-1])
            what = seg["label"] or seg.get("exclusion_reason") or "undecided"
            self.say(
                f"Segment #{seg['id']} {what} {clock(seg['start_s'])}-{clock(seg['end_s'])} "
                f"({seg['status']}). Space to watch it."
            )
            return seg["start_s"]
        elif key == "]":
            later = [play_start(r) for r in self.rounds if play_start(r) > seconds + 0.5]
            return later[0] if later else None
        elif key == "[":
            r = self.round_of(seconds)
            if seconds - play_start(r) < 3 and r["round"] > 1:
                r = self.rounds[r["round"] - 2]
            return play_start(r)
        return None

    def status_lines(self, seconds: float, mode: str) -> List[tuple]:
        r = self.round_of(seconds)
        into = seconds - play_start(r)
        checked = r["round"] in self.store.data["checked_rounds"]
        partial = ", started before recording" if r["partial"] else ""
        d = self.draft
        what = (
            self.reason
            and f"EXCLUDE ({self.reason})"
            or d["label"]
            or ("undecided" if self.undecided else "-")
        )
        here = [s for s in self.store.segments if s.get("estimated_round") == r["round"]]
        # ">" marks the saved segment being watched now.
        done = ", ".join(
            f"{'>' if s['start_s'] <= seconds <= s['end_s'] else ''}#{s['id']} "
            f"{s['label'] or s['status']} {clock(s['start_s'])}-{clock(s['end_s'])}"
            for s in here
        )
        total = self.store.summary()
        white, grey = (230, 230, 230), (170, 170, 170)
        return [
            (
                f"{self.store.data['match_folder']}  |  reviewer {self.store.reviewer}  |  {mode}",
                white,
            ),
            (
                f"Match time {clock(seconds)}  |  Round {r['round']}/{len(self.rounds)} "
                f"(estimated{partial})  |  round time {'-' if into < 0 else clock(into)}  |  "
                f"{'CHECKED' if checked else 'not checked'}",
                white,
            ),
            (
                f"Draft: start {clock(d['start_s'])}  end {clock(d['end_s'])}  class {what}  "
                f"site {d['site'] or '-'}{'  UNCERTAIN' if self.uncertain else ''}",
                (80, 200, 255),
            ),
            (f"This round: {done or 'no segments yet'}", white),
            (
                f"Match: {len(self.store.segments)} segments "
                + " ".join(f"{k} {v}" for k, v in total.items() if v)
                + f"  |  rounds checked {len(self.store.data['checked_rounds'])}/{len(self.rounds)}",
                grey,
            ),
            (self.message, (120, 230, 120) if self.ok else (90, 90, 255)),
            *[(h, grey) for h in HELP],
        ]

    def timeline(self, width: int, duration: float, seconds: float) -> np.ndarray:
        """A bar for the whole match: round starts, saved segments, the draft and the playhead."""
        bar = np.full((22, width, 3), 40, np.uint8)
        x = lambda s: int(np.clip(s / max(duration, 1e-6), 0, 1) * (width - 1))  # noqa: E731
        for r in self.rounds:
            cv2.line(bar, (x(play_start(r)), 0), (x(play_start(r)), 21), (110, 110, 110), 1)
        for s in self.store.segments:
            colour = STATUS_COLOUR.get(s["status"], (200, 200, 200))
            cv2.rectangle(
                bar, (x(s["start_s"]), 5), (max(x(s["end_s"]), x(s["start_s"]) + 1), 16), colour, -1
            )
        if self.draft["start_s"] is not None:
            end = self.draft["end_s"] if self.draft["end_s"] is not None else seconds
            cv2.rectangle(bar, (x(self.draft["start_s"]), 2), (x(end), 19), (255, 255, 255), 1)
        cv2.line(bar, (x(seconds), 0), (x(seconds), 21), (255, 255, 255), 2)
        return bar

    def compose(
        self,
        bgr: np.ndarray,
        seconds: float,
        mode: str,
        duration: float,
        scale: int = 2,
        size: Optional[tuple] = None,
    ) -> np.ndarray:
        """The window image: the frame, the match timeline and the status text.

        With size=(width, height) the image is built at exactly the window's size, so the
        window never rescales it and the text stays sharp; long lines wrap."""
        lines = self.status_lines(seconds, mode)
        fh, fw = bgr.shape[:2]
        width, height = size or (fw * scale, fh * scale + 360)
        bar = self.timeline(width, duration, seconds)
        text = text_panel(lines, width, float(np.clip(height / 1600, 0.45, 0.6)))
        frame = fit_into(bgr, width, max(height - len(bar) - len(text), 50))
        return to_height(np.vstack([frame, bar, text]), height)

    def run(self, video: Path, start: float = 0.0, scale: int = 2) -> None:
        """Open the window. All saving happens on S / Z / C / M, so Q or closing the
        window only loses an unsaved draft."""
        use_real_pixels()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        index, speed, shown, frame = int(np.clip(start * fps, 0, total - 1)), 0, None, None
        # Free ratio: compose() fills the window exactly and keeps the frame's own ratio.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        sized = False
        try:
            while True:
                t0 = time.perf_counter()
                if index != shown:
                    if shown is None or index != shown + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, bgr = cap.read()
                    if ok:
                        frame, shown = bgr, index
                    else:
                        index, speed = shown if shown is not None else 0, 0
                if frame is None:
                    raise ValueError(f"no frame at {start}s in {video}")
                secs = index / fps
                mode = "PLAYING" if speed else "PAUSED"
                if not sized:
                    # Fit the visible screen above the taskbar; the window stays resizable.
                    left, top, room_w, room_h = screen_work_area()
                    cv2.resizeWindow(WINDOW, min(room_w - 20, 1100), max(room_h - 70, 300))
                    cv2.moveWindow(WINDOW, left + 5, top + 5)
                    sized = True
                # Draw at the window's own size so the text is never stretched or blurred.
                _, _, w, h = cv2.getWindowImageRect(WINDOW)
                size = (w, h) if w > 200 and h > 200 else None
                cv2.imshow(WINDOW, self.compose(frame, secs, mode, duration, scale, size))
                wait = max(int(1000 / fps - (time.perf_counter() - t0) * 1000), 1) if speed else 0
                raw = cv2.waitKeyEx(wait)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if raw == -1:
                    if speed:
                        index = int(np.clip(index + speed, 0, total - 1))
                        if index in (0, total - 1):
                            speed = 0
                    continue
                key = raw & 0xFF
                if raw in (RIGHT_ARROW, LEFT_ARROW):
                    # Step and pause, so the reviewer can mark the exact moment.
                    step = int(round(STEP_SECONDS * fps))
                    speed, index = 0, index + (step if raw == RIGHT_ARROW else -step)
                elif key in (ord("q"), ESC):
                    break
                elif key == ord(" "):
                    speed = 0 if speed == 1 else 1
                elif key in (ord(","), ord(".")):
                    speed, index = 0, index + (1 if key == ord(".") else -1)
                else:
                    name = (
                        "enter" if key == ENTER else "x" if key == BACKSPACE else chr(key).lower()
                    )
                    target = self.press(name, secs)
                    if target is not None:
                        index = int(round(target * fps))
                index = int(np.clip(index, 0, total - 1))
        finally:
            cap.release()
            cv2.destroyAllWindows()
