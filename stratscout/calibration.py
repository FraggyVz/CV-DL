"""Desktop calibration with explicit team colour sampling and live previews."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import Profile, TeamSettings, save_profile
from .detector import ColourDetector, colour_mask, draw_detections
from .video import open_video, read_frame

CONTROLS = "StratScout controls"
PREVIEW = "StratScout minimap - click an icon to sample the selected team"
SLIDERS = {
    "Hue": ("hue", 179, 1),
    "Hue tolerance": ("hue_tolerance", 89, 1),
    "Saturation min": ("saturation_min", 255, 1),
    "Value min": ("value_min", 255, 1),
    "Area min": ("area_min", 2000, 1),
    "Area max": ("area_max", 5000, 1),
    "Aspect x10": ("aspect_max", 100, 10),
    "Fill percent": ("fill_min", 100, 100),
}


def sample_hue(crop: np.ndarray, x: int, y: int) -> int:
    """Circular hue mean from saturated pixels around a BGR icon centre."""
    if not (0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]):
        raise ValueError("Click inside the minimap image")
    patch = crop[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    usable = hsv[(hsv[:, 1] >= 60) & (hsv[:, 2] >= 50)]
    if not len(usable):
        raise ValueError("No saturated colour here; click the coloured core of an icon")
    angles = usable[:, 0].astype(float) * (2 * np.pi / 180)
    weights = usable[:, 1].astype(float) ** 2
    angle = np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights))
    return round(angle * 180 / (2 * np.pi)) % 180


def select_crop(frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    shown = cv2.resize(frame, (round(width * scale), round(height * scale)))
    name = "Select full minimap; ENTER accepts, C cancels"
    x, y, w, h = cv2.selectROI(name, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(name)
    if not w or not h:
        raise ValueError("Crop selection cancelled")
    sx, sy = width / shown.shape[1], height / shown.shape[0]
    return round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)


def calibrate(video: str | Path, destination: str | Path, frame_index: int = 0) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Profile already exists: {destination}. Choose a new filename.")
    capture, fps = open_video(video)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or not 0 <= frame_index < total:
            raise ValueError("Calibration needs a seekable video and a valid --frame index")
        frame = read_frame(capture, frame_index)
        crop_box = select_crop(frame)
        profile = Profile((frame.shape[1], frame.shape[0]), crop_box,
                          {"T": TeamSettings(30), "CT": TeamSettings(105)})
        active = "T"
        sampled = set()
        message = "Select T or CT, then click its icon. Both must be sampled before saving."
        x0, y0, x1, y1 = crop_box
        crop_width, crop_height = x1 - x0, y1 - y0
        scale = min(4.0, 1000 / crop_width, 650 / crop_height)
        display_size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
        cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS, 620, 600)
        cv2.namedWindow(PREVIEW, cv2.WINDOW_AUTOSIZE)
        noop = lambda value: None
        cv2.createTrackbar("Frame", CONTROLS, frame_index, max(1, total - 1), noop)
        cv2.createTrackbar("Team 0=T 1=CT", CONTROLS, 0, 1, noop)
        for label, (attribute, maximum, multiplier) in SLIDERS.items():
            value = round(getattr(profile.teams[active], attribute) * multiplier)
            cv2.createTrackbar(label, CONTROLS, value, maximum, noop)

        def show_settings(team):
            for label, (attribute, _, multiplier) in SLIDERS.items():
                cv2.setTrackbarPos(label, CONTROLS,
                                  round(getattr(profile.teams[team], attribute) * multiplier))

        def click(event, x, y, flags, parameter):
            nonlocal message
            if event != cv2.EVENT_LBUTTONDOWN or y >= display_size[1]:
                return
            try:
                cx = int(x * crop_width / display_size[0])
                cy = int(y * crop_height / display_size[1])
                hue = sample_hue(frame[y0:y1, x0:x1], cx, cy)
                profile.teams[active] = replace(profile.teams[active], hue=hue)
                sampled.add(active)
                cv2.setTrackbarPos("Hue", CONTROLS, hue)
                message = f"Sampled {active}: hue {hue}. Adjust thresholds and inspect other frames."
            except ValueError as exc:
                message = str(exc)

        cv2.setMouseCallback(PREVIEW, click)
        while True:
            selected = "CT" if cv2.getTrackbarPos("Team 0=T 1=CT", CONTROLS) else "T"
            if selected != active:
                active = selected
                show_settings(active)
            values = {}
            for label, (attribute, _, multiplier) in SLIDERS.items():
                value = cv2.getTrackbarPos(label, CONTROLS)
                values[attribute] = value if multiplier == 1 else value / multiplier
            profile.teams[active] = TeamSettings(**values)
            selected_frame = min(total - 1, cv2.getTrackbarPos("Frame", CONTROLS))
            if selected_frame != frame_index:
                frame = read_frame(capture, selected_frame)
                frame_index = selected_frame
            error = None
            try:
                detector = ColourDetector(profile)
                detections = detector.detect(frame)
                overlay = draw_detections(frame, profile, detections)[y0:y1, x0:x1]
            except ValueError as exc:
                error = str(exc)
                overlay = frame[y0:y1, x0:x1]
            preview = cv2.resize(overlay, display_size, interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            mask = colour_mask(hsv, profile.teams[active]) * 255
            # A small mask preview makes false blobs and merged icons visible.
            mask = cv2.resize(mask, (240, 160), interpolation=cv2.INTER_NEAREST)
            panel = np.zeros((285, 620, 3), np.uint8)
            panel[120:280, :240] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            lines = [f"Team: {active} | frame {frame_index}/{total - 1} | {frame_index / fps:.2f}s",
                     "T/C: team | N/B: next/previous frame | S: save | Q/ESC: cancel",
                     "Both team colours must be sampled. Sliders apply to selected team.",
                     (error or message)[:90], f"Sampled: {', '.join(sorted(sampled)) or 'none'}"]
            for index, line in enumerate(lines):
                cv2.putText(panel, line, (4, 18 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 150, 255) if error else (230, 230, 230), 1)
            cv2.imshow(CONTROLS, panel)
            cv2.imshow(PREVIEW, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                print("Calibration cancelled; no profile written.")
                return
            if key in (ord("t"), ord("c")):
                cv2.setTrackbarPos("Team 0=T 1=CT", CONTROLS, int(key == ord("c")))
            if key in (ord("n"), ord("b")):
                step = 1 if key == ord("n") else -1
                cv2.setTrackbarPos("Frame", CONTROLS, max(0, min(total - 1, frame_index + step)))
            if key == ord("s"):
                if sampled != {"T", "CT"}:
                    message = "Click at least one real icon for each team before saving."
                elif error:
                    message = error
                else:
                    save_profile(profile, destination)
                    print(f"Saved calibration profile: {destination}")
                    return
            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                   for name in (CONTROLS, PREVIEW)):
                print("Calibration cancelled; no profile written.")
                return
    finally:
        capture.release()
        cv2.destroyAllWindows()
