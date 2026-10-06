"""Native-resolution HSV component detector; inputs are OpenCV BGR uint8 frames."""

import cv2
import numpy as np

from .config import Profile, TEAMS, TeamSettings

DISPLAY_COLOURS = {"T": (0, 220, 255), "CT": (255, 190, 0)}


def colour_mask(hsv: np.ndarray, settings: TeamSettings) -> np.ndarray:
    distance = np.abs(hsv[:, :, 0].astype(np.int16) - settings.hue)
    distance = np.minimum(distance, 180 - distance)
    return ((distance <= settings.hue_tolerance)
            & (hsv[:, :, 1] >= settings.saturation_min)
            & (hsv[:, :, 2] >= settings.value_min)).astype(np.uint8)


class ColourDetector:
    def __init__(self, profile: Profile):
        profile.validate()
        self.profile = profile

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        width, height = self.profile.source_resolution
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError(
                f"Expected BGR uint8 frame {(height, width, 3)}; "
                f"got {frame.shape} {frame.dtype}. Recalibrate for this source."
            )
        x0, y0, x1, y1 = self.profile.crop
        return frame[y0:y1, x0:x1]

    def detect(self, frame: np.ndarray) -> dict[str, list[dict[str, float]]]:
        crop = self.crop_frame(frame)
        height, width = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        result = {team: [] for team in TEAMS}
        for team in TEAMS:
            settings = self.profile.teams[team]
            # No closing/dilation: these can merge nearby players into one component.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                colour_mask(hsv, settings), connectivity=8
            )
            for index in range(1, count):
                left, top, bw, bh, area = map(int, stats[index])
                if not settings.area_min <= area <= settings.area_max:
                    continue
                if max(bw, bh) / min(bw, bh) > settings.aspect_max:
                    continue
                if area / (bw * bh) < settings.fill_min:
                    continue
                ys, xs = np.nonzero(labels[top:top + bh, left:left + bw] == index)
                xs, ys = xs + left, ys + top
                weights = hsv[ys, xs, 1].astype(np.float64) ** 2
                result[team].append({
                    "x": float(np.average(xs, weights=weights) / width),
                    "y": float(np.average(ys, weights=weights) / height),
                })
            result[team].sort(key=lambda point: (point["x"], point["y"]))
        return result


def draw_detections(frame: np.ndarray, profile: Profile, detections: dict) -> np.ndarray:
    """Annotate a copy of a full source frame; never alter the detector input."""
    output = frame.copy()
    x0, y0, x1, y1 = profile.crop
    cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (100, 255, 100), 1)
    for team in TEAMS:
        colour = DISPLAY_COLOURS[team]
        for point in detections[team]:
            x = min(x1 - 1, x0 + round(point["x"] * (x1 - x0)))
            y = min(y1 - 1, y0 + round(point["y"] * (y1 - y0)))
            cv2.circle(output, (x, y), 5, colour, 1, cv2.LINE_AA)
            cv2.putText(output, team, (max(0, x - 5), max(10, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
    caption = f"Visible detections: T={len(detections['T'])} CT={len(detections['CT'])}"
    cv2.putText(output, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output
