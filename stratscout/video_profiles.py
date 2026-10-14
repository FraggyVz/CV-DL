"""Half-open, native-resolution minimap crops, independent of detector settings."""

import json
from pathlib import Path

from .media import probe


def validate_profile(profile, resolution=None):
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Crop profile requires schema_version: 1")
    size, crop = profile.get("source_resolution"), profile.get("crop")
    for value, count, name in ((size, 2, "source_resolution"), (crop, 4, "crop")):
        if (
            not isinstance(value, list)
            or len(value) != count
            or any(type(v) is not int for v in value)
        ):
            raise ValueError(f"{name} must contain {count} integers")
    width, height = size
    x0, y0, x1, y1 = crop
    if width <= 0 or height <= 0 or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Invalid crop: expected half-open coordinates inside source resolution")
    if (x1 - x0) % 2 or (y1 - y0) % 2:
        raise ValueError("Crop width and height must be even for yuv420p; do not resize icons")
    if resolution is not None and list(resolution) != size:
        raise ValueError(f"Profile resolution {size} does not match video {list(resolution)}")
    return profile


def load_profile(path, resolution=None):
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")), resolution)


def even_selection(roi, resolution):
    """Adjust interactive edges by one pixel, favouring expansion over clipping."""
    x, y, width, height = roi
    frame_width, frame_height = resolution
    if width <= 0 or height <= 0:
        raise RuntimeError("Crop selection cancelled or empty; no profile saved")
    if not (0 <= x < x + width <= frame_width and 0 <= y < y + height <= frame_height):
        raise ValueError("Selected crop is outside the video")
    crop = [x, y, x + width, y + height]
    for low, high, limit in ((0, 2, frame_width), (1, 3, frame_height)):
        if (crop[high] - crop[low]) % 2:
            if crop[high] < limit:
                crop[high] += 1
            elif crop[low] > 0:
                crop[low] -= 1
            else:
                crop[high] -= 1  # Full odd-sized source: no room to expand.
    validate_profile({"schema_version": 1, "source_resolution": list(resolution), "crop": crop})
    return crop


def create_profile(video, output, *, crop=None, at=0):
    import cv2

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Profile already exists: {output}")
    info = probe(video)
    if not 0 <= at < info["duration_seconds"]:
        raise ValueError("Preview timestamp is outside the video")
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode the selected preview frame")
        if crop is None:
            cv2.namedWindow("Select minimap", cv2.WINDOW_NORMAL)
            x, y, w, h = cv2.selectROI("Select minimap", frame, showCrosshair=True)
            cv2.destroyWindow("Select minimap")
            crop = even_selection((x, y, w, h), (info["width"], info["height"]))
            if crop != [x, y, x + w, y + h]:
                print(
                    "Adjusted crop edges by one pixel for encoding; image pixels are not resized."
                )
            profile = validate_profile(
                {
                    "schema_version": 1,
                    "source_resolution": [info["width"], info["height"]],
                    "crop": list(crop),
                }
            )
            x0, y0, x1, y1 = crop
            print(f"Preview {x1 - x0}x{y1 - y0}: press Enter in the preview to save this crop.")
            cv2.imshow("Native crop: Enter saves; Escape cancels", frame[y0:y1, x0:x1])
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (10, 13):
                    break
                if key == 27:
                    raise RuntimeError("Crop selection cancelled; no profile saved")
        else:
            profile = validate_profile(
                {
                    "schema_version": 1,
                    "source_resolution": [info["width"], info["height"]],
                    "crop": list(crop),
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as target:
            json.dump(profile, target, indent=2)
            target.write("\n")
        print(f"Reusable crop saved: {output.resolve()}")
        return profile
    finally:
        cap.release()
        cv2.destroyAllWindows()
