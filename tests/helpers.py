from pathlib import Path

import cv2
import numpy as np

from stratscout.config import Profile, TeamSettings


def profile():
    return Profile((320, 240), (80, 60, 280, 220),
                   {"T": TeamSettings(30), "CT": TeamSettings(105)})


def bgr(hue):
    return tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def frame(index=0):
    image = np.full((240, 320, 3), 25, np.uint8)
    cv2.circle(image, (110 + index, 110), 4, bgr(30), -1)
    cv2.circle(image, (210, 170 - index), 4, bgr(105), -1)
    # Same team colour outside the crop must never be detected.
    cv2.circle(image, (25, 100), 4, bgr(30), -1)
    return image


def video(path: Path, count=8, fps=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("The test environment cannot encode the MJPG fixture")
    try:
        for index in range(count):
            writer.write(frame(index))
    finally:
        writer.release()
