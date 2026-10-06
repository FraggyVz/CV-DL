from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from helpers import profile, video
from stratscout.calibration import calibrate
from stratscout.config import load_profile


class CalibrationTests(unittest.TestCase):
    def run_ui(self, source, destination, events):
        """Exercise the real calibration loop with simulated window inputs, without a desktop."""
        sliders, callbacks, images = {}, {}, []
        def create(label, window, value, maximum, callback):
            sliders[label] = value
        def update(label, window, value):
            sliders[label] = value
        def wait(delay):
            action = next(events)
            if isinstance(action, tuple):
                team, point, key = action
                self.assertEqual(sliders["Team 0=T 1=CT"], team)
                callbacks["click"](cv2.EVENT_LBUTTONDOWN, *point, None, None)
                return key
            return action
        with ExitStack() as stack:
            for method in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                stack.enter_context(patch(f"stratscout.calibration.cv2.{method}"))
            stack.enter_context(patch("stratscout.calibration.select_crop", return_value=profile().crop))
            stack.enter_context(patch("stratscout.calibration.cv2.createTrackbar", side_effect=create))
            stack.enter_context(patch("stratscout.calibration.cv2.setTrackbarPos", side_effect=update))
            stack.enter_context(patch("stratscout.calibration.cv2.getTrackbarPos", side_effect=lambda label, _: sliders[label]))
            stack.enter_context(patch("stratscout.calibration.cv2.setMouseCallback",
                                     side_effect=lambda _, callback: callbacks.update(click=callback)))
            stack.enter_context(patch("stratscout.calibration.cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("stratscout.calibration.cv2.waitKey", side_effect=wait))
            stack.enter_context(patch("stratscout.calibration.cv2.imshow", side_effect=lambda _, image: images.append(image.copy())))
            stack.enter_context(redirect_stdout(io.StringIO()))
            calibrate(source, destination)
        return images

    def test_sample_both_teams_seek_and_save_reusable_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "input.avi", root / "profile.json"
            video(source)
            images = self.run_ui(source, destination, iter([
                (0, (120, 200), ord("c")),  # T at crop coordinate (30,50), enlarged x4
                (1, (520, 440), ord("n")),  # CT at (130,110)
                ord("s"),
            ]))
            saved = load_profile(destination)
            self.assertEqual(saved.crop, profile().crop)
            self.assertAlmostEqual(saved.teams["T"].hue, 30, delta=2)
            self.assertAlmostEqual(saved.teams["CT"].hue, 105, delta=2)
            self.assertEqual(len(images), 6)  # Three preview/controls iterations.

    def test_unsampled_defaults_cannot_be_saved_and_cancel_preserves_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "input.avi", root / "profile.json"
            video(source)
            self.run_ui(source, destination, iter([ord("s"), ord("q")]))
            self.assertFalse(destination.exists())

    def test_existing_profile_rejected_before_opening_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("preserve")
            with self.assertRaises(FileExistsError):
                calibrate("absent.mp4", path)
            self.assertEqual(path.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
