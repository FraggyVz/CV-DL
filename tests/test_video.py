import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
from helpers import profile, video
from stratscout.__main__ import main
from stratscout.config import save_profile
from stratscout.video import detect_video


class VideoTests(unittest.TestCase):
    def test_public_cli_streams_and_exports_timestamped_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / "input.avi", root / "profile.json", root / "run"
            video(source)
            save_profile(profile(), config)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["detect", "--video", str(source), "--profile", str(config),
                             "--output", str(output)])
            self.assertEqual(code, 0)
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["frames_processed"], 8)
            self.assertEqual(metadata["duration_s"], 1)
            self.assertEqual(len(metadata["input"]["sha256"]), 64)
            records = [json.loads(line) for line in (output / "detections.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame_index"] for r in records], list(range(8)))
            self.assertEqual([r["timestamp_s"] for r in records], [i / 8 for i in range(8)])
            for row in records:
                self.assertEqual(len(row["T"]), 1)
                self.assertEqual(len(row["CT"]), 1)
            self.assertAlmostEqual(records[0]["T"][0]["x"], .15, delta=.01)
            self.assertAlmostEqual(records[-1]["T"][0]["x"], .185, delta=.01)
            cap = cv2.VideoCapture(str(output / "annotated.mp4"))
            ok, overlay = cap.read()
            self.assertTrue(ok)
            self.assertEqual(overlay.shape, (240, 320, 3))
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 8)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 8)
            cap.release()
            # The text/rectangle changes pixels far from the original dots.
            self.assertGreater(int(overlay[:25].max()), 100)
            before = (output / "detections.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                detect_video(source, profile(), output)
            self.assertEqual((output / "detections.jsonl").read_bytes(), before)

    def test_invalid_video_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.mp4"
            source.write_bytes(b"not a video")
            with self.assertRaises(ValueError):
                detect_video(source, profile(), root / "out")
            self.assertFalse((root / "out").exists())

    def test_resolution_mismatch_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            video(source)
            from dataclasses import replace
            wrong = replace(profile(), source_resolution=(640, 480))
            with self.assertRaisesRegex(ValueError, "Recalibrate"):
                detect_video(source, wrong, root / "out")
            self.assertFalse((root / "out").exists())

    def test_writer_failure_is_reported_and_marked_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            video(source)
            with patch("stratscout.video.cv2.VideoWriter") as writer:
                writer.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(OSError, "Cannot create"):
                    detect_video(source, profile(), root / "out")
            metadata = json.loads((root / "out" / "run.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            writer.return_value.release.assert_called()

    def test_failed_export_verification_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            video(source)
            with patch("stratscout.video._verify_export", side_effect=OSError("Short write")):
                with self.assertRaisesRegex(OSError, "Short write"):
                    detect_video(source, profile(), root / "out")
            self.assertEqual(json.loads((root / "out" / "run.json").read_text())["status"], "failed")

    def test_cli_error_has_no_traceback(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = main(["detect", "--video", "missing.mp4", "--profile", "missing.json", "--output", "unused"])
        self.assertEqual(result, 2)
        self.assertIn("Error:", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
