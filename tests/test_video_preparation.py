"""Real FFmpeg integration checks using numbered, lossless synthetic frames."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from stratscout.fps_comparison import compare_fps, parse_window
from stratscout.media import run_ffmpeg, tool
from stratscout.video_preparation import prepare_video, sampling_graph
from stratscout.video_profiles import load_profile, validate_profile


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def number(frame):
    return sum((int(frame[16, 8 + bit * 20].mean() > 128) << bit) for bit in range(7))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class VideoPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        # NUT preserves exact 1/60 timestamps; Matroska millisecond quantisation
        # would legitimately change which source frame falls on the sampling grid.
        cls.video = cls.root / "numbered.nut"
        process = subprocess.Popen(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                "192x128",
                "-framerate",
                "60",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                str(cls.video),
            ],
            stdin=subprocess.PIPE,
        )
        for index in range(120):
            frame = np.full((128, 192, 3), 60, np.uint8)
            crop = frame[8:120, 16:176]
            crop[:] = 0
            for bit in range(7):
                crop[8:25, bit * 20 : bit * 20 + 17] = 255 if index & (1 << bit) else 0
            cv2.putText(crop, str(index), (4, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            crop[72:104, :48] = (0, 0, 255)
            crop[72:104, 56:104] = (0, 255, 0)
            crop[72:104, 112:160] = (255, 0, 0)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        assert process.wait() == 0
        cls.profile = cls.root / "crop.json"
        cls.profile.write_text(
            json.dumps(
                {"schema_version": 1, "source_resolution": [192, 128], "crop": [16, 8, 176, 120]}
            )
        )
        cls.output = cls.root / "bundle"
        cls.manifest = prepare_video(cls.video, cls.profile, cls.output, match_id="same-match")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_numbered_sampling_alignment_colour_crop_duration_and_timestamps(self):
        sampled = {}
        for variant in self.manifest["variants"]:
            fps = int(variant["fps"])
            frames = read_frames(self.output / variant["file"])
            self.assertEqual(len(frames), fps * 2)
            self.assertEqual(frames[0].shape, (112, 160, 3))
            self.assertEqual(variant["duration_seconds"], 2)
            sampled[fps] = [number(f) for f in frames]
            self.assertEqual(sampled[fps], list(range(0, 120, 60 // fps)))
            # OpenCV is BGR; RGB users must explicitly convert. Colours must survive the crop.
            rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            for x, expected in ((24, (255, 0, 0)), (80, (0, 255, 0)), (136, (0, 0, 255))):
                np.testing.assert_allclose(rgb[88, x], expected, atol=8)
            payload = json.loads(
                subprocess.check_output(
                    [
                        tool("ffprobe"),
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "frame=best_effort_timestamp_time",
                        "-of",
                        "json",
                        str(self.output / variant["file"]),
                    ],
                    text=True,
                )
            )
            np.testing.assert_allclose(
                [float(f["best_effort_timestamp_time"]) for f in payload["frames"]],
                np.arange(fps * 2) / fps,
                atol=1e-6,
            )
        self.assertEqual(sampled[5], sampled[30][::6])
        self.assertEqual(sampled[10], sampled[30][::3])
        graph = sampling_graph([16, 8, 176, 120], [5, 10, 30])
        self.assertEqual(graph.count("crop="), 1)

    def test_offsets_and_fractional_duration(self):
        output = self.root / "offset"
        manifest = prepare_video(self.video, self.profile, output, start=0.2, duration=1.05)
        self.assertEqual(manifest["source_offset_seconds"], 0.2)
        for v in manifest["variants"]:
            self.assertLessEqual(abs(v["duration_seconds"] - 1.05), 1 / v["fps"])
        frames = read_frames(output / "minimap-30fps.mp4")
        self.assertEqual(number(frames[0]), 12)

    def test_matroska_timestamp_quantisation_keeps_30fps_observations(self):
        # OBS MKV timestamps may be rounded to milliseconds (0, 33, 67 ms).
        # Always rounding upwards would drop/repeat observations at an unchanged rate.
        video = self.root / "obs-like.mkv"
        subprocess.run(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(self.video),
                "-vf",
                "select='not(mod(n,2))',setpts=N/(30*TB)",
                "-r",
                "30",
                "-c:v",
                "ffv1",
                str(video),
            ],
            check=True,
        )
        output = self.root / "obs-like-prepared"
        prepare_video(video, self.profile, output)
        for fps in (5, 10, 30):
            self.assertEqual(
                [number(f) for f in read_frames(output / f"minimap-{fps}fps.mp4")],
                list(range(0, 120, 60 // fps)),
            )

    def test_comparison_holds_observations_and_outputs_stills(self):
        report = compare_fps(self.output, ["0:1"])
        self.assertEqual(report["status"], "complete")
        self.assertTrue((self.output / "comparison/report.html").is_file())
        frames = read_frames(self.output / "comparison/window-1.mp4")
        self.assertEqual(len(frames), 30)
        for index, frame in enumerate(frames):
            for column, fps in enumerate((5, 10, 30)):
                expected = (index // (30 // fps)) * (60 // fps)
                self.assertEqual(number(frame[64:, column * 160 : (column + 1) * 160]), expected)
        self.assertEqual(len(report["windows"][0]["stills"]), 3)
        with self.assertRaises(FileExistsError):
            compare_fps(self.output, ["0:1"])
        with self.assertRaises(ValueError):
            compare_fps(self.output, ["2:1"], output=self.root / "bad-windows")

    def test_refuses_invalid_inputs_and_overwrites(self):
        with self.assertRaises(FileExistsError):
            prepare_video(self.video, self.profile, self.output)
        for rates in ([], [15], [5, 5], [True]):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                prepare_video(self.video, self.profile, self.root / "bad", rates=rates)
        with self.assertRaises(ValueError):
            prepare_video(self.output / "minimap-5fps.mp4", self.profile, self.root / "too-low")
        with self.assertRaises(ValueError):
            prepare_video(self.video, self.profile, self.root / "outside", start=3)
        corrupt = self.root / "broken.mp4"
        corrupt.write_bytes(b"not a video")
        with self.assertRaises(RuntimeError):
            prepare_video(corrupt, self.profile, self.root / "corrupt")
        with (
            patch("stratscout.media.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Missing"),
        ):
            prepare_video(self.video, self.profile, self.root / "missing")

    def test_interrupted_encoding_keeps_source_and_incomplete_manifest(self):
        before = self.video.read_bytes()
        output = self.root / "interrupted"
        with (
            patch("stratscout.video_preparation.run_ffmpeg", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            prepare_video(self.video, self.profile, output)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "incomplete")
        self.assertEqual(self.video.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            prepare_video(self.video, self.profile, output)

    def test_encoding_failure_preserves_diagnostic_bundle(self):
        output = self.root / "failed-encode"
        with patch(
            "stratscout.video_preparation.run_ffmpeg", side_effect=RuntimeError("encoder failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                prepare_video(self.video, self.profile, output)
        saved = json.loads((output / "manifest.json").read_text())
        self.assertEqual(saved["status"], "incomplete")
        self.assertIn("encoder failed", saved["error"])
        self.assertTrue(self.video.is_file())

    def test_process_is_killed_on_interrupt(self):
        process = Mock()
        process.wait.side_effect = [KeyboardInterrupt, 1]
        with patch("stratscout.media.subprocess.Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                run_ffmpeg([], self.root / "cancel.log")
        process.kill.assert_called_once()


class ProfileTests(unittest.TestCase):
    def test_invalid_profiles(self):
        good = {"schema_version": 1, "source_resolution": [192, 128], "crop": [0, 0, 160, 112]}
        for crop in (
            [0, 0, 161, 112],
            [-2, 0, 160, 112],
            [0, 0, 300, 112],
            [160, 0, 0, 112],
            [False, 0, 160, 112],
            [0, 0, 1],
            None,
        ):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                validate_profile({**good, "crop": crop})
        with self.assertRaises(ValueError):
            validate_profile(good, (2560, 1440))
        with self.assertRaises(ValueError):
            validate_profile({})
        actual = load_profile(
            Path(__file__).resolve().parents[1] / "configs/falcons-furia-video.json"
        )
        self.assertEqual(actual["crop"], [16, 12, 508, 510])

    def test_window_syntax(self):
        self.assertEqual(parse_window("55:10"), (55, 10))
        for value in ("bad", "0:0", "-1:2", "nan:3", "2:inf"):
            with self.assertRaises(ValueError):
                parse_window(value)
