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
from stratscout.video_profiles import create_profile, even_selection, load_profile, validate_profile


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

    def test_double_speed_preserves_all_30fps_game_observations(self):
        output = self.root / "double-speed"
        manifest = prepare_video(self.video, self.profile, output, playback_speed=2)
        self.assertEqual(manifest["playback_speed"], 2)
        # NUT reports the final PTS, not the end of that frame's display interval.
        self.assertEqual(
            manifest["requested_duration_seconds"], manifest["source_video"]["duration_seconds"]
        )
        self.assertEqual(
            manifest["expected_output_duration_seconds"], manifest["requested_duration_seconds"] * 2
        )
        for fps in (5, 10, 30):
            frames = read_frames(output / f"minimap-{fps}fps.mp4")
            self.assertEqual(len(frames), fps * 4)
            self.assertEqual([number(f) for f in frames], list(range(0, 120, 30 // fps)))
        for variant in manifest["variants"]:
            self.assertEqual(variant["duration_seconds"], 4)

    def test_double_speed_trim_and_comparison_use_source_seconds(self):
        output = self.root / "double-trim"
        manifest = prepare_video(
            self.video, self.profile, output, start=0.2, duration=1, playback_speed=2
        )
        self.assertEqual(manifest["expected_output_duration_seconds"], 2)
        self.assertEqual(number(read_frames(output / "minimap-30fps.mp4")[0]), 12)
        report = compare_fps(output, ["0.4:0.5"])
        self.assertEqual(report["playback_speed"], 2)
        frames = read_frames(output / "comparison/window-1.mp4")
        self.assertEqual(len(frames), 30)
        self.assertEqual(report["windows"][0]["playback_duration_seconds"], 1)
        for index, frame in enumerate(frames):
            for column, fps in enumerate((5, 10, 30)):
                expected = 24 + (index // (30 // fps)) * (30 // fps)
                self.assertEqual(number(frame[64:, column * 160 : (column + 1) * 160]), expected)
        still = report["windows"][0]["stills"][-1]
        self.assertAlmostEqual(still["source_seconds"], 0.4 + 29 / 60)
        self.assertAlmostEqual(still["game_elapsed_seconds"], 0.4 + 29 / 30)
        with self.assertRaises(ValueError):
            compare_fps(output, ["1:0.5"], output=self.root / "outside-double-trim")

    def test_invalid_speed_and_insufficient_effective_rate(self):
        for speed in (0, -1, float("nan"), float("inf"), True, "2"):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                prepare_video(
                    self.video, self.profile, self.root / "bad-speed", playback_speed=speed
                )
        with self.assertRaisesRegex(ValueError, "game time"):
            prepare_video(
                self.output / "minimap-30fps.mp4",
                self.profile,
                self.root / "too-slow-at-double",
                playback_speed=2,
            )

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
    def test_interactive_edges_expand_without_leaving_source(self):
        cases = [
            ((2, 4, 5, 7), (20, 20), [2, 4, 8, 12]),
            ((15, 13, 5, 7), (20, 20), [14, 12, 20, 20]),
            ((2, 4, 6, 8), (20, 20), [2, 4, 8, 12]),
            ((0, 0, 19, 17), (19, 17), [0, 0, 18, 16]),
        ]
        for roi, size, expected in cases:
            with self.subTest(roi=roi, size=size):
                self.assertEqual(even_selection(roi, size), expected)
        with self.assertRaises(RuntimeError):
            even_selection((0, 0, 0, 0), (20, 20))
        with self.assertRaises(ValueError):
            even_selection((18, 0, 5, 4), (20, 20))

    def test_odd_selection_previews_native_pixels_and_saves_only_after_confirmation(self):
        frame = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
        for key in (13, 27):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "crop.json"
                cap = Mock()
                cap.read.return_value = (True, frame)
                with (
                    patch(
                        "stratscout.video_profiles.probe",
                        return_value={
                            "width": 20,
                            "height": 20,
                            "duration_seconds": 10,
                        },
                    ),
                    patch("cv2.VideoCapture", return_value=cap),
                    patch("cv2.namedWindow"),
                    patch("cv2.selectROI", return_value=(2, 4, 5, 7)),
                    patch("cv2.destroyWindow"),
                    patch("cv2.imshow") as preview,
                    patch("cv2.waitKey", return_value=key),
                    patch("cv2.destroyAllWindows"),
                    patch("builtins.print"),
                ):
                    if key == 13:
                        create_profile("unused.mkv", target)
                        self.assertEqual(load_profile(target)["crop"], [2, 4, 8, 12])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "cancelled"):
                            create_profile("unused.mkv", target)
                        self.assertFalse(target.exists())
                    np.testing.assert_array_equal(preview.call_args.args[1], frame[4:12, 2:8])
                    cap.release.assert_called_once()

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
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "crop.json"
            profile.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(load_profile(profile), good)

    def test_window_syntax(self):
        self.assertEqual(parse_window("55:10"), (55, 10))
        for value in ("bad", "0:0", "-1:2", "nan:3", "2:inf"):
            with self.assertRaises(ValueError):
                parse_window(value)
