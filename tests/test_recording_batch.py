"""Inbox orchestration checks using real encodes, plus failure/recovery boundaries."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stratscout.media import identity, tool
from stratscout.recording_batch import (
    SkipRecording,
    bundle_directory,
    comparison_windows,
    match_folder_name,
    parse_time,
    process_recordings,
    prompt_settings,
)
from stratscout.video_preparation import prepare_video
from stratscout.video_profiles import create_profile


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RecordingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inbox = self.root / "inbox"
        self.inbox.mkdir()
        self.video = self.inbox / "match.mkv"
        subprocess.run(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x120:rate=30:duration=1",
                "-c:v",
                "ffv1",
                str(self.video),
            ],
            check=True,
        )
        self.original = self.video.read_bytes()
        self.profile = self.root / "crop.json"
        self.output = self.root / "outputs"
        self.messages = []

    def run_batch(self, **kwargs):
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("playback_speed", 1)  # This small fixture runs at normal speed.
        return process_recordings(
            self.inbox,
            self.profile,
            self.output,
            emit=lambda message, **kwargs: self.messages.append(message),
            **kwargs,
        )

    def save_profile(self):
        create_profile(self.video, self.profile, crop=[0, 0, 160, 120])

    def test_first_run_creates_profile_then_rerun_skips_duplicate_content(self):
        # A different video in the same batch must reuse the first selected crop.
        subprocess.run(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(self.video),
                "-c",
                "copy",
                "-metadata",
                "title=second match",
                str(self.inbox / "second.mkv"),
            ],
            check=True,
        )
        with patch(
            "stratscout.recording_batch.create_profile",
            side_effect=lambda video, output, **kwargs: create_profile(
                video, output, crop=[0, 0, 160, 120], **kwargs
            ),
        ) as selector:
            first = self.run_batch(interactive=True, input_fn=lambda _: "")
            selector.assert_called_once()
        self.assertEqual([r["action"] for r in first], ["processed", "processed"])
        bundle = Path(first[0]["output"])
        report = bundle / "comparison/report.html"
        self.assertTrue(report.is_file())
        before = {p: p.stat().st_mtime_ns for p in bundle.rglob("*") if p.is_file()}
        shutil.copyfile(self.video, self.inbox / "renamed-copy.MKV")
        with (
            patch("stratscout.recording_batch.prepare_video") as encoder,
            patch("builtins.input", side_effect=AssertionError("Completed files need no prompts")),
        ):
            second = self.run_batch(interactive=True)
            encoder.assert_not_called()
        self.assertEqual([r["action"] for r in second], ["skipped"] * 3)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.assertEqual(self.video.read_bytes(), self.original)

    def test_resumes_missing_comparison_without_reencoding(self):
        self.save_profile()
        bundle = self.output / identity(self.video)["sha256"][:16]  # Legacy hash-only output.
        prepare_video(self.video, self.profile, bundle)
        with patch("stratscout.recording_batch.prepare_video") as encoder:
            result = self.run_batch()
            encoder.assert_not_called()
        self.assertEqual(result[0]["action"], "resumed")

    def test_failed_video_does_not_block_others_and_incomplete_output_is_preserved(self):
        self.save_profile()
        (self.inbox / "a-broken.mp4").write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "a-broken.mp4"):
            self.run_batch()
        bundle = self.output / "match"
        report = bundle / "comparison/report.json"
        report.write_text(json.dumps({"status": "incomplete"}))
        with self.assertRaisesRegex(RuntimeError, "match.mkv"):
            self.run_batch()
        self.assertEqual(json.loads(report.read_text()), {"status": "incomplete"})
        self.assertEqual(self.video.read_bytes(), self.original)

    def test_changed_profile_or_missing_exports_refused(self):
        self.save_profile()
        bundle = Path(self.run_batch()[0]["output"])
        original_profile = self.profile.read_text()
        changed = json.loads(original_profile)
        changed["crop"] = [0, 0, 80, 60]
        self.profile.write_text(json.dumps(changed))
        with self.assertRaises(RuntimeError):
            self.run_batch()
        self.profile.write_text(original_profile)
        (bundle / "minimap-5fps.mp4").unlink()
        with self.assertRaises(RuntimeError):
            self.run_batch()

    def test_resolution_mismatch_refused_before_encoding(self):
        self.profile.write_text(
            json.dumps(
                {"schema_version": 1, "source_resolution": [1920, 1080], "crop": [0, 0, 160, 120]}
            )
        )
        with patch("stratscout.recording_batch.prepare_video") as encoder:
            with self.assertRaises(RuntimeError):
                self.run_batch()
            encoder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_interruption_propagates_and_keeps_master(self):
        self.save_profile()
        with patch("stratscout.recording_batch.prepare_video", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.assertEqual(self.video.read_bytes(), self.original)

    def test_cli_prompts_trim_without_settings_files_and_rerun_needs_no_input(self):
        self.save_profile()
        command = [
            sys.executable,
            "-m",
            "stratscout",
            "process-recordings",
            "--input",
            str(self.inbox),
            "--profile",
            str(self.profile),
            "--output",
            str(self.output),
            "--playback-speed",
            "2",
        ]
        result = subprocess.run(command, input="1\n0.2\n0.8\n", text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bundle = self.output / "match"
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["source_offset_seconds"], 0.2)
        self.assertAlmostEqual(manifest["requested_duration_seconds"], 0.6)
        self.assertEqual(manifest["playback_speed"], 1)
        self.assertEqual(list(self.inbox.iterdir()), [self.video])
        before = {p: p.stat().st_mtime_ns for p in bundle.rglob("*") if p.is_file()}
        rerun = subprocess.run(command, input="", text=True, capture_output=True)
        self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
        self.assertIn("Skipped:", rerun.stdout)
        self.assertNotIn("Demo playback speed [", rerun.stdout)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.assertEqual(self.video.read_bytes(), self.original)

    def test_non_interactive_requires_profile_and_sufficient_effective_rate(self):
        with patch("stratscout.recording_batch.prepare_video") as encoder:
            with self.assertRaises(RuntimeError):
                self.run_batch()
            encoder.assert_not_called()
        self.save_profile()
        with patch("stratscout.recording_batch.prepare_video") as encoder:
            with self.assertRaises(RuntimeError):
                self.run_batch(playback_speed=2)
            encoder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_skip_and_closed_input_preserve_source_without_outputs(self):
        result = self.run_batch(interactive=True, input_fn=lambda _: "s")
        self.assertEqual(result[0]["action"], "skipped-by-user")
        with patch("builtins.input", side_effect=EOFError), self.assertRaises(EOFError):
            self.run_batch(interactive=True)
        self.assertFalse(self.profile.exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(self.video.read_bytes(), self.original)


class BatchWindowTests(unittest.TestCase):
    def test_portable_match_names(self):
        self.assertEqual(
            match_folder_name("Furia Falcons Mirage 2026.mkv"), "furia-falcons-mirage-2026"
        )
        self.assertEqual(match_folder_name("CON.mkv"), "match-con")
        self.assertEqual(match_folder_name("???.mkv"), "match")
        self.assertLessEqual(len(match_folder_name("a" * 200 + ".mkv")), 100)

    def test_bundle_identity_survives_renames_and_name_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = {"sha256": "a" * 64}
            original = output / "match"
            original.mkdir()
            (original / "manifest.json").write_text(json.dumps({"source": source}))
            self.assertEqual(bundle_directory(output, "renamed.mkv", source), original)
            different = {"sha256": "b" * 64}
            collision = bundle_directory(output, "match.mp4", different)
            self.assertEqual(collision.name, "match-" + "b" * 16)
            renamed = output / "team-a-vs-team-b"
            original.rename(renamed)
            self.assertEqual(bundle_directory(output, "match.mkv", source), renamed)
            blocked = output / "broken"
            blocked.mkdir()
            with self.assertRaisesRegex(ValueError, "no valid source identity"):
                bundle_directory(output, "broken.mkv", different)
            (blocked / "manifest.json").write_text(json.dumps({"source": source}))
            with self.assertRaisesRegex(ValueError, "Multiple bundles"):
                bundle_directory(output, "match.mkv", source)

    def test_cli_defaults_to_double_speed_and_interactive_crop_setup(self):
        from stratscout.__main__ import main

        with patch("stratscout.recording_batch.process_recordings") as batch:
            main(["process-recordings"])
        self.assertEqual(batch.call_args.kwargs["playback_speed"], 2)
        self.assertTrue(batch.call_args.kwargs["interactive"])
        self.assertEqual(batch.call_args.args[1], Path("configs/recording-video.json"))

    def test_speed_corrected_windows_include_trim_offset(self):
        windows = comparison_windows(
            {
                "variants": [{"duration_seconds": 60}],
                "source_offset_seconds": 120,
                "playback_speed": 2,
            }
        )
        self.assertEqual(windows, [(125, 5), (132.5, 5), (140, 5)])

    def test_time_formats_and_invalid_values(self):
        for value, expected in (("1431", 1431), ("23:51", 1431), ("1:02:03.5", 3723.5)):
            self.assertEqual(parse_time(value), expected)
        for value in ("-1", "nan", "inf", "1:60", "1:60:00", "1:2:3:4", "x", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(value)

    def test_prompt_defaults_and_accelerated_trim(self):
        info = {"duration_seconds": 1500, "fps": 60}
        defaults = prompt_settings(info, input_fn=lambda _: "", emit=lambda _: None)
        self.assertEqual(defaults, {"playback_speed": 2, "start": 0, "duration": None})
        answers = iter(["2x", "0:10", "23:51"])
        settings = prompt_settings(info, input_fn=lambda _: next(answers), emit=lambda _: None)
        self.assertEqual(settings, {"playback_speed": 2, "start": 10, "duration": 1421})

    def test_invalid_prompt_values_reprompt_instead_of_encoding(self):
        info = {"duration_seconds": 10, "fps": 30}
        answers = iter(["nan", "0", "2", "1", "-1", "10", "1", "0", "11", "1.1", "5"])
        messages = []
        settings = prompt_settings(info, input_fn=lambda _: next(answers), emit=messages.append)
        self.assertEqual(settings, {"playback_speed": 1, "start": 1, "duration": 4})
        self.assertTrue(any("fewer than 30" in message for message in messages))

    def test_prompt_skip_and_cancellation_at_any_stage(self):
        info = {"duration_seconds": 10, "fps": 60}
        for stage in range(3):
            answers = iter([""] * stage + ["skip"])
            with self.subTest(stage=stage), self.assertRaises(SkipRecording):
                prompt_settings(info, input_fn=lambda _: next(answers), emit=lambda _: None)
        for error in (KeyboardInterrupt, EOFError):
            with patch("builtins.input", side_effect=error), self.assertRaises(error):
                prompt_settings(info, emit=lambda _: None)

    def test_windows_fit_short_and_long_recordings(self):
        for duration in (0.2, 1, 10, 65.866, 2724.633):
            windows = comparison_windows({"variants": [{"duration_seconds": duration}]})
            self.assertEqual(len(windows), 1 if duration <= 10 else 3)
            for start, length in windows:
                self.assertGreaterEqual(start, 0)
                self.assertGreater(length, 0)
                self.assertLessEqual(start + length, duration + 1e-6)

    def test_empty_inbox_requires_no_tools_or_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("stratscout.recording_batch.versions") as tools:
                self.assertEqual(process_recordings(directory, emit=lambda *a: None), [])
                tools.assert_not_called()
