"""Inbox orchestration checks using real encodes, plus failure/recovery boundaries."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stratscout.media import identity, tool
from stratscout.recording_batch import comparison_windows, process_recordings
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

    def run_batch(self):
        return process_recordings(
            self.inbox,
            self.profile,
            self.output,
            emit=lambda message, **kwargs: self.messages.append(message),
        )

    def save_profile(self):
        create_profile(self.video, self.profile, crop=[0, 0, 160, 120])

    def test_first_run_creates_profile_then_rerun_skips_duplicate_content(self):
        with patch(
            "stratscout.recording_batch.create_profile",
            side_effect=lambda video, output, **kwargs: create_profile(
                video, output, crop=[0, 0, 160, 120], **kwargs
            ),
        ) as selector:
            first = self.run_batch()
            selector.assert_called_once()
        self.assertEqual(first[0]["action"], "processed")
        bundle = Path(first[0]["output"])
        report = bundle / "comparison/report.html"
        self.assertTrue(report.is_file())
        before = {p: p.stat().st_mtime_ns for p in bundle.rglob("*") if p.is_file()}
        shutil.copyfile(self.video, self.inbox / "renamed-copy.MKV")
        with patch("stratscout.recording_batch.prepare_video") as encoder:
            second = self.run_batch()
            encoder.assert_not_called()
        self.assertEqual([r["action"] for r in second], ["skipped", "skipped"])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.assertEqual(self.video.read_bytes(), self.original)

    def test_resumes_missing_comparison_without_reencoding(self):
        self.save_profile()
        bundle = self.output / identity(self.video)["sha256"][:16]
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
        bundle = self.output / identity(self.video)["sha256"][:16]
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


class BatchWindowTests(unittest.TestCase):
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
