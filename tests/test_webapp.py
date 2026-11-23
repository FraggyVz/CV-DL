"""Web application endpoint checks."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from stratscout import webapp
from stratscout.video_preparation import browser_compatible_video


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.upload_dir = Path(self.temporary.name)
        self.client = TestClient(webapp.app)
        self.upload_patch = patch.object(webapp, "UPLOAD_DIR", self.upload_dir)
        self.upload_patch.start()
        self.addCleanup(self.upload_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def test_health_endpoint_reports_application(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "application": "StratScout"})

    def upload_recording(self):
        with patch(
            "stratscout.media.probe",
            return_value={
                "codec": "mpeg4",
                "pixel_format": "yuv420p",
                "duration_seconds": 18.7,
            },
        ):
            response = self.client.post(
                "/api/videos",
                files={"file": ("match.mp4", b"example video bytes", "video/mp4")},
            )
        self.assertEqual(response.status_code, 201)
        return response.json()

    def upload_round_recording(self, match_id, round_number):
        with patch(
            "stratscout.media.probe",
            return_value={
                "codec": "mpeg4",
                "pixel_format": "yuv420p",
                "duration_seconds": 18.7,
            },
        ):
            response = self.client.post(
                "/api/videos",
                data={
                    "match_id": match_id,
                    "match_name": "Match 1",
                    "round_number": str(round_number),
                    "phase": "attack",
                },
                files={
                    "file": (
                        f"round-{round_number}.mp4",
                        b"example video bytes",
                        "video/mp4",
                    )
                },
            )
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_mp4_upload_is_listed_and_served(self):
        uploaded = self.upload_recording()

        listing = self.client.get("/api/videos")
        with patch(
            "stratscout.video_preparation.browser_compatible_video",
            side_effect=lambda source, output: Path(source),
        ):
            content = self.client.get(uploaded["content_url"])
            partial_content = self.client.get(
                uploaded["content_url"],
                headers={"Range": "bytes=0-6"},
            )

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["videos"], [uploaded])
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, b"example video bytes")
        self.assertEqual(content.headers["content-type"], "video/mp4")
        self.assertEqual(partial_content.status_code, 206)
        self.assertEqual(partial_content.content, b"example")

    def test_prepared_clip_setting_persists_per_recording(self):
        prepared = self.upload_recording()
        regular = self.upload_recording()

        updated = self.client.put(
            f"/api/videos/{prepared['id']}/analysis-settings",
            json={"prepared_clip": True},
        )
        listing = self.client.get("/api/videos")
        videos = {video["id"]: video for video in listing.json()["videos"]}
        metadata = json.loads(
            (self.upload_dir / f"{prepared['id']}.video.json").read_text(encoding="utf-8")
        )

        self.assertEqual(updated.status_code, 200)
        self.assertTrue(updated.json()["prepared_clip"])
        self.assertTrue(videos[prepared["id"]]["prepared_clip"])
        self.assertFalse(videos[regular["id"]]["prepared_clip"])
        self.assertTrue(metadata["prepared_clip"])

    def test_rejects_non_mp4_upload(self):
        response = self.client.post(
            "/api/videos",
            files={"file": ("match.mov", b"video bytes", "video/quicktime")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(list(self.upload_dir.glob("*")), [])

    def test_round_upload_saves_full_clip_as_next_round_segment(self):
        match_id = "d996320a-2be1-4c08-a363-e67b4b6452a5"
        with patch(
            "stratscout.media.probe",
            return_value={
                "codec": "mpeg4",
                "pixel_format": "yuv420p",
                "duration_seconds": 18.7,
            },
        ):
            response = self.client.post(
                "/api/videos",
                data={
                    "match_id": match_id,
                    "match_name": "Match 1",
                    "round_number": "3",
                    "phase": "retake",
                },
                files={"file": ("round-3.mp4", b"example video bytes", "video/mp4")},
            )

        self.assertEqual(response.status_code, 201)
        uploaded = response.json()
        self.assertEqual(uploaded["match_id"], match_id)
        self.assertEqual(uploaded["match_name"], "Match 1")
        self.assertEqual(uploaded["round_number"], 3)
        self.assertEqual(uploaded["phase"], "retake")
        self.assertEqual(uploaded["duration_seconds"], 18.7)
        segments = self.client.get(f"/api/videos/{uploaded['id']}/segments").json()["segments"]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start_s"], 0)
        self.assertEqual(segments[0]["end_s"], 18.7)
        self.assertEqual(segments[0]["round_number"], 3)
        self.assertEqual(segments[0]["phase"], "retake")

    def test_round_upload_requires_complete_match_metadata(self):
        response = self.client.post(
            "/api/videos",
            data={"match_id": "d996320a-2be1-4c08-a363-e67b4b6452a5"},
            files={"file": ("round-1.mp4", b"example video bytes", "video/mp4")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(self.upload_dir.glob("*")), [])

    def test_segment_boundaries_persist_and_validate(self):
        uploaded = self.upload_recording()
        video_id = uploaded["id"]
        segments = [
            {
                "id": "segment-1",
                "start_s": 12.5,
                "end_s": 42.0,
                "round_number": 2,
                "phase": "attack",
            }
        ]

        saved = self.client.put(
            f"/api/videos/{video_id}/segments",
            json={"segments": segments},
        )
        loaded = self.client.get(f"/api/videos/{video_id}/segments")
        invalid = self.client.put(
            f"/api/videos/{video_id}/segments",
            json={
                "segments": [
                    {
                        **segments[0],
                        "end_s": 12.5,
                    }
                ]
            },
        )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(loaded.json()["segments"], segments)
        self.assertEqual(invalid.status_code, 422)

    def test_analysis_runs_saved_segments_and_boundary_edits_stale_results(self):
        uploaded = self.upload_recording()
        segment = {
            "id": "segment-1",
            "start_s": 1.0,
            "end_s": 3.0,
            "round_number": 2,
            "phase": "attack",
        }
        self.client.put(
            f"/api/videos/{uploaded['id']}/segments",
            json={"segments": [segment]},
        )
        result = {
            "segment_id": "segment-1",
            "round_number": 2,
            "phase": "attack",
            "start_s": 1.0,
            "end_s": 3.0,
            "detection": {"frames": 20, "player_icon_observations": 10, "bomb_visible_frames": 4},
            "warnings": [],
            "predictions": {
                "tcn": {"tactic": "A_DIRECT", "confidence": 0.8},
                "gru": {"tactic": "A_SPLIT", "confidence": 0.7},
            },
        }

        with (
            patch(
                "stratscout.web_analysis._prepare_segments",
                return_value=[(segment, Path("match.segment-1.mp4"))],
            ),
            patch("stratscout.web_analysis._detect_and_predict", return_value=[result]),
        ):
            started = self.client.post(f"/api/videos/{uploaded['id']}/analysis", json={})
            state = self.client.get(f"/api/videos/{uploaded['id']}/analysis").json()

        self.assertEqual(started.status_code, 202)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["progress"], 100)
        self.assertEqual(state["results"], [result])

        updated_segment = {**segment, "end_s": 4.0}
        saved = self.client.put(
            f"/api/videos/{uploaded['id']}/segments",
            json={"segments": [updated_segment]},
        )
        stale_state = self.client.get(f"/api/videos/{uploaded['id']}/analysis").json()

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(stale_state["status"], "stale")

    def test_legacy_recording_defaults_to_full_clip_analysis(self):
        uploaded = self.upload_recording()
        segment_response = self.client.get(f"/api/videos/{uploaded['id']}/segments")
        segment = segment_response.json()["segments"][0]

        with (
            patch(
                "stratscout.web_analysis._prepare_segments",
                return_value=[(segment, Path("legacy.segment-1.mp4"))],
            ),
            patch("stratscout.web_analysis._detect_and_predict", return_value=[]),
        ):
            response = self.client.post(f"/api/videos/{uploaded['id']}/analysis", json={})

        self.assertEqual(segment["start_s"], 0)
        self.assertEqual(segment["end_s"], uploaded["duration_seconds"])
        self.assertEqual(response.status_code, 202)

    def test_removing_round_deletes_it_and_renumbers_later_rounds(self):
        match_id = "d996320a-2be1-4c08-a363-e67b4b6452a5"
        first = self.upload_round_recording(match_id, 1)
        removed = self.upload_round_recording(match_id, 2)
        later = self.upload_round_recording(match_id, 3)
        analysis_path = self.upload_dir / f"{later['id']}.analysis.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "results": [{"round_number": 3}],
                }
            ),
            encoding="utf-8",
        )

        response = self.client.delete(f"/api/videos/{removed['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["renumbered"], {later["id"]: 2})
        listing = self.client.get("/api/videos").json()["videos"]
        rounds = {video["id"]: video["round_number"] for video in listing}
        self.assertEqual(rounds, {first["id"]: 1, later["id"]: 2})
        remaining_segments = self.client.get(f"/api/videos/{later['id']}/segments").json()[
            "segments"
        ]
        updated_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        self.assertEqual(remaining_segments[0]["round_number"], 2)
        self.assertEqual(updated_analysis["results"][0]["round_number"], 2)
        self.assertFalse((self.upload_dir / f"{removed['id']}.mp4").exists())

    def test_reset_removes_all_app_recordings_but_preserves_unrelated_files(self):
        self.upload_recording()
        round_recording = self.upload_round_recording(
            "d996320a-2be1-4c08-a363-e67b4b6452a5",
            1,
        )
        unrelated = self.upload_dir / "keep-this.txt"
        unrelated.write_text("not an app upload", encoding="utf-8")
        analysis_directory = self.upload_dir / "analysis" / round_recording["id"]
        analysis_directory.mkdir(parents=True)
        (analysis_directory / "output.txt").write_text("analysis", encoding="utf-8")

        response = self.client.delete("/api/videos")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted_count": 2})
        self.assertEqual(self.client.get("/api/videos").json()["videos"], [])
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "not an app upload")
        self.assertFalse(analysis_directory.exists())

    def test_removal_and_reset_are_blocked_while_analysis_is_running(self):
        match_id = "d996320a-2be1-4c08-a363-e67b4b6452a5"
        first = self.upload_round_recording(match_id, 1)
        later = self.upload_round_recording(match_id, 2)
        analysis_id = "analysis-in-progress"
        (self.upload_dir / f"{later['id']}.analysis.json").write_text(
            json.dumps(
                {
                    "analysis_id": analysis_id,
                    "status": "preparing",
                }
            ),
            encoding="utf-8",
        )

        with patch("stratscout.web_analysis.ACTIVE_ANALYSES", {analysis_id}):
            remove_response = self.client.delete(f"/api/videos/{first['id']}")
            reset_response = self.client.delete("/api/videos")

        self.assertEqual(remove_response.status_code, 409)
        self.assertEqual(reset_response.status_code, 409)
        self.assertTrue((self.upload_dir / f"{first['id']}.mp4").is_file())
        self.assertTrue((self.upload_dir / f"{later['id']}.mp4").is_file())

    def test_upload_limit_removes_partial_file(self):
        with patch.object(webapp, "MAX_UPLOAD_BYTES", 3):
            response = self.client.post(
                "/api/videos",
                files={"file": ("match.mp4", b"more than three bytes", "video/mp4")},
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(list(self.upload_dir.glob("*")), [])


class BrowserVideoTests(unittest.TestCase):
    def test_h264_source_is_reused_without_encoding(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")
            output = Path(directory) / "browser.mp4"
            info = {"codec": "h264", "pixel_format": "yuv420p"}

            with (
                patch("stratscout.video_preparation.probe", return_value=info),
                patch("stratscout.video_preparation.run_ffmpeg") as run_ffmpeg,
            ):
                result = browser_compatible_video(source, output)

            self.assertEqual(result, source.resolve())
            run_ffmpeg.assert_not_called()
            self.assertFalse(output.exists())

    def test_unsupported_codec_gets_h264_browser_copy(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")
            output = Path(directory) / "browser.mp4"
            probe_results = iter(
                [
                    {"codec": "mpeg4", "pixel_format": "yuv420p"},
                    {"codec": "h264", "pixel_format": "yuv420p"},
                ]
            )

            def encode(arguments, _log):
                Path(arguments[-1]).write_bytes(b"encoded")

            with (
                patch(
                    "stratscout.video_preparation.probe",
                    side_effect=lambda _path: next(probe_results),
                ),
                patch("stratscout.video_preparation.run_ffmpeg", side_effect=encode) as run_ffmpeg,
            ):
                result = browser_compatible_video(source, output)

            self.assertEqual(result, output.resolve())
            self.assertEqual(output.read_bytes(), b"encoded")
            arguments = run_ffmpeg.call_args.args[0]
            self.assertIn("libx264", arguments)
            self.assertIn("yuv420p", arguments)
            self.assertIn("+faststart", arguments)


if __name__ == "__main__":
    unittest.main()
