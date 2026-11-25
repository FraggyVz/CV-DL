"""Analysis workflow checks without loading the vision runtime."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from stratscout import web_analysis


class PrepareSegmentsTests(unittest.TestCase):
    def test_prepared_model_clip_uses_identity_crop_and_source_speed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video_id = "video-1"
            source = root / f"{video_id}.mp4"
            source.write_bytes(b"source")
            segment = {
                "id": "segment-1",
                "start_s": 1.0,
                "end_s": 3.0,
                "round_number": 2,
                "phase": "attack",
            }

            def prepare(_source, profile, output, **kwargs):
                output = Path(output)
                output.mkdir(parents=True)
                (output / "minimap-10fps.mp4").write_bytes(b"prepared")
                self.assertEqual(kwargs["playback_speed"], 1.0)
                self.assertEqual(kwargs["rates"], (10,))
                self.assertEqual(
                    json.loads(Path(profile).read_text(encoding="utf-8"))["crop"],
                    [0, 0, 494, 500],
                )
                return {"variants": [{"file": "minimap-10fps.mp4"}]}

            with (
                patch(
                    "stratscout.web_analysis.probe",
                    return_value={"width": 494, "height": 500, "fps": 10.0},
                ),
                patch(
                    "stratscout.web_analysis.prepare_video", side_effect=prepare
                ) as prepare_video,
            ):
                result = web_analysis._prepare_segments(
                    root,
                    video_id,
                    "analysis-1",
                    [segment],
                    2.0,
                    True,
                    lambda *_: None,
                )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0], segment)
            self.assertEqual(result[0][1].read_bytes(), b"prepared")
            self.assertEqual(
                json.loads(result[0][1].with_name("manifest.json").read_text())["variants"][0][
                    "file"
                ],
                result[0][1].name,
            )
            prepare_video.assert_called_once()

    def test_full_frame_video_requires_a_matching_crop_profile(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video-1.mp4"
            source.write_bytes(b"source")
            segment = {
                "id": "segment-1",
                "start_s": 1.0,
                "end_s": 3.0,
                "round_number": 2,
                "phase": "attack",
            }
            missing_profile = root / "recording-video.json"

            with (
                patch(
                    "stratscout.web_analysis.probe",
                    return_value={"width": 2560, "height": 1440, "fps": 60.0},
                ),
                patch.object(web_analysis, "VIDEO_PROFILE", missing_profile),
                patch("stratscout.web_analysis.prepare_video") as prepare_video,
            ):
                with self.assertRaisesRegex(ValueError, "recording-video.json"):
                    web_analysis._prepare_segments(
                        root,
                        "video-1",
                        "analysis-1",
                        [segment],
                        2.0,
                        False,
                        lambda *_: None,
                    )

            prepare_video.assert_not_called()


class DetectionAndPredictionTests(unittest.TestCase):
    def test_detection_and_both_model_predictions_are_returned_per_segment(self):
        segment = {
            "id": "segment-1",
            "start_s": 1.0,
            "end_s": 3.0,
            "round_number": 2,
            "phase": "attack",
        }
        clip = Path("match.segment-1.mp4")
        prediction = {
            "clip": clip.stem,
            "warnings": [],
            "predictions": {
                "tcn": {"tactic": "A_DIRECT", "confidence": 0.8},
                "gru": {"tactic": "A_SPLIT", "confidence": 0.7},
            },
        }
        predictor = SimpleNamespace(
            load_models=Mock(
                side_effect=[
                    [(None, None, SimpleNamespace(step=2))],
                    [(None, {"settings": {"step": 2}})],
                ]
            ),
            detect=Mock(
                return_value=(
                    {
                        clip.stem: {
                            "n_frames": 20,
                            "players": [[0], [1], [2]],
                            "bomb": [[1.0], [0.0]],
                        }
                    },
                    {clip.stem: ["background uses one segment"]},
                )
            ),
            clip_features=Mock(return_value={clip.stem: {"features": []}}),
            predict_clips=Mock(return_value=[prediction]),
        )
        progress = Mock()

        with patch("stratscout.web_analysis._predictor", return_value=predictor):
            results = web_analysis._detect_and_predict(
                [(segment, clip)],
                0.6,
                progress,
            )

        self.assertEqual(results[0]["segment_id"], "segment-1")
        self.assertEqual(results[0]["detection"]["player_icon_observations"], 3)
        self.assertEqual(results[0]["detection"]["bomb_visible_frames"], 1)
        self.assertEqual(results[0]["warnings"], ["background uses one segment"])
        self.assertEqual(results[0]["predictions"], prediction["predictions"])
        predictor.detect.assert_called_once()
        predictor.predict_clips.assert_called_once()
        self.assertIn("detecting", [call.args[0] for call in progress.call_args_list])
        self.assertIn("classifying", [call.args[0] for call in progress.call_args_list])
