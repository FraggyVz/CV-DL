"""The web app's stage-by-stage pipeline record, without loading the vision runtime."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from stratscout import web_analysis, web_pipeline


class PipelineHelperTests(unittest.TestCase):
    def test_time_is_binned_to_at_most_the_column_limit(self):
        bins = web_pipeline._bins(312)
        self.assertEqual(len(bins), web_pipeline.MAX_COLUMNS)
        self.assertEqual(sum(len(b) for b in bins), 312)
        self.assertEqual(len(web_pipeline._bins(40)), 40)

    def test_heatmap_is_channels_by_columns_and_json_short(self):
        steps = np.arange(6, dtype=np.float32)[:, None] * np.array([[0.1, 1.0]], np.float32)
        heat = web_pipeline._heatmap(steps, web_pipeline._bins(6, 3), ["a", "b"])
        self.assertEqual(heat["labels"], ["a", "b"])
        self.assertEqual(heat["values"], [[0.1, 0.2, 0.4], [0.5, 2.5, 4.5]])
        self.assertEqual(heat["max"], 4.5)
        self.assertNotIn("0000", json.dumps(heat))

    def test_peak_columns_count_channels_per_column(self):
        bins = web_pipeline._bins(6, 3)
        self.assertEqual(web_pipeline._peak_columns(np.array([0, 1, 5, 5]), bins), [2, 0, 2])


class SavedPipelineTests(unittest.TestCase):
    def _complete_state(self, root, video_id, analysis_id="analysis-1"):
        (root / f"{video_id}.analysis.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "analysis_id": analysis_id,
                    "status": "complete",
                    "results": [{"segment_id": "segment-1"}],
                }
            ),
            encoding="utf-8",
        )
        return root / "analysis" / video_id / analysis_id / "pipeline"

    def test_saved_record_is_served_for_the_latest_analysis(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline_dir = self._complete_state(root, "video-1")
            record = {"format": web_pipeline.FORMAT, "steps": 3}
            with patch.object(web_pipeline, "build_pipeline", return_value=record):
                saved = web_analysis._save_pipeline(
                    pipeline_dir, "segment-1", Path("clip.mp4"), {}, {}, [], []
                )
            self.assertEqual(saved, {"pipeline": True})
            self.assertEqual(web_analysis.get_pipeline(root, "video-1", "segment-1"), record)
            with self.assertRaises(FileNotFoundError):
                web_analysis.get_pipeline(root, "video-1", "segment-2")

    def test_a_failed_record_is_reported_without_failing_the_analysis(self):
        with TemporaryDirectory() as directory:
            with patch.object(web_pipeline, "build_pipeline", side_effect=RuntimeError("boom")):
                saved = web_analysis._save_pipeline(
                    Path(directory), "segment-1", Path("clip.mp4"), {}, {}, [], []
                )
            self.assertEqual(saved, {"pipeline": False, "pipeline_error": "RuntimeError: boom"})

    def test_segment_ids_cannot_leave_the_pipeline_folder(self):
        with self.assertRaises(ValueError):
            web_analysis._pipeline_path(Path("x"), "../secret")


if __name__ == "__main__":
    unittest.main()
