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

    def test_memory_change_is_the_step_to_step_distance_per_column(self):
        outputs = np.array([[0, 0], [3, 4], [3, 4], [0, 0]], np.float32)
        # Changes: none for the first row, then 5, 0, 5; two rows per column.
        self.assertEqual(web_pipeline._memory_change(outputs, web_pipeline._bins(4, 2)), [5.0, 2.5])
        self.assertEqual(
            web_pipeline._memory_change(outputs, web_pipeline._bins(4, 4)), [0.0, 5.0, 0.0, 5.0]
        )


GRU_MODELS = (
    Path(__file__).resolve().parents[1] / "runs" / "tactic-gru-clipdata-final-20261121" / "models"
)


@unittest.skipUnless(GRU_MODELS.is_dir(), "the final GRU checkpoints are not present")
class GruStageTests(unittest.TestCase):
    """The GRU walkthrough on the saved final GRU, with made-up clip features."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        from stratscout import gru_predict

        cls.members = gru_predict.load_gru(GRU_MODELS)
        settings = cls.members[0][1]["settings"]
        rows = 47
        rng = np.random.default_rng(0)
        cls.features = {
            "detector": rng.uniform(0, 1, (rows, settings["detector_width"])).astype(np.float32),
            "resnet": rng.uniform(0, 2, (rows, settings["resnet_width"])).astype(np.float32),
            "t_s": (np.arange(rows) / settings["rows_per_second"]).astype(np.float32),
        }
        cls.settings = settings
        cls.bins = web_pipeline._bins(rows, 20)
        cls.stage = web_pipeline._gru_stage(cls.members, cls.features, cls.bins)

    def test_architecture_widths_match_the_saved_config(self):
        from stratscout.tactic_temporal import TIME_SCALE_S

        sizes = self.settings["gru"]
        arch = self.stage["architecture"]
        self.assertEqual(arch["projection_in"], self.stage["input_width"])
        self.assertEqual(
            self.stage["input_width"], self.settings["detector_width"] + self.settings["resnet_pcs"]
        )
        self.assertEqual(arch["projection"], sizes["projection"])
        self.assertEqual(arch["hidden"], sizes["hidden"])
        self.assertEqual(arch["readout"], sizes["readout"])
        self.assertEqual(arch["time_inputs"], 2)
        self.assertEqual(arch["gru_input"], sizes["projection"] + arch["time_inputs"])
        self.assertEqual(arch["readout_width"], 2 * sizes["hidden"])
        self.assertEqual(arch["classes"], len(self.stage["ensemble"]))
        self.assertEqual(arch["time_scale_s"], TIME_SCALE_S)

    def test_cell_inputs_memory_and_change_cover_every_column(self):
        columns = len(self.bins)
        self.assertEqual(
            len(self.stage["projection"]["labels"]), self.stage["architecture"]["projection"]
        )
        self.assertEqual(len(self.stage["time_inputs"]["labels"]), 2)
        self.assertEqual(len(self.stage["hidden"]["labels"]), self.stage["architecture"]["hidden"])
        for heat in (self.stage["projection"], self.stage["time_inputs"], self.stage["hidden"]):
            self.assertTrue(all(len(row) == columns for row in heat["values"]))
        self.assertEqual(len(self.stage["memory_change"]), columns)
        self.assertGreaterEqual(self.stage["hidden"]["min"], -1.0)
        self.assertLessEqual(self.stage["hidden"]["max"], 1.0)
        self.assertTrue(all(v >= 0 for v in self.stage["memory_change"]))
        self.assertEqual(len(self.stage["row_times_s"]), len(self.features["t_s"]))

    def test_time_inputs_are_seconds_over_the_time_scale(self):
        from stratscout.tactic_temporal import TIME_SCALE_S

        elapsed, gap = self.stage["time_inputs"]["values"]
        times = self.features["t_s"]
        last = self.bins[-1]
        self.assertAlmostEqual(elapsed[-1], float(times[last].mean()) / TIME_SCALE_S, places=3)
        step = 1 / self.settings["rows_per_second"]
        self.assertAlmostEqual(gap[-1], step / TIME_SCALE_S, places=3)

    def test_walkthrough_ensemble_matches_the_prediction_api(self):
        from stratscout import gru_predict
        from stratscout.config import TACTIC_CLASSES

        expected = gru_predict.gru_probs(self.members, [self.features])[0]
        got = [self.stage["ensemble"][c] for c in TACTIC_CLASSES]
        np.testing.assert_allclose(got, expected, atol=1e-4)


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
