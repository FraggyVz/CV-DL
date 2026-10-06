import json
from pathlib import Path
import tempfile
import unittest

from helpers import profile
from stratscout.evaluation import evaluate_run, match_points


class MatchingTests(unittest.TestCase):
    def test_duplicate_prediction_cannot_match_same_player_twice(self):
        point = {"x": .5, "y": .5}
        self.assertEqual(match_points([point, point], [point], 100, 100), (1, 1, 0))

    def test_assignment_recovers_match_that_greedy_would_miss(self):
        predicted = [{"x": .5, "y": .5}, {"x": .51, "y": .5}]
        truth = [{"x": .5, "y": .5}, {"x": .475, "y": .5}]
        self.assertEqual(match_points(predicted, truth, 100, 100), (2, 0, 0))

    def test_radius_uses_native_aspect_ratio(self):
        self.assertEqual(match_points([{"x": .54, "y": .5}], [{"x": .5, "y": .5}],
                                     100, 300), (1, 0, 0))
        self.assertEqual(match_points([{"x": .5, "y": .54}], [{"x": .5, "y": .5}],
                                     100, 300), (0, 1, 1))


class EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "run"
        run.mkdir()
        config = json.loads(json.dumps(profile().to_dict()))
        (run / "run.json").write_text(json.dumps({"status": "complete", "profile": config,
                                                 "frames_processed": 30, "input": {"sha256": "0" * 64}}))
        gold = {"schema_version": 1, "video_sha256": "0" * 64,
                "source_resolution": config["source_resolution"], "crop": config["crop"], "frames": []}
        predictions = []
        for index in range(30):
            points = {"T": [{"x": .3, "y": .4}], "CT": [{"x": .7, "y": .6}]}
            gold["frames"].append({"frame_index": index, "round_id": f"round_{index // 10}",
                                   "split": "calibration" if index < 10 else "evaluation",
                                   "review_status": "reviewed", **points})
            predictions.append({"frame_index": index, **points})
        (run / "detections.jsonl").write_text("\n".join(json.dumps(p) for p in predictions))
        annotations = root / "gold.json"
        annotations.write_text(json.dumps(gold))
        return run, annotations, gold

    def test_metrics_and_sample_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, annotations, _ = self.fixture(root)
            report = evaluate_run(run, annotations, root / "metrics.json")
            self.assertTrue(report["quantitative_target_met"])
            for team in ("T", "CT"):
                self.assertEqual(report["teams"][team]["tp"], 20)
                self.assertEqual(report["teams"][team]["precision"], 1)
                self.assertEqual(report["teams"][team]["recall"], 1)
            with self.assertRaises(FileExistsError):
                evaluate_run(run, annotations, root / "metrics.json")

    def test_bad_provenance_review_and_split_are_rejected(self):
        for scenario in ("hash", "crop", "overlap", "unreviewed", "duplicate", "missing_team", "nan"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, annotations, gold = self.fixture(root)
                if scenario == "hash": gold["video_sha256"] = "wrong"
                if scenario == "crop": gold["crop"][0] = 0
                if scenario == "overlap": gold["frames"][10]["round_id"] = "round_0"
                if scenario == "unreviewed": gold["frames"][10]["review_status"] = "pending"
                if scenario == "duplicate": gold["frames"].append(gold["frames"][0])
                if scenario == "missing_team": del gold["frames"][10]["CT"]
                if scenario == "nan": gold["frames"][10]["T"][0]["x"] = float("nan")
                annotations.write_text(json.dumps(gold))
                with self.assertRaises(ValueError):
                    evaluate_run(run, annotations, root / "metrics.json")
                self.assertFalse((root / "metrics.json").exists())

    def test_tiny_sample_does_not_pass_milestone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, annotations, gold = self.fixture(root)
            gold["frames"] = [gold["frames"][10]]
            annotations.write_text(json.dumps(gold))
            result = evaluate_run(run, annotations, root / "metrics.json")
            self.assertFalse(result["quantitative_target_met"])

    def test_malformed_annotation_containers_have_clear_errors(self):
        for invalid in ([], None, {"schema_version": 1, "video_sha256": "0" * 64,
                                  "source_resolution": [320, 240], "crop": [80, 60, 280, 220],
                                  "frames": ["not a frame object"]}):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, annotations, _ = self.fixture(root)
                annotations.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    evaluate_run(run, annotations, root / "metrics.json")


if __name__ == "__main__":
    unittest.main()
