import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stratscout.config import TACTIC_CLASSES
from stratscout.tactic_predict import find_clips, prediction_rows, run_predict
from stratscout.tactic_tcn import TacticConfig, run_tactic_experiment


def detections(n_frames, team_points, fps=10.0):
    """Same players on every frame, no bomb: team_points = [(x, y, team), ...]."""
    players = [(k, x, y, team, 0.0) for k in range(n_frames) for x, y, team in team_points]
    return {
        "players": np.asarray(players, np.float32).reshape(-1, 5),
        "bomb": np.zeros((n_frames, 4), np.float32),
        "fps": np.float32(fps),
    }


class PredictTests(unittest.TestCase):
    def test_saved_ensemble_reproduces_the_reported_test_predictions(self):
        rng = np.random.default_rng(0)
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), (0.5, 0.5)]
        rows, dets = [], {}
        for split, match in (("train", "m1"), ("train", "m2"), ("test", "m3")):
            for y, (x0, y0) in enumerate(corners):
                for k in range(2):
                    clip_id = f"{match}.{y}.{k}"
                    n = int(rng.integers(20, 30))
                    dets[clip_id] = detections(n, [(x0, y0 + rng.normal(0, 0.02), 0)])
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "split": split,
                            "label": TACTIC_CLASSES[y],
                            "y": y,
                            "match": match,
                            "duration_s": n / 10,
                        }
                    )
        cfg = TacticConfig(epochs=10, eval_every=10, seeds=(0, 1), channels=8, dilations=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            run_tactic_experiment(
                "synthetic", out, cfg, evaluate_test=True, rows=rows, dets=dets, log=lambda m: None
            )
            saved = {
                r["clip_id"]: r
                for r in csv.DictReader((out / "predictions.csv").read_text().splitlines())
                if r["model"] == "tcn"
            }
            test = {k: v for k, v in dets.items() if k.startswith("m3.")}
            csv_out = Path(tmp) / "new.csv"
            got = run_predict([], out / "models", output=csv_out, dets=test, log=lambda m: None)
            self.assertEqual(len(got), 10)
            for r in got:
                self.assertEqual(r["predicted"], saved[r["clip"]]["predicted"])
                for c in TACTIC_CLASSES:
                    self.assertAlmostEqual(r[f"p_{c}"], float(saved[r["clip"]][f"p_{c}"]), 4)
            self.assertTrue(csv_out.exists())
            with self.assertRaises(FileExistsError):
                run_predict([], out / "models", output=csv_out, dets=test, log=lambda m: None)
            with self.assertRaisesRegex(ValueError, "control-seed"):
                run_predict([], Path(tmp), kind="control", dets=test, log=lambda m: None)

    def test_low_confidence_is_flagged_with_the_runner_up(self):
        probs = np.array([[0.5, 0.1, 0.3, 0.05, 0.05], [0.05, 0.9, 0.02, 0.02, 0.01]])
        rows = prediction_rows(["a", "b"], ["m", "m"], probs, min_confidence=0.6)
        self.assertEqual((rows[0]["predicted"], rows[0]["runner_up"]), ("A_DIRECT", "A_SPLIT"))
        self.assertEqual([r["uncertain"] for r in rows], [True, False])

    def test_find_clips_reads_folders_and_rejects_repeated_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("x/RETAKE/m.s2.mp4", "x/A_DIRECT/m.s1.mp4", "y/m.s2.mp4"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"")
            self.assertEqual([p.name for p in find_clips([root / "x"])], ["m.s1.mp4", "m.s2.mp4"])
            with self.assertRaisesRegex(ValueError, "unique"):
                find_clips([root])
            with self.assertRaisesRegex(ValueError, "does not exist"):
                find_clips([root / "missing.mp4"])


if __name__ == "__main__":
    unittest.main()
