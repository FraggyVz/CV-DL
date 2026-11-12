import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stratscout.config import TACTIC_CLASSES
from stratscout.tactic_features import clip_features, feature_dim, load_dataset
from stratscout.tactic_tcn import (
    ResidualTCN,
    TacticConfig,
    _eval_features,
    classification_metrics,
    cross_fold_choice,
    pad_batch,
    predict,
    receptive_field,
    run_tactic_experiment,
    train_model,
)


def detections(n_frames, team_points, bomb=(0, 0, 0, 0), fps=10.0):
    """Same players on every frame: team_points = [(x, y, team), ...]."""
    players = [(k, x, y, team, 0.0) for k in range(n_frames) for x, y, team in team_points]
    return {
        "players": np.asarray(players, np.float32).reshape(-1, 5),
        "bomb": np.tile(np.asarray(bomb, np.float32), (n_frames, 1)),
        "fps": np.float32(fps),
    }


class FeatureTests(unittest.TestCase):
    def test_grid_mass_counts_and_elapsed_time(self):
        det = detections(20, [(0.2, 0.3, 0), (0.8, 0.7, 0), (0.5, 0.5, 1)], bomb=(1, 0.4, 0.6, 1))
        f = clip_features(det, step=2, grid=8)
        self.assertEqual(f.shape, (10, feature_dim(8)))
        g2 = 64
        np.testing.assert_allclose(f[:, :g2].sum(1), 2 / 5, atol=1e-5)  # two Ts
        np.testing.assert_allclose(f[:, g2 : 2 * g2].sum(1), 1 / 5, atol=1e-5)  # one CT
        np.testing.assert_allclose(f[0, 2 * g2 : 2 * g2 + 2], [0.4, 0.2], atol=1e-6)
        np.testing.assert_allclose(f[0, 2 * g2 + 8 : 2 * g2 + 12], [1, 0.4, 0.6, 1])
        # Step 3 is frame 6 at 10 fps: 0.6 s, stored as seconds / 30.
        self.assertAlmostEqual(float(f[3, -1]), 0.6 / 30, places=6)

    def test_points_at_the_edges_stay_inside_the_grid(self):
        det = detections(2, [(0.0, 0.0, 0), (1.0, 1.0, 1)])
        f = clip_features(det, step=1, grid=4)
        self.assertAlmostEqual(float(f[0, :16].sum()), 0.2, places=5)
        self.assertAlmostEqual(float(f[0, 16:32].sum()), 0.2, places=5)


class ModelTests(unittest.TestCase):
    def test_padding_does_not_change_a_clips_output(self):
        torch.manual_seed(0)
        model = ResidualTCN(6, 5, channels=8, kernel=3, dilations=(1, 2, 4)).eval()
        short, long = np.random.randn(7, 6), np.random.randn(30, 6)
        alone = model(*pad_batch([short.astype(np.float32)]))
        batched = model(*pad_batch([short.astype(np.float32), long.astype(np.float32)]))
        torch.testing.assert_close(alone[0], batched[0], atol=1e-5, rtol=1e-5)

    def test_receptive_field(self):
        self.assertEqual(receptive_field(3, (1, 2, 4, 8, 16)), 125)

    def test_checkpoint_inside_a_longer_run_equals_a_run_of_that_length(self):
        """The epoch chosen in cross-validation must mean the same model when the final
        run trains for exactly that many epochs, so the schedule cannot depend on it."""
        rng = np.random.default_rng(1)
        train = [(detections(12, [(rng.random(), rng.random(), k % 2)]), k % 5) for k in range(10)]
        cfg = TacticConfig(eval_every=4, channels=8, dilations=(1, 2), batch=4)
        seen = []

        def record(model, norm):
            seen.append(predict(model, _eval_features([d for d, _ in train], norm, cfg)))
            return 0.0

        train_model("tcn", train, cfg, 0, epochs=12, validate=record)
        short, norm, _ = train_model("tcn", train, cfg, 0, epochs=8)
        direct = predict(short, _eval_features([d for d, _ in train], norm, cfg))
        np.testing.assert_allclose(seen[1], direct, atol=1e-5)  # checkpoint 2 = epoch 8

    def test_each_fold_uses_the_other_folds_best_epoch(self):
        curves = {"a": np.array([0.9, 0.1, 0.2]), "b": np.array([0.1, 0.2, 0.8])}
        self.assertEqual(cross_fold_choice(curves), {"a": 2, "b": 0})
        three = {"a": np.array([1.0, 0]), "b": np.array([0, 0.5]), "c": np.array([0, 0.6])}
        self.assertEqual(cross_fold_choice(three), {"a": 1, "b": 0, "c": 0})


class MetricTests(unittest.TestCase):
    def test_macro_f1_uses_classes_present_in_truth(self):
        m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(m["confusion"][0][:2], [1, 1])
        self.assertAlmostEqual(m["per_class"]["A_DIRECT"]["recall"], 0.5)
        self.assertAlmostEqual(m["per_class"]["B_DIRECT"]["precision"], 2 / 3)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(m["per_class"]["RETAKE"]["support"], 0)


class ExperimentTests(unittest.TestCase):
    def test_load_dataset_reads_both_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, match in (("train", "furia-x"), ("test", "navi-x")):
                clip = Path("clips/A_DIRECT") / f"{match}.jx.s001.mp4"
                (root / split / clip).parent.mkdir(parents=True)
                (root / split / clip).write_bytes(b"")
                with (root / split / "manifest.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, ["clip_id", "label", "match_folder", "duration_s", "clip"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "clip_id": f"{match}.s001",
                            "label": "A_DIRECT",
                            "match_folder": match,
                            "duration_s": "3.5",
                            "clip": clip.as_posix(),
                        }
                    )
            rows = load_dataset(root)
        self.assertEqual(
            [(r["split"], r["match"], r["y"]) for r in rows],
            [("train", "furia-x", 0), ("test", "navi-x", 0)],
        )

    def test_synthetic_experiment_learns_separable_classes_and_writes_reports(self):
        """Each class puts its players in its own corner; cross-validation must beat chance."""
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), (0.5, 0.5)]
        rng = np.random.default_rng(0)
        rows, dets = [], {}
        for split, matches in (("train", ("m1", "m2")), ("test", ("m3",))):
            for match in matches:
                for y, (x0, y0) in enumerate(corners):
                    for k in range(3):
                        clip_id = f"{match}.{y}.{k}"
                        n = int(rng.integers(20, 40))
                        pts = [(x0 + rng.normal(0, 0.03), y0 + rng.normal(0, 0.03), 0)]
                        dets[clip_id] = detections(n, pts)
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
        cfg = TacticConfig(epochs=40, eval_every=10, seeds=(0,), channels=16, dilations=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            report = run_tactic_experiment(
                "synthetic", out, cfg, evaluate_test=True, rows=rows, dets=dets, log=lambda m: None
            )
            self.assertTrue((out / "report.md").exists())
            self.assertEqual(
                len(list(csv.DictReader((out / "predictions.csv").read_text().splitlines()))), 30
            )
            # Every training clip is predicted once per model while its match is held out.
            cv_rows = list(csv.DictReader((out / "cv_predictions.csv").read_text().splitlines()))
            self.assertEqual(len(cv_rows), 60)
            tcn_ids = [r["clip_id"] for r in cv_rows if r["model"] == "tcn"]
            self.assertEqual(sorted(tcn_ids), sorted(r["clip_id"] for r in rows[:30]))
            self.assertIn(
                "Per class, pooled over held-out matches", (out / "report.md").read_text()
            )
            self.assertTrue((out / "models" / "tcn-seed0.pt").exists())
            self.assertEqual(
                json.loads((out / "report.json").read_text())["clips"]["test"]["RETAKE"], 3
            )
            with self.assertRaises(FileExistsError):
                run_tactic_experiment(
                    "synthetic", out, cfg, rows=rows, dets=dets, log=lambda m: None
                )
            # An uncertain boundary is refused unless explicitly excluded, then reported.
            marked = [{**rows[0], "uncertain": True}] + rows[1:]
            quick = TacticConfig(epochs=10, eval_every=10, seeds=(0,), models=("control",))
            refused = Path(tmp) / "refused"
            with self.assertRaisesRegex(ValueError, "exclude-uncertain"):
                run_tactic_experiment(
                    "synthetic", refused, quick, rows=marked, dets=dets, log=lambda m: None
                )
            self.assertFalse(refused.exists())
            kept = run_tactic_experiment(
                "synthetic",
                Path(tmp) / "excluded",
                quick,
                exclude_uncertain=True,
                rows=marked,
                dets=dets,
                log=lambda m: None,
            )
            self.assertEqual(kept["excluded_uncertain"], [rows[0]["clip_id"]])
            self.assertEqual(sum(kept["clips"]["train"].values()), 29)
        cv = report["models"]["tcn"]["cv"]
        self.assertEqual(set(cv["cross_fold_epochs"]), {"m1", "m2"})
        self.assertLessEqual(cv["cross_fold_macro_f1"], cv["optimistic_macro_f1"] + 1e-9)
        self.assertGreater(cv["cross_fold_macro_f1"], 0.6)
        self.assertGreater(report["models"]["tcn"]["test"]["ensemble"]["macro_f1"], 0.6)
        pooled = report["models"]["tcn"]["cv"]["pooled"]["ensemble"]
        self.assertEqual(sum(m["support"] for m in pooled["per_class"].values()), 30)
        self.assertEqual(np.sum(pooled["confusion"]), 30)


if __name__ == "__main__":
    unittest.main()
