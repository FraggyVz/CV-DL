"""Synthetic regression checks for Plan A contracts, not measured tactic accuracy."""

import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from stratscout.__main__ import main
from stratscout.config import TACTIC_CLASSES
from stratscout.context import RunContext
from stratscout.segment_export import COLUMNS
from stratscout.tactic_data import MatchFold
from stratscout.tactic_model import (
    RGB_MEAN,
    RGB_STD,
    TacticBaseline,
    build_baseline,
    load_baseline,
    normalise_frames,
)
from stratscout.tactic_training import (
    BaselineConfig,
    classification_metrics,
    fit_classifier,
    fit_folds,
    run_baseline,
    summarise_runs,
    training_class_weights,
)


def tiny_model(*, context, pretrained=True):
    # BatchNorm catches accidental updating of a frozen encoder's running state.
    return TacticBaseline(
        nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3
    ).to(context.device)


class TacticModelTests(unittest.TestCase):
    def setUp(self):
        self.context = RunContext()

    def test_normalisation_preserves_full_rgb_map_and_rejects_double_scaling(self):
        frames = np.zeros((2, 7, 9, 3), dtype=np.uint8)
        frames[0, 0, 0] = (255, 128, 0)
        frames[1, -1, -1] = (0, 64, 255)
        actual = normalise_frames(frames, context=self.context)
        self.assertEqual(tuple(actual.shape), (2, 3, 7, 9))
        expected = (frames.astype(np.float32) / 255 - RGB_MEAN) / RGB_STD
        np.testing.assert_allclose(actual.numpy(), expected.transpose(0, 3, 1, 2), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "uint8"):
            normalise_frames(frames.astype(np.float32) / 255, context=self.context)

    def test_encoder_parameters_and_batchnorm_stay_frozen(self):
        model = tiny_model(context=self.context).train()
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        frames = torch.randn(2, 3, 3, 8, 8)
        optimiser = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        head_before = model.classifier.weight.detach().clone()
        model(frames).sum().backward()
        optimiser.step()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(
            all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
        )
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(head_before, model.classifier.weight))
        torch.testing.assert_close(model.encode(frames), model.encode(frames.flip(1)))

    def test_resnet_has_five_outputs_without_network_access(self):
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Network used")):
            model = build_baseline(context=self.context, pretrained=False).eval()
            output = model(torch.zeros(1, 2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 5))
        self.assertEqual(model.classifier.in_features, 512)

    def test_metrics_use_all_classes_and_true_rows(self):
        metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], TACTIC_CLASSES)
        self.assertEqual(metrics["confusion_matrix"][0], [1, 1, 0, 0, 0])
        self.assertEqual(metrics["confusion_matrix"][2], [0, 1, 0, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.5) / 5)
        self.assertEqual(metrics["per_class"][TACTIC_CLASSES[4]]["support"], 0)

    def test_weights_and_one_epoch_fit_do_not_use_validation_data(self):
        features = torch.arange(39, dtype=torch.float32).reshape(13, 3) / 39
        targets = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        fold = MatchFold(tuple(range(8)), tuple(range(8, 13)), ("furia",), "spirit")
        config = BaselineConfig(epochs=1, seeds=(7,))
        head, run, _ = fit_classifier(
            features,
            targets,
            fold,
            classes=TACTIC_CLASSES,
            config=config,
            seed=7,
            context=self.context,
        )
        changed_features, changed_targets = features.clone(), targets.clone()
        changed_features[8:] += 200
        changed_targets[8:] = 4
        other_head, other_run, _ = fit_classifier(
            changed_features,
            changed_targets,
            fold,
            classes=TACTIC_CLASSES,
            config=config,
            seed=7,
            context=self.context,
        )
        np.testing.assert_allclose(list(run["class_weights"].values()), [0.4, 1.6, 1.6, 1.6, 1.6])
        self.assertEqual(run["class_weights"], other_run["class_weights"])
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, other_head.state_dict()[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "no examples"):
            training_class_weights(torch.tensor([0, 1]), TACTIC_CLASSES)

    def test_best_epoch_is_restored_and_repeated_seed_reproduces(self):
        features = torch.eye(5).repeat(2, 1)
        targets = torch.arange(5).repeat(2)
        fold = MatchFold(tuple(range(5)), tuple(range(5, 10)), ("furia",), "spirit")
        config = BaselineConfig(epochs=12, patience=3)
        _, run, probabilities = fit_classifier(
            features,
            targets,
            fold,
            classes=TACTIC_CLASSES,
            config=config,
            seed=2,
            context=self.context,
        )
        best = max(run["history"], key=lambda r: (r["validation_macro_f1"], -r["validation_loss"]))
        self.assertEqual(run["best_epoch"], best["epoch"])
        self.assertEqual(run["metrics"]["macro_f1"], best["validation_macro_f1"])
        _, repeated, repeated_probs = fit_classifier(
            features,
            targets,
            fold,
            classes=TACTIC_CLASSES,
            config=config,
            seed=2,
            context=self.context,
        )
        self.assertEqual(run, repeated)
        np.testing.assert_array_equal(probabilities, repeated_probs)

    def test_summary_pairs_seeds_before_averaging_folds(self):
        folds = [
            {
                "runs": [
                    {"seed": i, "metrics": {"macro_f1": score, "accuracy": score}}
                    for i, score in enumerate(scores)
                ],
                "majority": {"macro_f1": 0.1, "accuracy": 0.2},
            }
            for scores in ([0.1, 0.9], [0.9, 0.1])
        ]
        result = summarise_runs(folds, (0, 1))
        self.assertEqual(result["model"]["macro_f1"]["mean"], 0.5)
        self.assertEqual(result["model"]["macro_f1"]["std"], 0)
        self.assertEqual(result["majority"]["accuracy"], 0.2)

    def test_fixed_epochs_scores_only_n_and_ignores_validation_for_training(self):
        features = torch.eye(5).repeat(2, 1)
        targets = torch.arange(5).repeat(2)
        fold = MatchFold(tuple(range(5)), tuple(range(5, 10)), ("furia",), "spirit")
        config = BaselineConfig(epochs=2, patience=1, fixed_epochs=5)
        with patch(
            "stratscout.tactic_training.classification_metrics", wraps=classification_metrics
        ) as scores:
            head, run, probabilities = fit_classifier(
                features,
                targets,
                fold,
                classes=TACTIC_CLASSES,
                config=config,
                seed=2,
                context=self.context,
            )
        self.assertEqual(scores.call_count, 1)
        self.assertEqual(run["epochs_run"], 5)  # Overrides both --epochs and patience.
        self.assertEqual(run["evaluation_epoch"], 5)
        self.assertIsNone(run["best_epoch"])
        self.assertEqual(run["epoch_selection"], "fixed")
        self.assertFalse(any("validation_loss" in row for row in run["history"][:-1]))
        self.assertEqual(run["history"][-1]["validation_macro_f1"], run["metrics"]["macro_f1"])
        changed_targets = targets.clone()
        changed_targets[5:] = (changed_targets[5:] + 1) % 5
        other, _, _ = fit_classifier(
            features,
            changed_targets,
            fold,
            classes=TACTIC_CLASSES,
            config=config,
            seed=2,
            context=self.context,
        )
        for key, tensor in head.state_dict().items():
            torch.testing.assert_close(tensor, other.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            np.testing.assert_array_equal(probabilities, head(features[5:]).softmax(1).numpy())

    def test_cross_fold_uses_other_fold_epoch_for_same_seed_and_refits_exactly(self):
        features = torch.eye(5).repeat(2, 1)
        targets = torch.arange(5).repeat(2)
        folds = (
            MatchFold(tuple(range(5)), tuple(range(5, 10)), ("furia",), "spirit"),
            MatchFold(tuple(range(5, 10)), tuple(range(5)), ("spirit",), "furia"),
        )
        config = BaselineConfig(epochs=6, patience=1, seeds=(3, 7))
        selected = {"spirit": {3: 1, 7: 2}, "furia": {3: 5, 7: 4}}

        def select_or_fit(x, y, fold, *, classes, config, seed, context):
            if config.fixed_epochs is None:
                # Controlled pilot results expose fold/seed mix-ups. Refit the actual head below.
                return (
                    None,
                    {"seed": seed, "best_epoch": selected[fold.validation_match_id][seed]},
                    None,
                )
            return fit_classifier(
                x, y, fold, classes=classes, config=config, seed=seed, context=context
            )

        with patch("stratscout.tactic_training.fit_classifier", side_effect=select_or_fit) as fit:
            results = fit_folds(
                features,
                targets,
                folds,
                classes=TACTIC_CLASSES,
                config=config,
                context=self.context,
            )
        self.assertEqual(fit.call_count, 8)  # Four pilots and four fresh exact-epoch fits.
        for index, runs in enumerate(results):
            for head, run, probs in runs:
                other = 1 - index
                expected_epoch = selected[folds[other].validation_match_id][run["seed"]]
                self.assertEqual(run["evaluation_epoch"], expected_epoch)
                self.assertEqual(run["epochs_run"], expected_epoch)
                self.assertEqual(run["epoch_source_fold"], other + 1)
                self.assertEqual(run["epoch_selection"], "cross-fold")
                self.assertEqual(
                    run["optimistic"]["best_epoch"],
                    selected[folds[index].validation_match_id][run["seed"]],
                )
                reference, _, reference_probs = fit_classifier(
                    features,
                    targets,
                    folds[index],
                    classes=TACTIC_CLASSES,
                    config=replace(config, fixed_epochs=expected_epoch),
                    seed=run["seed"],
                    context=self.context,
                )
                for key, tensor in head.state_dict().items():
                    torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=0, atol=0)
                np.testing.assert_array_equal(probs, reference_probs)

    def test_cross_fold_rejects_ambiguous_number_of_folds_before_fitting(self):
        with patch("stratscout.tactic_training.fit_classifier") as fit:
            with self.assertRaisesRegex(ValueError, "exactly two folds"):
                fit_folds(
                    torch.eye(5),
                    torch.arange(5),
                    (),
                    classes=TACTIC_CLASSES,
                    config=BaselineConfig(),
                    context=self.context,
                )
        fit.assert_not_called()

    def test_cli_epoch_modes_and_conflicts(self):
        base = ["tactic-baseline", "--manifest", "unused.csv", "--output", "unused-output"]
        for flags, expected in (([], None), (["--cross-fold"], None), (["--fixed-epochs", "5"], 5)):
            with self.subTest(flags=flags), patch("stratscout.tactic_training.run_baseline") as run:
                main(base + flags)
                self.assertEqual(run.call_args.kwargs["config"].fixed_epochs, expected)
        for flags in (["--cross-fold", "--fixed-epochs", "5"], ["--fixed-epochs", "0"]):
            with (
                patch("stratscout.tactic_training.run_baseline") as run,
                patch("sys.stderr", new=io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    main(base + flags)
                run.assert_not_called()

    def test_invalid_settings_fail_early(self):
        for kwargs in (
            {"seeds": ()},
            {"seeds": (1, 1)},
            {"size": 16},
            {"epochs": 0},
            {"learning_rate": float("nan")},
            {"fixed_epochs": 0},
            {"fixed_epochs": -1},
            {"fixed_epochs": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BaselineConfig(**kwargs)


class TacticRunTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "train" / "manifest.csv"
        self.manifest.parent.mkdir()
        self.output = self.root / "run"
        self.context = RunContext()
        self.config = BaselineConfig(size=32, n_frames=2, epochs=3, patience=2, seeds=(3, 7))
        # Every class exists in both matches, with different majority classes.
        rows = []
        for match, labels in (("furia", [0, 0, 1, 2, 3, 4]), ("spirit", [0, 1, 1, 1, 2, 3, 4])):
            for i, label in enumerate(labels):
                clip_id = f"{match}-{i}"
                path = self.manifest.parent / f"{clip_id}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
                self.assertTrue(writer.isOpened())
                try:
                    for frame in range(3):
                        writer.write(np.full((32, 32, 3), (label * 40, 20 + frame, 100), np.uint8))
                finally:
                    writer.release()
                row = dict.fromkeys(COLUMNS, "")
                row.update(
                    clip_id=clip_id,
                    clip=path.name,
                    split="train",
                    label=TACTIC_CLASSES[label],
                    match_id=match,
                    match_folder=match,
                    start_s=1,
                    end_s=1.2,
                    duration_s=0.2,
                    frames=3,
                    status="reviewed",
                    boundary_uncertain=False,
                    definition_version="synthetic-test",
                    source_video="unused.mp4",
                )
                rows.append(row)
        excluded = dict(rows[0], clip_id="uncertain", clip="absent.mp4", boundary_uncertain=True)
        rows.append(excluded)
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        # An invalid neighbouring test manifest must never be opened.
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "manifest.csv").write_text("DO NOT READ", encoding="utf-8")

    def test_end_to_end_outputs_majorities_exclusions_and_offline_reload(self):
        with patch("stratscout.tactic_training.build_baseline", side_effect=tiny_model):
            with self.assertWarns(UserWarning):
                summary = run_baseline(
                    self.manifest,
                    self.output,
                    context=self.context,
                    config=self.config,
                    exclude_uncertain=True,
                )
        settings = json.loads((self.output / "config.json").read_text())
        self.assertEqual(settings["excluded_clip_ids"], ["uncertain"])
        self.assertEqual(sum(settings["class_counts"].values()), 13)
        self.assertEqual(len(json.loads((self.output / "samples.json").read_text())), 13)
        furia = json.loads((self.output / "fold-1.json").read_text())
        spirit = json.loads((self.output / "fold-2.json").read_text())
        self.assertEqual(furia["majority_class"], "B_DIRECT")
        self.assertEqual(spirit["majority_class"], "A_DIRECT")
        self.assertAlmostEqual(furia["majority"]["accuracy"], 1 / 6)
        self.assertAlmostEqual(spirit["majority"]["accuracy"], 1 / 7)
        self.assertTrue(set(furia["train_clip_ids"]).isdisjoint(furia["validation_clip_ids"]))
        self.assertEqual(summary, json.loads((self.output / "summary.json").read_text()))
        with (self.output / "predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 13 * 2)
        checkpoint = self.output / "fold-1-seed-3.pt"
        with patch("stratscout.tactic_model.build_baseline", side_effect=tiny_model) as factory:
            model, saved = load_baseline(checkpoint, context=self.context)
        factory.assert_called_once_with(context=self.context, pretrained=False)
        self.assertFalse(model.training)
        data = torch.load(self.output / "features.pt", weights_only=True)
        indices = [data["clip_ids"].index(clip) for clip in furia["validation_clip_ids"]]
        with torch.no_grad():
            probs = model.classifier(data["features"][indices]).softmax(1).numpy()
        rows = [r for r in predictions if r["fold"] == "1" and r["seed"] == "3"]
        expected = [[float(r[f"p_{name}"]) for name in TACTIC_CLASSES] for r in rows]
        np.testing.assert_allclose(probs, expected, atol=1e-7)
        self.assertEqual(saved["settings"]["size"], 32)
        self.assertEqual(summary["epoch_mode"], "cross-fold")
        self.assertEqual(saved["epoch_selection"], "cross-fold")
        self.assertEqual(saved["epoch_source_fold"], 2)
        self.assertEqual(saved["evaluation_epoch"], spirit["runs"][0]["optimistic"]["best_epoch"])
        report = (self.output / "report.md").read_text(encoding="utf-8")
        self.assertIn("optimistic (selected on validation)", report)
        self.assertIn("Majority accuracy", report)
        self.assertIn("+/-", report)
        self.assertNotIn("\u00b1", report)
        for fold, name in ((furia, "furia"), (spirit, "spirit")):
            section = report.split(f"## Per-class F1: {name}")[1].split("##")[0]
            for label in TACTIC_CLASSES:
                mean_f1 = np.mean(
                    [run["metrics"]["per_class"][label]["f1"] for run in fold["runs"]]
                )
                self.assertIn(
                    f"| {label} | {fold['validation_counts'][label]} | {mean_f1:.3f} |", section
                )
        with (self.output / "encoder.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_baseline(checkpoint, context=self.context)

    def test_fixed_run_skips_pilots_and_saves_final_epoch(self):
        config = replace(self.config, fixed_epochs=4)
        with patch("stratscout.tactic_training.build_baseline", side_effect=tiny_model):
            with patch("stratscout.tactic_training.fit_classifier", wraps=fit_classifier) as fit:
                with self.assertWarns(UserWarning):
                    summary = run_baseline(
                        self.manifest,
                        self.output,
                        context=self.context,
                        config=config,
                        exclude_uncertain=True,
                    )
        self.assertEqual(fit.call_count, 4)
        self.assertTrue(all(call.kwargs["config"].fixed_epochs == 4 for call in fit.call_args_list))
        self.assertEqual(summary["epoch_mode"], "fixed")
        self.assertIsNone(summary["optimistic"])
        for number in (1, 2):
            fold = json.loads((self.output / f"fold-{number}.json").read_text())
            for run in fold["runs"]:
                self.assertEqual(run["evaluation_epoch"], 4)
                self.assertEqual(run["epochs_run"], 4)
                self.assertIsNone(run["optimistic"])
                checkpoint = torch.load(self.output / run["checkpoint"], weights_only=True)
                self.assertEqual(checkpoint["evaluation_epoch"], 4)
                history = json.loads(
                    (self.output / f"fold-{number}-seed-{run['seed']}-history.json").read_text()
                )
                self.assertEqual(len(history), 4)
        report = (self.output / "report.md").read_text(encoding="utf-8")
        self.assertIn("Main result: fixed", report)
        self.assertIn("not run", report)
        self.assertFalse(list(self.output.glob("*-optimistic-history.json")))

    def test_default_uncertain_rejection_does_not_create_output(self):
        with self.assertRaisesRegex(ValueError, "uncertain boundary"):
            run_baseline(self.manifest, self.output, context=self.context, config=self.config)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_changed_or_inputs_opened(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with patch("stratscout.tactic_training.TacticClipDataset") as loader:
            with self.assertRaises(FileExistsError):
                run_baseline(self.manifest, self.output, context=self.context)
        loader.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_test_split_is_rejected_before_building_model(self):
        text = self.manifest.read_text(encoding="utf-8").replace(",train,", ",test,")
        self.manifest.write_text(text, encoding="utf-8")
        with patch("stratscout.tactic_training.build_baseline") as build:
            with self.assertRaisesRegex(ValueError, "expected split 'train'"):
                run_baseline(self.manifest, self.output, context=self.context, config=self.config)
        build.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
