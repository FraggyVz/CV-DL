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
    inner_match_folds,
    nested_epoch,
    other_folds_epochs,
    run_baseline,
    seed_ensemble,
    selection_mode,
    summarise_runs,
    training_class_weights,
)


def three_match_folds():
    """Matches a, b and c with five clips each: one clip per class."""
    matches = {name: tuple(range(start, start + 5)) for name, start in zip("abc", (0, 5, 10))}
    return tuple(
        MatchFold(
            tuple(i for other, clips in matches.items() if other != name for i in clips),
            clips,
            tuple(other for other in matches if other != name),
            name,
        )
        for name, clips in matches.items()
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

    def test_fewer_than_two_folds_are_rejected_before_fitting(self):
        one = (MatchFold((0, 1, 2, 3, 4), (0, 1, 2, 3, 4), ("furia",), "furia"),)
        for folds in ((), one):
            with (
                self.subTest(folds=len(folds)),
                patch("stratscout.tactic_training.fit_classifier") as fit,
            ):
                with self.assertRaisesRegex(ValueError, "at least two folds"):
                    fit_folds(
                        torch.eye(5),
                        torch.arange(5),
                        folds,
                        classes=TACTIC_CLASSES,
                        config=BaselineConfig(),
                        context=self.context,
                    )
                fit.assert_not_called()

    def test_selection_mode_follows_fold_count_and_fixed_epochs(self):
        self.assertEqual(selection_mode(BaselineConfig(), 2), "cross-fold")
        for count in (3, 4):
            self.assertEqual(selection_mode(BaselineConfig(), count), "nested")
            self.assertEqual(selection_mode(BaselineConfig(fixed_epochs=5), count), "fixed")

    def test_inner_folds_leave_out_one_training_match_and_never_the_scored_match(self):
        folds = three_match_folds()
        inner = inner_match_folds(folds[2], folds)  # Scored match: c, clips 10-14.
        self.assertEqual([f.validation_match_id for f in inner], ["a", "b"])
        self.assertEqual(inner[0].train_indices, tuple(range(5, 10)))
        self.assertEqual(inner[0].validation_indices, tuple(range(5)))
        self.assertEqual(inner[1].train_match_ids, ("a",))
        for part in inner:
            self.assertTrue(set(part.train_indices + part.validation_indices) <= set(range(10)))
        with self.assertRaisesRegex(ValueError, "every training match"):
            inner_match_folds(folds[2], folds[1:])

    def test_nested_epoch_scores_the_mean_inner_curve_with_ties_and_patience(self):
        folds = three_match_folds()
        curves = {  # Per inner validation match: (macro-F1, loss) at each epoch.
            "a": [(0.2, 1.0), (0.8, 1.0), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0), (1.0, 1.0)],
            "b": [(0.6, 1.0), (0.2, 1.0), (0.6, 0.5), (0.4, 1.0), (0.0, 1.0), (1.0, 1.0)],
        }

        def pilot(x, y, fold, *, classes, config, seed, context):
            self.assertEqual(config.patience, config.epochs)  # Inner fits return full curves.
            self.assertIsNone(config.fixed_epochs)
            history = [
                dict(epoch=i, validation_macro_f1=score, validation_loss=loss)
                for i, (score, loss) in enumerate(curves[fold.validation_match_id], start=1)
            ]
            return None, {"history": history}, None

        def choose(patience):
            with patch("stratscout.tactic_training.fit_classifier", side_effect=pilot):
                return nested_epoch(
                    torch.eye(5).repeat(3, 1),
                    torch.arange(5).repeat(3),
                    folds[2],
                    folds,
                    classes=TACTIC_CLASSES,
                    config=BaselineConfig(epochs=6, patience=patience),
                    seed=1,
                    context=self.context,
                )

        # Means are 0.4, 0.5, 0.5, 0.4, 0.0, 1.0: epoch 3 ties epoch 2 and wins on lower loss.
        epoch, selection = choose(patience=2)  # Stops at epoch 5, before the late peak.
        self.assertEqual(epoch, 3)
        self.assertEqual(selection["epoch"], 3)
        self.assertAlmostEqual(selection["mean_validation_macro_f1"], 0.5)
        self.assertEqual(selection["inner_validation_match_ids"], ["a", "b"])
        self.assertEqual(len(selection["history"]), 5)
        self.assertEqual(selection["history"][1]["validation_macro_f1"], {"a": 0.8, "b": 0.2})
        epoch, selection = choose(patience=3)  # A longer wait reaches epoch 6.
        self.assertEqual(epoch, 6)
        self.assertEqual(len(selection["history"]), 6)

    def test_nested_fit_never_uses_the_scored_match_for_its_epoch_or_weights(self):
        folds = three_match_folds()
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(15, 6, generator=generator)
        targets = torch.arange(5).repeat(3)
        config = BaselineConfig(epochs=8, patience=3, seeds=(1, 4))

        def fit(x, y):
            return fit_folds(
                x, y, folds, classes=TACTIC_CLASSES, config=config, context=self.context
            )

        results = fit(features, targets)
        changed_features, changed_targets = features.clone(), targets.clone()
        changed_features[10:] += 50  # Match c is fold 3's scored match.
        changed_targets[10:] = (changed_targets[10:] + 2) % 5
        changed = fit(changed_features, changed_targets)
        self.assertEqual(len(results), 3)
        for (head, run, probs), (other_head, other_run, _) in zip(
            results[2], changed[2], strict=True
        ):
            self.assertEqual(run["epoch_selection"], "nested")
            self.assertIsNone(run["epoch_source_fold"])
            self.assertEqual(run["selection"], other_run["selection"])
            self.assertEqual(run["evaluation_epoch"], run["selection"]["epoch"])
            self.assertEqual(run["epochs_run"], run["selection"]["epoch"])
            self.assertEqual(run["selection"]["inner_validation_match_ids"], ["a", "b"])
            self.assertEqual(run["optimistic"]["epoch_selection"], "early-stopping")
            for key, tensor in head.state_dict().items():
                torch.testing.assert_close(tensor, other_head.state_dict()[key], rtol=0, atol=0)
            reference, _, reference_probs = fit_classifier(
                features,
                targets,
                folds[2],
                classes=TACTIC_CLASSES,
                config=replace(config, fixed_epochs=run["selection"]["epoch"]),
                seed=run["seed"],
                context=self.context,
            )
            for key, tensor in head.state_dict().items():
                torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=0, atol=0)
            np.testing.assert_array_equal(probs, reference_probs)

    def test_other_folds_epoch_ignores_its_own_curve_and_takes_the_earliest_tie(self):
        curves = np.array(
            [
                [0.1, 0.9, 0.2, 0.2],  # a's own peak is epoch 2 ...
                [0.5, 0.1, 0.5, 0.1],
                [0.3, 0.1, 0.3, 0.9],
            ]
        )
        # ... but a uses mean(b, c) = [0.4, 0.1, 0.4, 0.5] -> epoch 4.
        # b uses mean(a, c) = [0.2, 0.5, 0.25, 0.55] -> 4; c uses mean(a, b) -> 2.
        self.assertEqual(other_folds_epochs(curves), [4, 4, 2])
        for own in (np.zeros(4), np.ones(4), np.array([9.0, 0, 0, 0])):
            changed = curves.copy()
            changed[0] = own
            self.assertEqual(other_folds_epochs(changed)[0], 4)  # Own curve never matters.
        tied = np.array([[0.0, 0.0, 0.0], [0.2, 0.7, 0.7], [0.4, 0.4, 0.4]])
        self.assertEqual(other_folds_epochs(tied)[2], 2)  # Earliest of the tied epochs 2 and 3.
        with self.assertRaisesRegex(ValueError, "at least two folds"):
            other_folds_epochs(np.ones((1, 3)))

    def test_other_folds_fit_scores_each_fold_at_the_other_folds_seed_mean_epoch(self):
        folds = three_match_folds()
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(15, 6, generator=generator)
        targets = torch.arange(5).repeat(3)
        config = BaselineConfig(epochs=5, patience=1, seeds=(3, 7))
        # Validation macro-F1 per epoch for each (fold, seed). Seed-mean curves:
        # a [0.2, 0.2, 0.9, 0.2, 0.2]; b [0.6, 0.1, 0.1, 0.1, 0.6]; c [0.1, 0.5, 0.1, 0.5, 0.1].
        curves = {
            ("a", 3): [0.2, 0.1, 0.8, 0.3, 0.2],
            ("a", 7): [0.2, 0.3, 1.0, 0.1, 0.2],
            ("b", 3): [0.7, 0.1, 0.1, 0.1, 0.5],
            ("b", 7): [0.5, 0.1, 0.1, 0.1, 0.7],
            ("c", 3): [0.1, 0.4, 0.1, 0.6, 0.1],
            ("c", 7): [0.1, 0.6, 0.1, 0.4, 0.1],
        }
        expected = {"a": 1, "b": 3, "c": 3}  # mean(b,c): ties at 1 and 5 -> 1; others -> 3.

        def full_curve_or_refit(x, y, fold, *, classes, config, seed, context):
            if config.fixed_epochs is None:
                self.assertEqual(config.patience, config.epochs)  # Full curves, no stopping.
                history = [
                    {"epoch": e, "train_loss": 1.0, "validation_macro_f1": score}
                    for e, score in enumerate(curves[(fold.validation_match_id, seed)], start=1)
                ]
                return None, {"seed": seed, "best_epoch": 1, "history": history}, None
            return fit_classifier(
                x, y, fold, classes=classes, config=config, seed=seed, context=context
            )

        with patch(
            "stratscout.tactic_training.fit_classifier", side_effect=full_curve_or_refit
        ) as fit:
            results = fit_folds(
                features,
                targets,
                folds,
                classes=TACTIC_CLASSES,
                config=config,
                context=self.context,
                epoch_mode="other-folds",
            )
        self.assertEqual(fit.call_count, 12)  # Six full curves, six exact-epoch refits.
        for fold, runs in zip(folds, results, strict=True):
            match = fold.validation_match_id
            for head, run, probs in runs:
                self.assertEqual(run["epoch_selection"], "other-folds")
                self.assertEqual(run["evaluation_epoch"], expected[match])  # Same for every seed.
                self.assertEqual(run["epochs_run"], expected[match])
                self.assertNotIn(match, run["selection"]["source_validation_match_ids"])
                reference, _, reference_probs = fit_classifier(
                    features,
                    targets,
                    fold,
                    classes=TACTIC_CLASSES,
                    config=replace(config, fixed_epochs=expected[match]),
                    seed=run["seed"],
                    context=self.context,
                )
                for key, tensor in head.state_dict().items():
                    torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=0, atol=0)
                np.testing.assert_array_equal(probs, reference_probs)

    def test_other_folds_mode_is_named_and_refuses_fixed_epochs(self):
        config = BaselineConfig(seeds=(0,))
        self.assertEqual(selection_mode(config, 4), "nested")
        self.assertEqual(selection_mode(config, 4, "other-folds"), "other-folds")
        self.assertEqual(selection_mode(config, 2, "other-folds"), "other-folds")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            selection_mode(replace(config, fixed_epochs=5), 4, "other-folds")
        with self.assertRaisesRegex(ValueError, "epoch_mode must be one of"):
            selection_mode(config, 4, "cross")

    def test_seed_ensemble_averages_probabilities_then_pools_the_clips(self):
        classes = ("X", "Y")

        def row(clip, seed, truth, p):
            return {"clip_id": clip, "seed": seed, "truth": truth, "p_X": p, "p_Y": 1 - p}

        predictions = [
            row("one", 0, "X", 0.9),
            row("one", 1, "X", 0.2),  # Seed 1 alone is wrong; the mean 0.55 is right.
            row("two", 0, "Y", 0.4),
            row("two", 1, "Y", 0.45),
        ]
        result = seed_ensemble(predictions, classes, (0, 1))
        self.assertEqual(result["clips"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["pooled_seed_macro_f1"][0], 1.0)
        self.assertLess(result["pooled_seed_macro_f1"][1], 1.0)
        with self.assertRaisesRegex(ValueError, "one prediction per seed"):
            seed_ensemble(predictions[:3], classes, (0, 1))

    def test_cli_epoch_modes_and_conflicts(self):
        base = ["tactic-baseline", "--manifest", "unused.csv", "--output", "unused-output"]
        for flags, expected in (([], None), (["--cross-fold"], None), (["--fixed-epochs", "5"], 5)):
            with self.subTest(flags=flags), patch("stratscout.tactic_training.run_baseline") as run:
                main(base + flags)
                self.assertEqual(run.call_args.kwargs["config"].fixed_epochs, expected)
                self.assertEqual(run.call_args.kwargs["epoch_mode"], "nested")
        with patch("stratscout.tactic_training.run_baseline") as run:
            main(base + ["--epoch-mode", "other-folds"])
            self.assertEqual(run.call_args.kwargs["epoch_mode"], "other-folds")
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
        self.write_manifest((("furia", [0, 0, 1, 2, 3, 4]), ("spirit", [0, 1, 1, 1, 2, 3, 4])))
        # An invalid neighbouring test manifest must never be opened.
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "manifest.csv").write_text("DO NOT READ", encoding="utf-8")

    def write_manifest(self, matches):
        rows = []
        for match, labels in matches:
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

    def test_end_to_end_outputs_majorities_exclusions_and_offline_reload(self):
        with patch("stratscout.tactic_features.build_baseline", side_effect=tiny_model):
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
        with patch("stratscout.tactic_features.build_baseline", side_effect=tiny_model):
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

    def test_four_matches_use_nested_selection_and_keep_the_output_files(self):
        self.write_manifest(
            (
                ("aurora", [0, 1, 2, 3, 3, 4]),
                ("furia", [0, 0, 1, 2, 3, 4]),
                ("mouz", [0, 1, 2, 3, 4, 4]),
                ("spirit", [0, 1, 1, 1, 2, 3, 4]),
            )
        )
        with patch("stratscout.tactic_features.build_baseline", side_effect=tiny_model):
            with self.assertWarns(UserWarning):
                summary = run_baseline(
                    self.manifest,
                    self.output,
                    context=self.context,
                    config=self.config,
                    exclude_uncertain=True,
                )
        self.assertEqual(summary["epoch_mode"], "nested")
        self.assertEqual(
            json.loads((self.output / "config.json").read_text())["epoch_mode"], "nested"
        )
        self.assertIn("not an independent test score", summary["limitations"][0])
        names = ["aurora", "furia", "mouz", "spirit"]
        total = 0
        for number, name in enumerate(names, start=1):
            fold = json.loads((self.output / f"fold-{number}.json").read_text())
            self.assertEqual(fold["validation_match_id"], name)
            self.assertEqual(fold["train_match_ids"], [n for n in names if n != name])
            self.assertTrue(set(fold["train_clip_ids"]).isdisjoint(fold["validation_clip_ids"]))
            total += len(fold["validation_clip_ids"])
            for run in fold["runs"]:
                selection = run["selection"]
                self.assertEqual(run["epoch_selection"], "nested")
                self.assertEqual(run["evaluation_epoch"], selection["epoch"])
                self.assertNotIn(name, selection["inner_validation_match_ids"])
                self.assertEqual(len(selection["inner_validation_match_ids"]), 3)
                self.assertNotIn("history", selection)
                stem = f"fold-{number}-seed-{run['seed']}"
                curve = json.loads((self.output / f"{stem}-selection-history.json").read_text())
                self.assertEqual(set(curve[0]["validation_macro_f1"]), set(names) - {name})
                self.assertEqual(
                    selection["mean_validation_macro_f1"],
                    curve[selection["epoch"] - 1]["mean_validation_macro_f1"],
                )
                self.assertEqual(
                    len(json.loads((self.output / f"{stem}-history.json").read_text())),
                    selection["epoch"],
                )
                self.assertTrue((self.output / f"{stem}-optimistic-history.json").is_file())
                saved = torch.load(self.output / run["checkpoint"], weights_only=True)
                self.assertEqual(saved["epoch_selection"], "nested")
                self.assertIsNone(saved["epoch_source_fold"])
        self.assertEqual(total, 25)
        with (self.output / "predictions.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 25 * 2)
        report = (self.output / "report.md").read_text(encoding="utf-8")
        self.assertIn("Main result: nested", report)
        self.assertIn("| training matches |", report)
        for name in names:
            self.assertIn(f"## Per-class F1: {name}", report)

    def test_nested_run_rejects_an_inner_fold_without_every_class_before_writing(self):
        # Class 4 is missing from mouz: fine for the outer folds, not for an inner one.
        self.write_manifest(
            (
                ("furia", [0, 1, 2, 3, 4]),
                ("mouz", [0, 1, 2, 3, 3]),
                ("spirit", [0, 1, 2, 3, 4]),
            )
        )
        with patch("stratscout.tactic_features.build_baseline") as build:
            with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "no examples"):
                run_baseline(
                    self.manifest,
                    self.output,
                    context=self.context,
                    config=self.config,
                    exclude_uncertain=True,
                )
        build.assert_not_called()
        self.assertFalse(self.output.exists())

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
        with patch("stratscout.tactic_features.build_baseline") as build:
            with self.assertRaisesRegex(ValueError, "expected split 'train'"):
                run_baseline(self.manifest, self.output, context=self.context, config=self.config)
        build.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
