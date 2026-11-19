"""Synthetic regression checks for the temporal models' contracts, not measured accuracy."""

import csv
import io
import json
import tempfile
import unittest
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
from stratscout.tactic_data import MatchFold, TacticClipDataset
from stratscout.tactic_features import load_frame_features
from stratscout.tactic_model import TacticBaseline, file_sha256
from stratscout.tactic_positions import TOTALS, feature_names
from stratscout.tactic_temporal import (
    FIRST_GRU,
    GRU_DEFAULTS,
    SMALL_GRU,
    TIME_SCALE_S,
    TemporalGRU,
    load_temporal,
    mean_inputs,
    random_thirds_inputs,
    run_temporal,
    sequence_inputs,
    shuffled_label_check,
    thirds_inputs,
)
from stratscout.tactic_training import BaselineConfig, fit_classifier, run_baseline

MATCHES = (
    ("aurora", [0, 1, 2, 3, 3, 4]),
    ("furia", [0, 0, 1, 2, 3, 4]),
    ("mouz", [0, 1, 2, 3, 4, 4]),
    ("spirit", [0, 1, 1, 1, 2, 3, 4]),
)


def tiny_model(*, context, pretrained=True):
    return TacticBaseline(
        nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3
    ).to(context.device)


class TemporalInputTests(unittest.TestCase):
    def test_thirds_keep_time_order_and_drop_repeated_frames(self):
        steps = torch.arange(6, dtype=torch.float32)
        features = torch.stack([steps, steps * 10], dim=1).unsqueeze(0)  # (1, 6, 2)
        no_repeats = torch.zeros(1, 6, dtype=torch.bool)
        thirds = thirds_inputs(features, no_repeats)
        np.testing.assert_allclose(thirds[0].numpy(), [0.5, 5, 2.5, 25, 4.5, 45])
        reversed_thirds = thirds_inputs(features.flip(1), no_repeats)
        self.assertFalse(torch.equal(thirds, reversed_thirds))  # Order matters...
        torch.testing.assert_close(
            mean_inputs(features), mean_inputs(features.flip(1))
        )  # ...here not.
        # Frames 0,1,1,2,2,3: the repeats must not pull a third towards a duplicate.
        repeated = features[:, [0, 1, 1, 2, 2, 3]]
        mask = torch.tensor([[False, False, True, False, True, False]])
        np.testing.assert_allclose(thirds_inputs(repeated, mask)[0].numpy(), [0.5, 5, 2, 20, 3, 30])
        # One distinct frame: the empty middle and late parts fall back to the clip average.
        single = features[:, [4, 4, 4]]
        mask = torch.tensor([[False, True, True]])
        np.testing.assert_allclose(thirds_inputs(single, mask)[0].numpy(), [4, 40] * 3)

    def test_sequence_inputs_carry_real_seconds_and_pad_after_repeats(self):
        features = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + 1
        times = torch.tensor([[0.0, 3.0, 3.0, 9.0]], dtype=torch.float64)
        mask = torch.tensor([[False, False, True, False]])
        packed = sequence_inputs(features, times, mask)
        self.assertEqual(tuple(packed.shape), (1, 4, 5))
        np.testing.assert_allclose(packed[0, :3, :2].numpy(), [[1, 2], [3, 4], [7, 8]])
        np.testing.assert_allclose(packed[0, :, 2].numpy() * TIME_SCALE_S, [0, 3, 9, 0], atol=1e-5)
        np.testing.assert_allclose(packed[0, :, 3].numpy() * TIME_SCALE_S, [0, 3, 6, 0], atol=1e-5)
        np.testing.assert_array_equal(packed[0, :, 4].numpy(), [1, 1, 1, 0])
        self.assertTrue(torch.equal(packed[0, 3], torch.zeros(5)))
        # The same frames spread over a longer time are a different input.
        slower = sequence_inputs(features, times * 2, mask)
        torch.testing.assert_close(slower[:, :, :2], packed[:, :, :2])
        self.assertFalse(torch.equal(slower[:, :, 2:4], packed[:, :, 2:4]))

    def test_random_thirds_keeps_group_sizes_but_not_time_order(self):
        # Frame k has value 2**k, so each group's average shows exactly which frames it holds.
        n_frames, clips = 7, 12
        values = 2.0 ** torch.arange(n_frames, dtype=torch.float64)
        features = values.reshape(1, n_frames, 1).repeat(clips, 1, 1)
        mask = torch.zeros(clips, n_frames, dtype=torch.bool)
        mask[0, [2, 5]] = True  # Clip 0: five distinct frames, as after dropping repeats.
        ids = [f"clip-{i}" for i in range(clips)]
        result = random_thirds_inputs(features, mask, ids)
        ordered = thirds_inputs(features, mask)
        self.assertEqual(result.shape, ordered.shape)
        torch.testing.assert_close(random_thirds_inputs(features, mask, ids), result)

        def groups(row, distinct):
            # Recover each group's frames from its average and size (sizes as tensor_split).
            sizes = [len(part) for part in torch.tensor_split(torch.arange(len(distinct)), 3)]
            found = []
            for average, size in zip(row, sizes, strict=True):
                total = int(round(float(average) * size))
                found.append(sorted(k for k in distinct if total >> k & 1))
            return found

        for i in range(clips):
            distinct = [k for k in range(n_frames) if not mask[i, k]]
            found = groups(result[i], distinct)
            sizes = [len(part) for part in torch.tensor_split(torch.tensor(distinct), 3)]
            self.assertEqual([len(group) for group in found], sizes)  # Same sizes as thirds.
            self.assertEqual(sorted(sum(found, [])), distinct)  # Distinct frames once; no repeats.
        in_time_order = [
            groups(result[i], list(range(n_frames))) == groups(ordered[i], list(range(n_frames)))
            for i in range(1, clips)
        ]
        self.assertLess(sum(in_time_order), 3)  # Random, not time-ordered, for most clips.
        other = random_thirds_inputs(features, mask, [f"other-{i}" for i in range(clips)])
        self.assertFalse(torch.equal(other, result))  # The seed comes from the clip ID.
        torch.testing.assert_close(  # Clip 3's groups depend only on its own ID.
            random_thirds_inputs(features[3:4], mask[3:4], ["clip-3"])[0], result[3]
        )
        with self.assertRaisesRegex(ValueError, "one clip ID per clip"):
            random_thirds_inputs(features, mask, ids[:-1])


class TemporalGRUTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(6, 5, 4, generator=generator)
        times = torch.arange(5, dtype=torch.float64).repeat(6, 1) * 2.0
        self.mask = torch.zeros(6, 5, dtype=torch.bool)
        self.mask[0, 2] = self.mask[0, 4] = True  # Clip 0 has three real frames.
        self.features, self.times = features, times
        self.packed = sequence_inputs(features, times, self.mask)
        torch.manual_seed(1)
        self.model = TemporalGRU(feature_dim=4, projection=3, hidden=5, n_classes=5)
        self.model.train()
        self.model(self.packed)  # One training pass sets the standardisation statistics.
        self.model.eval()

    def logits(self, packed):
        with torch.no_grad():
            return self.model(packed)

    def test_output_shape_and_input_checks(self):
        self.assertEqual(tuple(self.logits(self.packed).shape), (6, 5))
        with self.assertRaisesRegex(ValueError, "sequence_inputs"):
            self.model(self.packed[:, :, :-1])
        empty = self.packed.clone()
        empty[1, :, -1] = 0
        with self.assertRaisesRegex(ValueError, "at least one real frame"):
            self.model(empty)

    def test_padding_is_ignored_but_order_and_elapsed_time_are_used(self):
        reference = self.logits(self.packed)
        padded = self.packed.clone()
        padded[0, 3:, :6] = 99.0  # Clip 0's padding rows; the real-frame flag stays 0.
        torch.testing.assert_close(self.logits(padded), reference, rtol=0, atol=0)
        slower = sequence_inputs(self.features, self.times * 3, self.mask)
        self.assertFalse(torch.allclose(self.logits(slower), reference))
        backwards = sequence_inputs(self.features.flip(1), self.times, self.mask.flip(1) & False)
        self.assertFalse(torch.allclose(self.logits(backwards)[1:], reference[1:]))

    def test_standardisation_comes_only_from_training_frames(self):
        mean = self.model.standardise.running_mean.clone()
        real = self.packed[:, :, -1] > 0
        torch.testing.assert_close(mean, self.packed[:, :, :4][real].mean(dim=0))
        self.assertEqual(sum(p.numel() for p in self.model.standardise.parameters()), 0)
        shifted = self.packed.clone()
        shifted[3:, :, :4] += 50  # Validation-style inputs far from the training statistics.
        before = self.logits(self.packed)[:3]
        torch.testing.assert_close(self.logits(shifted)[:3], before, rtol=0, atol=0)
        torch.testing.assert_close(self.model.standardise.running_mean, mean, rtol=0, atol=0)


class SmallGRUTests(unittest.TestCase):
    def test_defaults_are_the_final_gru_and_the_earlier_ones_stay_available(self):
        self.assertEqual(
            GRU_DEFAULTS, {"projection": 8, "hidden": 16, "dropout": 0.3, "readout": "mean-max"}
        )
        self.assertEqual(SMALL_GRU, {**GRU_DEFAULTS, "readout": "last"})
        self.assertEqual(
            FIRST_GRU, {"projection": 32, "hidden": 32, "dropout": 0.0, "readout": "last"}
        )

        def weights(model):
            return sum(p.numel() for p in model.parameters())

        def build(settings):
            return TemporalGRU(
                512,
                settings["projection"],
                settings["hidden"],
                5,
                settings["dropout"],
                settings["readout"],
            )

        final = TemporalGRU(512)
        self.assertEqual((final.project.out_features, final.gru.hidden_size), (8, 16))
        self.assertEqual(final.classifier.in_features, 32)  # Mean and max of 16 joined.
        recurrent = 512 * 8 + 8 + 3 * (16 * 10 + 16 * 16 + 32)
        self.assertEqual(weights(final), recurrent + 32 * 5 + 5)
        self.assertEqual(weights(final), 5613)
        self.assertEqual(weights(build(SMALL_GRU)), 5533)
        self.assertEqual(weights(build(FIRST_GRU)), 23109)
        with self.assertRaisesRegex(ValueError, "readout must be"):
            TemporalGRU(512, readout="attention")

    def test_mean_max_readout_pools_every_real_frame_and_ignores_padding(self):
        generator = torch.Generator().manual_seed(2)
        features = torch.randn(3, 6, 4, generator=generator)
        times = torch.arange(6, dtype=torch.float64).repeat(3, 1)
        mask = torch.zeros(3, 6, dtype=torch.bool)
        mask[0, 3:] = True  # Clip 0 has three real frames, then padding.
        packed = sequence_inputs(features, times, mask)
        torch.manual_seed(0)
        model = TemporalGRU(4, 3, 5, 5, dropout=0.3)
        model.train()
        model(packed)  # Sets the standardisation statistics.
        model.eval()
        with torch.no_grad():
            reference = model(packed)
            padded = packed.clone()
            padded[0, 3:, :6] = 99.0  # Padding rows; their real-frame flag stays 0.
            torch.testing.assert_close(model(padded), reference, rtol=0, atol=0)
            # The pooled readout is the mean and max of the GRU outputs at real frames only.
            seen = {}
            model.classifier.register_forward_hook(lambda m, i, o: seen.update(x=i[0]))
            model(packed)
            single = packed[:1, :3]  # Clip 0 alone, without its padding.
            projected = model.project(model.standardise(single[0, :, :4]))
            steps = torch.cat([projected, single[0, :, 4:6]], 1)[None]
            outputs, _ = model.gru(steps)
            expected = torch.cat([outputs[0].mean(0), outputs[0].amax(0)])
            torch.testing.assert_close(seen["x"][0], expected)
            # An early frame changes the readout even though the last state sees it only
            # through the recurrence: every real frame reaches the classifier directly.
            earlier = packed.clone()
            earlier[1, 0, :4] += 3.0
            self.assertFalse(torch.allclose(model(earlier)[1], reference[1]))

    def test_dropout_acts_only_in_training(self):
        generator = torch.Generator().manual_seed(0)
        packed = sequence_inputs(
            torch.randn(4, 5, 6, generator=generator),
            torch.arange(5, dtype=torch.float64).repeat(4, 1),
            torch.zeros(4, 5, dtype=torch.bool),
        )
        torch.manual_seed(0)
        model = TemporalGRU(6, 4, 5, 5, dropout=0.5)
        model.train()
        first, second = model(packed), model(packed)
        self.assertFalse(torch.allclose(first, second))  # Different dropout masks.
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(packed), model(packed), rtol=0, atol=0)

    def test_dropout_training_is_seeded_so_a_refit_repeats_a_longer_fit(self):
        generator = torch.Generator().manual_seed(1)
        packed = sequence_inputs(
            torch.randn(10, 4, 3, generator=generator),
            torch.arange(4, dtype=torch.float64).repeat(10, 1),
            torch.zeros(10, 4, dtype=torch.bool),
        )
        targets = torch.arange(5).repeat(2)
        fold = MatchFold(tuple(range(5)), tuple(range(5, 10)), ("a",), "b")

        def fit(epochs, seed=4):
            torch.manual_seed(99 + epochs)  # Outside state must not matter.
            return fit_classifier(
                packed,
                targets,
                fold,
                classes=TACTIC_CLASSES,
                config=BaselineConfig(epochs=8, patience=8, seeds=(seed,), fixed_epochs=epochs),
                seed=seed,
                context=RunContext(),
                build_head=lambda: TemporalGRU(3, 4, 5, 5, dropout=0.3),
            )

        short, long = fit(3), fit(6)
        losses = [row["train_loss"] for row in long[1]["history"]]
        self.assertEqual([row["train_loss"] for row in short[1]["history"]], losses[:3])
        again = fit(6)
        np.testing.assert_array_equal(again[2], long[2])
        self.assertNotEqual([r["train_loss"] for r in fit(6, seed=5)[1]["history"]], losses)


class ShuffledLabelTests(unittest.TestCase):
    def test_shuffles_stay_within_each_match_and_leave_targets_alone(self):
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 4])
        folds = (
            MatchFold(tuple(range(5, 12)), tuple(range(5)), ("b",), "a"),
            MatchFold(tuple(range(5)), tuple(range(5, 12)), ("a",), "b"),
        )
        original, seen = targets.clone(), []

        def fake_fit(inputs, labels, given, *, classes, config, context, **extra):
            seen.append((labels.clone(), extra))
            score = {"metrics": {"macro_f1": 0.25, "accuracy": 0.5}}
            return [[(None, score, None) for _ in config.seeds] for _ in given]

        with patch("stratscout.tactic_temporal.fit_folds", side_effect=fake_fit):
            rows = shuffled_label_check(
                torch.zeros(12, 2),
                targets,
                folds,
                classes=TACTIC_CLASSES,
                config=BaselineConfig(seeds=(0, 1)),
                context=RunContext(),
                build_head=None,
                count=3,
            )
        self.assertEqual([r["permutation"] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]["macro_f1"], 0.25)
        self.assertTrue(torch.equal(targets, original))
        for labels, extra in seen:
            self.assertEqual(extra, {})
            self.assertEqual(sorted(labels[:5].tolist()), sorted(original[:5].tolist()))
            self.assertEqual(sorted(labels[5:].tolist()), sorted(original[5:].tolist()))
        self.assertTrue(any(not torch.equal(labels, original) for labels, _ in seen))
        self.assertFalse(torch.equal(seen[0][0], seen[1][0]))


class TemporalRunTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "train" / "manifest.csv"
        self.manifest.parent.mkdir()
        self.output = self.root / "run"
        self.context = RunContext()
        # Three frames sampled four times: every clip repeats one frame.
        self.config = BaselineConfig(size=32, n_frames=4, epochs=3, patience=2, seeds=(3, 7))
        rows = []
        for match, labels in MATCHES:
            for i, label in enumerate(labels):
                clip_id = f"{match}-{i}"
                path = self.manifest.parent / f"{clip_id}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
                self.assertTrue(writer.isOpened())
                try:
                    for frame in range(3):
                        colour = (label * 40, 20 + frame * 60, 100 + i * 10)
                        writer.write(np.full((32, 32, 3), colour, np.uint8))
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
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        # An invalid neighbouring test manifest must never be opened.
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "manifest.csv").write_text("DO NOT READ", encoding="utf-8")

    def run_models(self, output, **kwargs):
        with patch("stratscout.tactic_features.build_baseline", side_effect=tiny_model):
            return run_temporal(
                self.manifest, output, context=self.context, config=self.config, **kwargs
            )

    def test_mean_model_reproduces_the_baseline_from_the_frame_cache(self):
        self.run_models(self.output, models=("mean",))
        baseline = self.root / "baseline"
        with patch("stratscout.tactic_features.build_baseline", side_effect=tiny_model):
            run_baseline(self.manifest, baseline, context=self.context, config=self.config)
        frames = torch.load(self.output / "frame-features.pt", weights_only=True)
        saved = torch.load(baseline / "features.pt", weights_only=True)
        self.assertEqual(tuple(frames["features"].shape), (25, 4, 3))
        self.assertTrue(torch.equal(frames["features"].mean(dim=1), saved["features"]))
        self.assertEqual(frames["clip_ids"], saved["clip_ids"])
        self.assertEqual(int(frames["repeat_mask"].sum()), 25)
        self.assertEqual(tuple(frames["timestamps_s"].shape), (25, 4))
        for name in ("predictions.csv", *(f"fold-{n}.json" for n in range(1, 5))):
            self.assertEqual(
                (self.output / "mean" / name).read_bytes(), (baseline / name).read_bytes(), name
            )
        self.assertEqual(
            json.loads((self.output / "mean" / "summary.json").read_text())["model"],
            json.loads((baseline / "summary.json").read_text())["model"],
        )

    def test_all_models_share_the_protocol_and_write_a_comparison(self):
        summary = self.run_models(self.output, shuffles=1)
        self.assertEqual(list(summary["models"]), ["mean", "thirds", "gru"])
        self.assertEqual(summary["epoch_mode"], "nested")
        self.assertEqual(summary["repeated_frames"], 25)
        self.assertEqual(summary, json.loads((self.output / "summary.json").read_text()))
        reference = [
            json.loads((self.output / "mean" / f"fold-{n}.json").read_text()) for n in range(1, 5)
        ]
        for name, weights in (("mean", 20), ("thirds", 50), ("gru", None)):
            folder = self.output / name
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["split"], "train")
            self.assertEqual(config["epoch_mode"], "nested")
            self.assertEqual(config["settings"]["seeds"], [3, 7])
            if weights:
                self.assertEqual(config["trainable_parameters"], weights)
            for number, expected in enumerate(reference, start=1):
                fold = json.loads((folder / f"fold-{number}.json").read_text())
                for key in ("validation_clip_ids", "train_clip_ids", "majority", "train_counts"):
                    self.assertEqual(fold[key], expected[key])
                for run in fold["runs"]:
                    self.assertEqual(run["epoch_selection"], "nested")
                    self.assertEqual(run["class_weights"], expected["runs"][0]["class_weights"])
                    self.assertTrue((folder / run["checkpoint"]).is_file())
            with (folder / "predictions.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 25 * 2)
            self.assertIn("development results", (folder / "report.md").read_text(encoding="utf-8"))
            shuffled = json.loads((folder / "shuffled-labels.json").read_text())
            self.assertEqual(len(shuffled["permutations"]), 1)
            self.assertEqual(
                shuffled["real"]["macro_f1"], summary["models"][name]["macro_f1"]["mean"]
            )
        gru = torch.load(self.output / "gru" / "fold-1-seed-3.pt", weights_only=True)
        self.assertIn("gru.weight_ih_l0", gru["classifier"])
        self.assertEqual(gru["model"], "resnet18-gru-linear")
        report = (self.output / "report.md").read_text(encoding="utf-8")
        for text in (
            "| Mean pooling + linear | 20 |",
            "| Temporal average pooling (3 segments) + linear classifier | 50 |",
            "| GRU |",
            "within noise",
            "Shuffled-label check",
            "Repeated frames in this run: 25.",
            "## F1 for each class",
            "| aurora | 6 |",
        ):
            self.assertIn(text, report)

    def test_random_thirds_run_reuses_the_cache_and_matches_thirds_protocol(self):
        self.run_models(self.output, models=("thirds",))
        control = self.root / "control"
        with patch("stratscout.tactic_features.build_baseline") as build:
            summary = run_temporal(
                self.manifest,
                control,
                context=self.context,
                config=self.config,
                models=("random-thirds",),
                features=self.output / "frame-features.pt",
                shuffles=1,
            )
        build.assert_not_called()
        entry = summary["models"]["random-thirds"]
        self.assertEqual(entry["trainable_parameters"], 50)  # Same head size as thirds.
        self.assertEqual(entry["model"], "resnet18-random-thirds-linear")
        config = json.loads((control / "random-thirds" / "config.json").read_text())
        self.assertIn("at random", config["temporal"]["pooling"])
        for number in range(1, 5):
            ordered = json.loads((self.output / "thirds" / f"fold-{number}.json").read_text())
            random = json.loads((control / "random-thirds" / f"fold-{number}.json").read_text())
            for key in ("train_clip_ids", "validation_clip_ids", "majority", "train_counts"):
                self.assertEqual(random[key], ordered[key])
        self.assertTrue((control / "random-thirds" / "shuffled-labels.json").is_file())
        self.assertIn(
            "| Random-segment average pooling + linear classifier (control) | 50 |",
            (control / "report.md").read_text(),
        )

    def test_saved_frame_features_are_reused_only_for_the_same_inputs(self):
        self.run_models(self.output, models=("mean",))
        cache = self.output / "frame-features.pt"
        again = self.root / "again"
        with patch("stratscout.tactic_features.build_baseline") as build:
            run_temporal(
                self.manifest,
                again,
                context=self.context,
                config=self.config,
                models=("thirds",),
                features=cache,
            )
        build.assert_not_called()
        self.assertFalse((again / "frame-features.pt").exists())
        self.assertTrue((again / "thirds" / "report.md").is_file())
        other = TacticClipDataset(self.manifest, context=self.context, n_frames=2, size=(32, 32))
        with self.assertRaisesRegex(ValueError, "n_frames does not match"):
            load_frame_features(cache, other)
        refused = self.root / "refused"
        with self.assertRaisesRegex(ValueError, "n_frames does not match"):
            run_temporal(
                self.manifest,
                refused,
                context=self.context,
                config=BaselineConfig(size=32, n_frames=2, epochs=3, patience=2, seeds=(3,)),
                features=cache,
            )
        self.assertFalse(refused.exists())

    def test_cache_is_refused_when_a_clip_changes_but_the_manifest_does_not(self):
        self.run_models(self.output, models=("mean",))
        cache = self.output / "frame-features.pt"
        manifest_bytes = self.manifest.read_bytes()
        clip = self.manifest.parent / "furia-2.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        try:
            for frame in range(3):
                writer.write(np.full((32, 32, 3), (200, 10, frame * 80), np.uint8))
        finally:
            writer.release()
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)
        dataset = TacticClipDataset(self.manifest, context=self.context, n_frames=4, size=(32, 32))
        with self.assertRaisesRegex(ValueError, "furia-2 has changed since it was cached"):
            load_frame_features(cache, dataset)
        refused = self.root / "refused"
        with self.assertRaisesRegex(ValueError, "furia-2 has changed"):
            run_temporal(
                self.manifest,
                refused,
                context=self.context,
                config=self.config,
                models=("thirds",),
                features=cache,
            )
        self.assertFalse(refused.exists())

    def assert_reload_matches_saved_predictions(self, run, frames, expected_encoder):
        """Every saved head, reloaded from disk, repeats its saved validation predictions."""
        rows = {}
        index = {clip_id: i for i, clip_id in enumerate(frames["clip_ids"])}
        for folder in sorted(p for p in run.iterdir() if p.is_dir()):
            with (folder / "predictions.csv").open(encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    rows.setdefault((folder.name, row["fold"], row["seed"]), []).append(row)
            for checkpoint in sorted(folder.glob("fold-*-seed-*.pt")):
                with patch("stratscout.tactic_temporal.build_baseline", side_effect=tiny_model):
                    model = load_temporal(checkpoint, context=self.context)
                self.assertEqual(model.name, folder.name)
                self.assertEqual(model.encoder is not None, expected_encoder)
                _, fold, _, seed = checkpoint.stem.split("-")
                probabilities = model.predict(frames).softmax(1)
                for row in rows[(folder.name, fold, seed)]:
                    saved = torch.tensor([float(row[f"p_{c}"]) for c in TACTIC_CLASSES])
                    torch.testing.assert_close(
                        probabilities[index[row["clip_id"]]], saved, rtol=0, atol=1e-6
                    )
        return len(rows)

    def test_every_temporal_checkpoint_reloads_with_identical_predictions(self):
        self.run_models(self.output, models=("mean", "thirds", "random-thirds", "gru"))
        frames = torch.load(self.output / "frame-features.pt", weights_only=True)
        self.assertEqual(
            self.assert_reload_matches_saved_predictions(self.output, frames, True), 32
        )
        # The encoder is found in the run folder and must be the one the heads were fitted on.
        checkpoint = self.output / "gru" / "fold-1-seed-3.pt"
        encoder = self.output / "encoder.pt"
        state = torch.load(encoder, weights_only=True)
        state["0.running_mean"] += 1
        torch.save(state, encoder)
        with self.assertRaisesRegex(ValueError, "does not match this checkpoint"):
            with patch("stratscout.tactic_temporal.build_baseline", side_effect=tiny_model):
                load_temporal(checkpoint, context=self.context)

    def test_position_checkpoints_reload_with_identical_predictions(self):
        with patch(
            "stratscout.tactic_temporal.extract_position_features", side_effect=self.fake_positions
        ):
            run_temporal(
                self.manifest,
                self.output,
                context=self.context,
                config=self.config,
                models=("mean", "thirds", "gru"),
                inputs="positions",
                drop_channels=("bomb_badge",),
            )
        frames = torch.load(self.output / "position-features.pt", weights_only=True)
        self.assertEqual(frames["features"].shape[2], 260)  # Reload drops the channel itself.
        self.assertEqual(
            self.assert_reload_matches_saved_predictions(self.output, frames, False), 24
        )
        model = load_temporal(self.output / "mean" / "fold-1-seed-3.pt", context=self.context)
        self.assertEqual(model.inputs, "positions")
        self.assertEqual(model.head.in_features, 260 - 65)

    def fake_positions(self, dataset, *, context):
        """Deterministic stand-in for the detector: one T per frame, moving across cells."""
        self.assertIsNone(dataset.size)  # Positions are read at native size.
        names = feature_names()
        clips = len(dataset)
        features = torch.zeros(clips, dataset.n_frames, len(names))
        for i, record in enumerate(dataset.records):
            label = TACTIC_CLASSES.index(record.label)
            for k in range(dataset.n_frames):
                features[i, k, names.index(f"t_observed[{label},{k}]")] = 1
                features[i, k, names.index(f"bomb_badge[{k},{label}]")] = 1
                features[i, k, len(names) - len(TOTALS)] = 1
        samples = [
            {
                "clip_id": r.clip_id,
                "clip_sha256": file_sha256(r.path),
                "timestamps_s": [0.0, 0.1, 0.2, 0.3][: dataset.n_frames],
                "repeat_mask": [False] * dataset.n_frames,
            }
            for r in dataset.records
        ]
        return {
            "features": features,
            "samples": samples,
            "encoder_sha256": None,
            "feature_names": names,
        }

    def test_position_inputs_build_their_own_cache_and_can_drop_a_channel(self):
        with patch(
            "stratscout.tactic_temporal.extract_position_features", side_effect=self.fake_positions
        ):
            with patch("stratscout.tactic_features.build_baseline") as build:
                summary = run_temporal(
                    self.manifest,
                    self.output,
                    context=self.context,
                    config=self.config,
                    models=("mean", "gru"),
                    inputs="positions",
                )
        build.assert_not_called()  # No ResNet for position inputs.
        self.assertEqual(summary["inputs"], "positions")
        self.assertEqual(summary["feature_width"], 260)
        self.assertTrue((self.output / "position-features.pt").is_file())
        self.assertFalse((self.output / "frame-features.pt").exists())
        config = json.loads((self.output / "mean" / "config.json").read_text())
        self.assertEqual(config["model"], "positions-mean-linear")
        self.assertEqual(config["trainable_parameters"], 261 * 5)
        self.assertEqual(config["inputs"]["width"], 260)
        self.assertEqual(len(config["inputs"]["feature_names"]), 260)
        self.assertIn("Inputs: positions (260 numbers", (self.output / "report.md").read_text())
        cache = self.output / "position-features.pt"
        ablation = self.root / "ablation"
        with patch("stratscout.tactic_temporal.extract_position_features") as extract:
            dropped = run_temporal(
                self.manifest,
                ablation,
                context=self.context,
                config=self.config,
                models=("mean",),
                inputs="positions",
                features=cache,
                drop_channels=("bomb_badge",),
            )
        extract.assert_not_called()
        self.assertEqual(dropped["feature_width"], 260 - 65)
        names = json.loads((ablation / "mean" / "config.json").read_text())["inputs"][
            "feature_names"
        ]
        self.assertFalse(any(n.startswith("bomb_badge") for n in names))
        # A position cache is never accepted as image features, and the reverse.
        with self.assertRaisesRegex(ValueError, "does not match this run"):
            run_temporal(
                self.manifest,
                self.root / "wrong",
                context=self.context,
                config=self.config,
                models=("mean",),
                features=cache,
            )
        self.assertFalse((self.root / "wrong").exists())
        with self.assertRaisesRegex(ValueError, "only to position features"):
            run_temporal(
                self.manifest,
                self.root / "wrong",
                context=self.context,
                drop_channels=("bomb_badge",),
            )

    def test_bad_requests_fail_before_any_output(self):
        for kwargs, message in (
            ({"models": ("gru", "gru")}, "distinct models"),
            ({"models": ("tcn",)}, "distinct models"),
            ({"shuffles": -1}, "non-negative"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                run_temporal(self.manifest, self.output, context=self.context, **kwargs)
            self.assertFalse(self.output.exists())
        text = self.manifest.read_text(encoding="utf-8").replace(",train,", ",test,")
        self.manifest.write_text(text, encoding="utf-8")
        with patch("stratscout.tactic_features.build_baseline") as build:
            with self.assertRaisesRegex(ValueError, "expected split 'train'"):
                run_temporal(self.manifest, self.output, context=self.context, config=self.config)
        build.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_other_folds_run_writes_the_tcn_rule_and_a_seed_ensemble(self):
        summary = self.run_models(self.output, models=("mean", "gru"), epoch_mode="other-folds")
        self.assertEqual(summary["epoch_mode"], "other-folds")
        for name in ("mean", "gru"):
            config = json.loads((self.output / name / "config.json").read_text())
            self.assertEqual(config["epoch_mode"], "other-folds")
            self.assertIn("TCN", config["selection"])
            epochs = {}
            for number in range(1, 5):
                fold = json.loads((self.output / name / f"fold-{number}.json").read_text())
                for run in fold["runs"]:
                    self.assertEqual(run["epoch_selection"], "other-folds")
                    self.assertNotIn(
                        fold["validation_match_id"], run["selection"]["source_validation_match_ids"]
                    )
                    epochs.setdefault(number, set()).add(run["evaluation_epoch"])
            self.assertTrue(all(len(e) == 1 for e in epochs.values()))  # One epoch per fold.
            model_summary = json.loads((self.output / name / "summary.json").read_text())
            self.assertEqual(model_summary["seed_ensemble"]["clips"], 25)
            self.assertIn("slightly optimistic", " ".join(model_summary["limitations"]))
            self.assertIn("Seed ensemble", (self.output / name / "report.md").read_text())
            self.assertEqual(
                summary["models"][name]["seed_ensemble"]["macro_f1"],
                model_summary["seed_ensemble"]["macro_f1"],
            )

    def test_first_gru_run_reloads_from_a_config_saved_before_these_options_existed(self):
        summary = self.run_models(self.output, models=("gru",), gru=FIRST_GRU)
        config_path = self.output / "gru" / "config.json"
        config = json.loads(config_path.read_text())
        self.assertEqual(
            {k: config["temporal"][k] for k in FIRST_GRU}, FIRST_GRU
        )  # Each run saves its sizes.
        self.assertGreater(summary["models"]["gru"]["trainable_parameters"], 0)
        del config["temporal"]["dropout"]  # As in runs saved before these options existed.
        del config["temporal"]["readout"]
        config_path.write_text(json.dumps(config))
        frames = torch.load(self.output / "frame-features.pt", weights_only=True)
        self.assertEqual(self.assert_reload_matches_saved_predictions(self.output, frames, True), 8)
        default = self.run_models(self.root / "small", models=("gru",))
        small = json.loads((self.root / "small" / "gru" / "config.json").read_text())
        self.assertEqual({k: small["temporal"][k] for k in GRU_DEFAULTS}, GRU_DEFAULTS)
        self.assertLess(
            default["models"]["gru"]["trainable_parameters"],
            summary["models"]["gru"]["trainable_parameters"],
        )

    def test_cli_passes_models_features_and_shuffles(self):
        base = ["tactic-temporal", "--manifest", "unused.csv", "--output", "unused-output"]
        with patch("stratscout.tactic_temporal.run_temporal") as run:
            main(base)
            self.assertEqual(run.call_args.kwargs["models"], ("mean", "thirds", "gru"))
            self.assertEqual(run.call_args.kwargs["shuffles"], 0)
            self.assertIsNone(run.call_args.kwargs["features"])
            self.assertEqual(run.call_args.kwargs["config"].seeds, (0, 1, 2))
            self.assertEqual(run.call_args.kwargs["epoch_mode"], "nested")
            self.assertEqual(run.call_args.kwargs["gru"], GRU_DEFAULTS)
            main(
                base
                + ["--gru-projection", "32", "--gru-hidden", "32", "--gru-dropout", "0"]
                + ["--gru-readout", "last"]
            )
            self.assertEqual(run.call_args.kwargs["gru"], FIRST_GRU)
            main(base + ["--epoch-mode", "other-folds"])
            self.assertEqual(run.call_args.kwargs["epoch_mode"], "other-folds")
            main(base + ["--models", "gru", "--shuffles", "5", "--features", "cache.pt"])
            self.assertEqual(run.call_args.kwargs["models"], ("gru",))
            self.assertEqual(run.call_args.kwargs["shuffles"], 5)
            self.assertEqual(run.call_args.kwargs["features"], Path("cache.pt"))
            main(base + ["--models", "random-thirds"])
            self.assertEqual(run.call_args.kwargs["models"], ("random-thirds",))
        with patch("stratscout.tactic_temporal.run_temporal") as run:
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                main(base + ["--models", "tcn"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
