"""Synthetic media checks for shared inputs, timing and match isolation, not accuracy."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stratscout.config import TACTIC_CLASSES
from stratscout.context import RunContext
from stratscout.segment_export import COLUMNS, cut_clip


class TacticDataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "train" / "manifest.csv"
        self.manifest.parent.mkdir()
        self.context = RunContext()

    def video(self, path, count=11, fps=10):
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(count):
                # Red-dominant BGR frame, with increasing green to expose temporal order.
                frame = np.full((24, 32, 3), (10, 20 + i * 5, 200), np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def row(self, clip_id="f1", match_id="furia", *, count=11, fps=10, **overrides):
        relative = f"clips/{clip_id}.mp4"
        self.video(self.manifest.parent / relative, count, fps)
        row = dict.fromkeys(COLUMNS, "")
        row.update(
            clip_id=clip_id,
            split="train",
            label="A_DIRECT",
            match_id=match_id,
            match_folder=match_id,
            start_s=12,
            end_s=12 + (count - 1) / fps,
            duration_s=(count - 1) / fps,
            frames=count,
            status="reviewed",
            boundary_uncertain=False,
            definition_version="pilot-v1",
            clip=relative,
            source_video="unavailable-on-teammates-computer.mp4",
        )
        row.update(overrides)
        return row

    def write(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def dataset(self, **kwargs):
        return TacticClipDataset(self.manifest, context=self.context, **kwargs)

    def test_rgb_uint8_order_and_elapsed_time_survive_sampling(self):
        self.write([self.row()])
        sample = self.dataset(n_frames=3)[0]
        self.assertEqual(sample["frames"].shape, (3, 24, 32, 3))
        self.assertEqual(sample["frames"].dtype, np.uint8)
        np.testing.assert_array_equal(sample["frame_indices"], [0, 5, 10])
        np.testing.assert_allclose(sample["timestamps_s"], [0, 0.5, 1])
        np.testing.assert_array_equal(sample["repeat_mask"], [False, False, False])
        colour = sample["frames"].mean(axis=(1, 2))
        np.testing.assert_allclose(colour[:, 0], 200, atol=8)
        np.testing.assert_allclose(colour[:, 2], 10, atol=8)
        self.assertTrue(np.all(np.diff(colour[:, 1]) > 15))
        self.assertEqual(sample["duration_s"], 1)
        self.assertEqual(sample["metadata"]["source_start_s"], 12)

    def test_equal_frame_counts_do_not_erase_different_clip_timing(self):
        self.write([self.row("fast", fps=10), self.row("slow", fps=5)])
        data = self.dataset(n_frames=3)
        np.testing.assert_allclose(data[0]["timestamps_s"], [0, 0.5, 1])
        np.testing.assert_allclose(data[1]["timestamps_s"], [0, 1, 2])
        self.assertEqual(data[1]["duration_s"], 2)

    def test_short_clip_repeats_real_frames_and_marks_repeats(self):
        self.write([self.row(count=3)])
        sample = self.dataset(n_frames=5)[0]
        np.testing.assert_array_equal(sample["frame_indices"], [0, 0, 1, 2, 2])
        np.testing.assert_allclose(sample["timestamps_s"], [0, 0, 0.1, 0.2, 0.2])
        np.testing.assert_array_equal(sample["repeat_mask"], [False, True, False, False, True])
        np.testing.assert_array_equal(sample["frames"][0], sample["frames"][1])

    def test_resize_and_pytorch_batches_work_for_all_five_targets(self):
        self.write([self.row(f"c{i}", label=label) for i, label in enumerate(TACTIC_CLASSES)])
        data = self.dataset(n_frames=2, size=(12, 16))
        batch = next(iter(DataLoader(data, batch_size=5)))
        self.assertEqual(batch["frames"].shape, (5, 2, 12, 16, 3))
        self.assertEqual(batch["frames"].dtype, torch.uint8)
        self.assertEqual(batch["target"].dtype, torch.int64)
        self.assertEqual(batch["target"].tolist(), list(range(5)))
        self.assertEqual(batch["timestamps_s"].shape, (5, 2))
        self.assertEqual(data.classes, tuple(TACTIC_CLASSES))
        self.assertIs(data.context, self.context)

    def test_exporter_inclusive_endpoints_are_compatible(self):
        source = self.root / "source.mp4"
        self.video(source, count=31)
        row = self.row(start_s=0.56, end_s=1.64, duration_s=1.08)
        row["frames"] = cut_clip(source, 0.56, 1.64, self.manifest.parent / row["clip"])
        self.write([row])
        sample = self.dataset(n_frames=3)[0]
        self.assertEqual(row["frames"], 11)
        np.testing.assert_allclose(sample["timestamps_s"], [0, 0.5, 1])
        self.assertEqual(sample["duration_s"], 1.08)

    def test_folds_group_all_variants_and_ignore_neighbouring_test_data(self):
        self.write(
            [
                self.row("f10", "furia", match_folder="furia-10fps"),
                self.row("s10", "spirit"),
                self.row("f5", "furia", fps=5, match_folder="furia-5fps"),
            ]
        )
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "manifest.csv").write_text("deliberately invalid: never read this")
        with patch("stratscout.tactic_data.cv2.VideoCapture") as capture:
            data = self.dataset(n_frames=2)
            folds = whole_match_folds(data)
            capture.assert_not_called()
        self.assertEqual(len(folds), 2)
        first, second = folds
        self.assertEqual(first.validation_match_id, "furia")
        self.assertEqual(first.validation_indices, (0, 2))
        self.assertEqual(first.train_indices, (1,))
        self.assertEqual(second.train_indices, first.validation_indices)
        self.assertEqual(second.validation_indices, first.train_indices)
        for fold in folds:
            self.assertNotIn(fold.validation_match_id, fold.train_match_ids)
            self.assertFalse(set(fold.train_indices) & set(fold.validation_indices))
            self.assertEqual(set(fold.train_indices + fold.validation_indices), {0, 1, 2})
        batch = next(iter(DataLoader(Subset(data, first.validation_indices), batch_size=2)))
        self.assertEqual(batch["metadata"]["match_id"], ["furia", "furia"])

    def test_test_split_requires_explicit_selection_and_cannot_make_folds(self):
        self.write([self.row(split="test")])
        with self.assertRaisesRegex(ValueError, "expected split"):
            self.dataset()
        data = self.dataset(split="test")
        with self.assertRaisesRegex(ValueError, "never test"):
            whole_match_folds(data)

    def test_one_match_cannot_make_validation_folds(self):
        self.write([self.row()])
        with self.assertRaisesRegex(ValueError, "at least two"):
            whole_match_folds(self.dataset())

    def test_uncertain_boundaries_require_explicit_exclusion_and_keep_audit_ids(self):
        self.write([self.row("uncertain", boundary_uncertain=True), self.row("agreed")])
        with self.assertRaisesRegex(ValueError, "uncertain boundary"):
            self.dataset()
        with self.assertWarnsRegex(UserWarning, "uncertain"):
            data = self.dataset(exclude_uncertain=True)
        self.assertEqual(data.excluded_clip_ids, ("uncertain",))
        self.assertEqual([r.clip_id for r in data.records], ["agreed"])
        self.assertEqual(len(data), 1)

    def test_invalid_manifest_contracts_are_refused(self):
        invalid = [
            {"label": "A_RUSH"},
            {"status": "pending"},
            {"status": "excluded"},
            {"split": "test"},
            {"match_id": ""},
            {"definition_version": ""},
            {"boundary_uncertain": "maybe"},
            {"start_s": "nan"},
            {"end_s": "inf"},
            {"duration_s": 10},
            {"frames": 0},
            {"frames": "1.5"},
            {"clip": ""},
            {"clip": "../escape.mp4"},
            {"clip": str(self.root / "absolute.mp4")},
        ]
        for override in invalid:
            with self.subTest(override=override):
                self.write([self.row(**override)])
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_missing_clip_is_refused_without_falling_back_to_source(self):
        row = self.row()
        (self.manifest.parent / row["clip"]).unlink()
        self.write([row])
        with self.assertRaises(FileNotFoundError):
            self.dataset()

    def test_duplicate_and_conflicting_provenance_is_refused(self):
        first = self.row("f1")
        second = self.row("f2")
        variants = [
            {"clip_id": first["clip_id"]},
            {"clip": first["clip"]},
            {"match_id": "different-id"},
            {"definition_version": "different-rules"},
        ]
        for override in variants:
            with self.subTest(override=override):
                self.write([first, {**second, **override}])
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_missing_columns_empty_and_malformed_manifests_are_refused(self):
        self.manifest.write_text("clip_id\na\n")
        with self.assertRaisesRegex(ValueError, "missing manifest columns"):
            self.dataset()
        self.write([])
        with self.assertRaisesRegex(ValueError, "no eligible"):
            self.dataset()
        with self.manifest.open("a") as handle:
            handle.write("incomplete,row\n")
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.dataset()

    def test_video_manifest_mismatches_and_decode_failure_are_refused(self):
        row = self.row()
        for overrides, message in [
            ({"frames": 12}, "frame count"),
            ({"end_s": 14, "duration_s": 2}, "video timing"),
        ]:
            with self.subTest(overrides=overrides):
                self.write([{**row, **overrides}])
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset()[0]
        self.write([row])
        with patch("stratscout.tactic_data.cv2.VideoCapture") as factory:
            cap = factory.return_value
            cap.isOpened.return_value = True
            cap.get.side_effect = [10.0, 11.0]
            cap.read.return_value = (False, None)
            with self.assertRaisesRegex(ValueError, "cannot decode"):
                self.dataset()[0]
            cap.release.assert_called_once()

    def test_invalid_sampling_settings_are_refused(self):
        self.write([self.row()])
        for kwargs in (
            {"n_frames": 0},
            {"n_frames": 1.5},
            {"n_frames": True},
            {"size": (0, 32)},
            {"size": (32,)},
            {"split": "validation"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.dataset(**kwargs)


if __name__ == "__main__":
    unittest.main()
