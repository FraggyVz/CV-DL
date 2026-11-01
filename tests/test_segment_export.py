import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stratscout.segment_export import clip_rows, cut_clip, export_segments, load_label_files
from stratscout.segments import SegmentFile


def segment(label, start, end, status="pending", **extra):
    return {"start_s": start, "end_s": end, "label": label, "status": status, **extra}


class ExportTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.labels = self.root / "labels"
        self.recordings = self.root / "recordings"

    def make_match(self, name, match_id, seconds=6, fps=10):
        """A prepared match folder whose frame k has brightness k, so cuts can be checked."""
        folder = self.recordings / name
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"match_id": match_id}))
        video = folder / f"minimap-{fps}fps.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
        for k in range(seconds * fps):
            writer.write(np.full((32, 32, 3), 4 * k, np.uint8))
        writer.release()
        return folder

    def label(self, name, match_id, segments, reviewer="jx"):
        store = SegmentFile(
            self.labels / f"{name}.{reviewer}.segments.json",
            match_id=match_id,
            match_folder=name,
            video="minimap-10fps.mp4",
            video_fps=10,
            reviewer=reviewer,
        )
        for seg in segments:
            store.add(seg)
        return store


class ReviewTests(ExportTestCase):
    def test_review_marks_labelled_pending_segments(self):
        store = self.label(
            "furia", "f", [segment("A_DIRECT", 1, 2), segment("RETAKE", 3, 4, site="A")]
        )
        store.review([1, 2], "gin")
        self.assertEqual({s["status"] for s in store.segments}, {"reviewed"})
        self.assertEqual(store.segments[0]["reviewed_by"], "gin")

    def test_review_refuses_undecided_and_changes_nothing(self):
        store = self.label(
            "furia", "f", [segment("A_DIRECT", 1, 2), segment(None, 3, 4, undecided=True)]
        )
        with self.assertRaises(ValueError):
            store.review([1, 2], "gin")
        self.assertEqual(store.segments[0]["status"], "pending")

    def test_reviewed_segments_cannot_be_deleted(self):
        store = self.label("furia", "f", [segment("A_DIRECT", 1, 2)])
        store.review([1], "gin")
        with self.assertRaises(ValueError):
            store.remove(1)


class RowTests(ExportTestCase):
    def test_only_reviewed_classes_by_default_and_whole_matches_split(self):
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)]).review([1], "gin")
        navi = self.label(
            "navi",
            "n",
            [
                segment("B_SPLIT", 1, 2),
                segment("B_DIRECT", 3, 4),
                segment(None, 5, 6, undecided=True),
                segment(None, 7, 8, status="excluded", exclusion_reason="unusable footage"),
            ],
        )
        navi.review([1], "gin")
        rows = clip_rows(load_label_files(self.labels), ["navi"])
        self.assertEqual(
            [(r["match_folder"], r["split"]) for r in rows], [("furia", "train"), ("navi", "test")]
        )
        pending = clip_rows(load_label_files(self.labels), ["navi"], include_pending=True)
        self.assertEqual(len(pending), 3)  # undecided and excluded never export

    def test_two_reviewers_for_one_match_need_a_choice(self):
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)])
        self.label("furia", "f", [segment("A_SPLIT", 1, 2)], reviewer="gin")
        with self.assertRaises(ValueError):
            load_label_files(self.labels)
        self.assertEqual(len(load_label_files(self.labels, "gin")), 1)


class CutTests(ExportTestCase):
    def test_cut_uses_the_labeller_frame_timing(self):
        folder = self.make_match("furia", "f")
        out = self.root / "clip.mp4"
        self.assertEqual(cut_clip(folder / "minimap-10fps.mp4", 1.0, 2.0, out), 11)
        cap = cv2.VideoCapture(str(out))
        ok, first = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertAlmostEqual(float(first.mean()), 40, delta=6)  # frame 10 = 1.0 s

    def test_export_writes_manifest_clips_and_summary(self):
        self.make_match("furia", "f")
        self.make_match("navi", "n")
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)]).review([1], "gin")
        self.label("navi", "n", [segment("RETAKE", 2, 3, site="B")]).review([1], "gin")
        out = self.root / "dataset"
        summary = export_segments(self.labels, self.recordings, out, ["navi"])
        manifests = {}
        for split in ("train", "test"):
            with (out / split / "manifest.csv").open(encoding="utf-8") as handle:
                manifests[split] = list(csv.DictReader(handle))
        self.assertEqual([r["match_folder"] for r in manifests["train"]], ["furia"])
        self.assertEqual([r["match_folder"] for r in manifests["test"]], ["navi"])
        test_clip = out / "test" / manifests["test"][0]["clip"]
        self.assertTrue(test_clip.exists())
        self.assertFalse(list((out / "train").rglob("navi*")))
        self.assertEqual(summary["counts"]["test"]["RETAKE"], 1)
        with self.assertRaises(FileExistsError):
            export_segments(self.labels, self.recordings, out, ["navi"])

    def test_misspelled_test_match_is_refused(self):
        self.make_match("furia", "f")
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)]).review([1], "gin")
        with self.assertRaisesRegex(ValueError, "matched no labelled match"):
            export_segments(self.labels, self.recordings, self.root / "out", ["nav1"])
        self.assertFalse((self.root / "out").exists())

    def test_export_refuses_labels_from_a_different_video(self):
        self.make_match("furia", "other-id")
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)]).review([1], "gin")
        with self.assertRaisesRegex(ValueError, "different video"):
            export_segments(self.labels, self.recordings, self.root / "out", ["furia"])

    def test_nothing_reviewed_is_an_error(self):
        self.make_match("furia", "f")
        self.label("furia", "f", [segment("A_DIRECT", 1, 2)])
        with self.assertRaisesRegex(ValueError, "no usable segments"):
            export_segments(self.labels, self.recordings, self.root / "out", ["furia"])


if __name__ == "__main__":
    unittest.main()
