"""Round label files and datasets must be keyed by match, not by video filename."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stratscout.annotations import (
    RoundLabeller,
    build_prepared_rounds_dataset,
    match_rounds_json,
    prepared_match_id,
)
from stratscout.config import Config
from stratscout.context import RunContext


class MatchRoundsJsonTests(unittest.TestCase):
    def test_matches_with_same_variant_name_get_separate_files(self):
        paths = {
            match_rounds_json(Path("recordings/furia-falcons")),
            match_rounds_json(Path("recordings/spirit-falcons")),
        }
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.parent == Path("labels") for p in paths))

    def test_path_is_named_after_the_match_folder(self):
        path = match_rounds_json(Path("data/recordings/navi-falcons"), labels_dir="out")
        self.assertEqual(path, Path("out") / "navi-falcons.rounds.json")


class PreparedRoundsDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.labels = self.root / "labels"

    def match(self, name, source_hash=None, label="A_RUSH"):
        folder = self.root / "recordings" / name
        folder.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(folder / "minimap-10fps.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (40, 36)
        )
        self.assertTrue(writer.isOpened())
        for _ in range(40):
            writer.write(np.full((36, 40, 3), 90, np.uint8))
        writer.release()
        if source_hash:
            (folder / "manifest.json").write_text(json.dumps({"match_id": source_hash}))
        round_ = {"round_idx": 0, "start_s": 0.5, "end_s": 3.0, "label": label}
        round_ |= {"winner": "T", "plant_s": None, "plant_site": None, "ct_label": None}
        match_rounds_json(folder, self.labels).parent.mkdir(exist_ok=True)
        match_rounds_json(folder, self.labels).write_text(json.dumps({"rounds": [round_]}))
        return folder

    def test_match_id_comes_from_manifest_else_folder(self):
        self.assertEqual(prepared_match_id(self.match("furia", "abc123")), "abc123")
        self.assertEqual(prepared_match_id(self.match("spirit")), "spirit")

    def test_each_match_keeps_its_own_labels_and_identity(self):
        folders = [
            self.match("furia", "hash-f", "A_RUSH"),
            self.match("spirit", "hash-s", "B_SPLIT"),
        ]
        out = self.root / "rounds.npz"
        context = RunContext(Config(seq_len=4, labels_dir=str(self.labels)))
        build_prepared_rounds_dataset(folders, str(out), fps=10, context=context)
        data = np.load(out)
        self.assertEqual(list(data["match_ids"]), ["hash-f", "hash-s"])
        self.assertEqual(list(data["labels"]), ["A_RUSH", "B_SPLIT"])
        self.assertEqual(data["frames"].shape, (2, 4, 36, 40, 3))


class EstimatedRoundTests(unittest.TestCase):
    GUIDES = [
        {"round": 1, "start": 0.0, "freeze": 0.0, "partial": True, "end": 46.4},
        {"round": 2, "start": 46.4, "freeze": 20.8, "partial": False, "end": 131.2},
        {"round": 3, "start": 131.2, "freeze": 20.4, "partial": False, "end": 264.4},
    ]

    def labeller(self, guides):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = Path(tmp.name) / "minimap-10fps.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (40, 36))
        self.assertTrue(writer.isOpened())
        for _ in range(20):
            writer.write(np.zeros((36, 40, 3), np.uint8))
        writer.release()
        out = Path(tmp.name) / "labels" / "match.rounds.json"
        labeller = RoundLabeller(
            str(video), (0, 0, 40, 36), str(out), round_guides=guides, context=RunContext()
        )
        self.addCleanup(labeller.cap.release)
        return labeller

    def commit(self, labeller, start, end):
        labeller.start, labeller.end, labeller.label = start, end, "DEFAULT"
        labeller._commit()
        return labeller.rounds[-1]

    def test_skipping_the_partial_round_keeps_game_numbering(self):
        labeller = self.labeller(self.GUIDES)
        first = self.commit(labeller, 67.2, 120.0)
        second = self.commit(labeller, 151.6, 250.0)
        self.assertEqual([first["round_idx"], second["round_idx"]], [0, 1])
        self.assertEqual([first["estimated_round"], second["estimated_round"]], [2, 3])

    def test_without_guides_the_label_format_is_unchanged(self):
        entry = self.commit(self.labeller(None), 67.2, 120.0)
        self.assertNotIn("estimated_round", entry)


if __name__ == "__main__":
    unittest.main()
