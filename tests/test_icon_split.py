"""Splitting overlapping same-team icons and the all-alive count evaluation."""

import unittest

import cv2
import numpy as np

from stratscout.config import Config
from stratscout.context import RunContext
from stratscout.count_eval import all_alive_windows, counts, score
from stratscout.icon_split import SplitParams, find_icons_split, icons_in_blob

T_YELLOW = (255, 225, 0)
CT_CYAN = (40, 200, 240)
HUES = {"T": 27.0, "CT": 94.0}


def frame_with(icons, size=(200, 200)):
    """RGB frame with filled icon discs of radius 9 at (x, y, colour)."""
    f = np.full((*size, 3), 70, np.uint8)
    for x, y, colour in icons:
        cv2.circle(f, (x, y), 9, colour, -1)
    return f


class IconSplitTests(unittest.TestCase):
    def setUp(self):
        self.ctx = RunContext(Config())
        self.params = SplitParams(single_area=250.0)

    def detect(self, frame):
        return find_icons_split(frame, HUES, None, self.params, context=self.ctx)

    def test_single_icons_are_not_split(self):
        pts = self.detect(frame_with([(50, 50, CT_CYAN), (120, 120, CT_CYAN)]))
        self.assertEqual(len(pts), 2)

    def test_two_overlapping_icons_become_two_detections(self):
        pts = self.detect(frame_with([(100, 96, CT_CYAN), (100, 106, CT_CYAN)]))
        self.assertEqual(len(pts), 2)
        ys = sorted(p[1] * 200 for p in pts)
        self.assertLess(ys[0], 101)
        self.assertGreater(ys[1], 101)

    def test_touching_icons_of_different_teams_stay_separate(self):
        pts = self.detect(frame_with([(100, 100, CT_CYAN), (114, 100, T_YELLOW)]))
        self.assertEqual(sorted(p[2] for p in pts), [0.0, 1.0])

    def test_icon_count_grows_with_area(self):
        p = SplitParams(single_area=250.0, split_ratio=1.15, extra_area=0.6)
        self.assertEqual(icons_in_blob(250, p), 1)
        self.assertEqual(icons_in_blob(400, p), 2)
        self.assertEqual(icons_in_blob(250 + 4 * 150, p), 5)


class CountEvalTests(unittest.TestCase):
    def test_windows_skip_partial_rounds(self):
        rounds = [
            {"round": 1, "start": 0.0, "freeze": 0.0, "partial": True, "end": 60.0},
            {"round": 2, "start": 60.0, "freeze": 20.0, "partial": False, "end": 200.0},
        ]
        self.assertEqual(all_alive_windows(rounds), [("freeze", 60.0, 80.0), ("early", 80.0, 88.0)])

    def test_hidden_carrier_counts_for_t_and_scores_are_per_phase(self):
        params = SplitParams(single_area=250.0)
        rec = {"phase": "early", "carried": True, "baseline": [3, 5]}
        rec["areas"] = [(0, 250)] * 4 + [(1, 250)] * 3 + [(1, 400)]
        self.assertEqual(counts(rec), [4, 5])
        self.assertEqual(counts(rec, params), [5, 5])
        result = score([rec], params)
        self.assertEqual(result["early_T"]["exact5"], 1.0)
        self.assertEqual(result["mean_exact5"], 1.0)


if __name__ == "__main__":
    unittest.main()
