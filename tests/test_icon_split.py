"""Splitting overlapping same-team icons, removing death markers and the count evaluation."""

import unittest

import cv2
import numpy as np

from stratscout.config import Config
from stratscout.context import RunContext
from stratscout.count_eval import all_alive_windows, counts, impossible_rate, score
from stratscout.icon_split import SplitParams, allocate, find_icons_split, icons_in_blob

T_YELLOW = (255, 225, 0)
CT_CYAN = (40, 200, 240)
HUES = {"T": 27.0, "CT": 94.0}


def frame_with(icons, markers=(), size=(200, 200)):
    """RGB frame with icon discs of radius 9 and a white arrow to the right of each, plus
    arrowless ✕ markers, at (x, y, colour)."""
    f = np.full((*size, 3), 70, np.uint8)
    for x, y, colour in icons:
        cv2.circle(f, (x, y), 9, colour, -1)
        arrow = np.array([[x + 9, y - 4], [x + 9, y + 4], [x + 16, y]], np.int32)
        cv2.fillPoly(f, [arrow], (255, 255, 255))
    for x, y, colour in markers:
        cv2.line(f, (x - 8, y - 8), (x + 8, y + 8), colour, 5)
        cv2.line(f, (x - 8, y + 8), (x + 8, y - 8), colour, 5)
    return f


class IconSplitTests(unittest.TestCase):
    def setUp(self):
        self.ctx = RunContext(Config())
        self.params = SplitParams(single_area=250.0)

    def detect(self, frame, **kw):
        return find_icons_split(frame, HUES, None, self.params, context=self.ctx, **kw)

    def test_single_icons_are_not_split(self):
        pts = self.detect(frame_with([(50, 50, CT_CYAN), (120, 120, CT_CYAN)]))
        self.assertEqual(len(pts), 2)

    def test_two_overlapping_icons_become_two_detections(self):
        pts = self.detect(frame_with([(100, 94, CT_CYAN), (100, 108, CT_CYAN)]))
        self.assertEqual(len(pts), 2)
        ys = sorted(p[1] * 200 for p in pts)
        self.assertLess(ys[0], 101)
        self.assertGreater(ys[1], 101)

    def test_two_touching_icons_are_not_counted_as_three(self):
        # Area alone reads two barely touching discs like three overlapping ones.
        pts = self.detect(frame_with([(100, 90, CT_CYAN), (100, 108, CT_CYAN)]))
        self.assertEqual(len(pts), 2)

    def test_touching_icons_of_different_teams_stay_separate(self):
        pts = self.detect(frame_with([(100, 100, CT_CYAN), (82, 100, T_YELLOW)]))
        self.assertEqual(sorted(p[2] for p in pts), [0.0, 1.0])

    def test_death_markers_without_an_arrow_are_not_players(self):
        frame = frame_with([(50, 50, T_YELLOW)], markers=[(120, 120, T_YELLOW)])
        self.assertEqual(len(self.detect(frame)), 1)

    def test_team_never_exceeds_five(self):
        icons = [(30 + 34 * i, 60, CT_CYAN) for i in range(5)] + [(30, 140, CT_CYAN)]
        self.assertEqual(len(self.detect(frame_with(icons))), 5)
        self.assertEqual(len(self.detect(frame_with(icons), t_reserved=1)), 5)

    def test_area_rule_grows_with_area(self):
        p = SplitParams(single_area=250.0, split_ratio=1.15, extra_area=0.6, method="area")
        self.assertEqual(icons_in_blob(250, p), 1)
        self.assertEqual(icons_in_blob(400, p), 2)
        self.assertEqual(icons_in_blob(250 + 4 * 150, p), 5)

    def test_cap_takes_icons_from_the_weakest_fit(self):
        blobs = [
            {"area": 500, "white": 40, "fits": [0.4, 0.84, 0.85]},  # 3 barely beats 2
            {"area": 500, "white": 40, "fits": [0.4, 0.86]},
            {"area": 250, "white": 40, "fits": [0.9, 0.6]},
        ]
        uncapped = SplitParams(single_area=250.0, max_per_team=None)
        self.assertEqual(allocate(blobs, uncapped), [3, 2, 1])
        self.assertEqual(allocate(blobs, SplitParams(single_area=250.0)), [2, 2, 1])


class CountEvalTests(unittest.TestCase):
    def test_windows_skip_partial_rounds(self):
        rounds = [
            {"round": 1, "start": 0.0, "freeze": 0.0, "partial": True, "end": 60.0},
            {"round": 2, "start": 60.0, "freeze": 20.0, "partial": False, "end": 200.0},
        ]
        self.assertEqual(all_alive_windows(rounds), [("freeze", 60.0, 80.0), ("early", 80.0, 88.0)])

    def test_hidden_carrier_counts_for_t_and_scores_are_per_phase(self):
        params = SplitParams(single_area=250.0)
        single = {"area": 250, "white": 40, "fits": [0.9, 0.6]}
        pair = {"area": 420, "white": 60, "fits": [0.5, 0.85, 0.7]}
        rec = {"phase": "early", "carried": True, "baseline": [3, 5]}
        rec["blobs"] = [{"team": 0, **single}] * 4 + [{"team": 1, **single}] * 3
        rec["blobs"] += [{"team": 1, **pair}]
        self.assertEqual(counts(rec), [4, 5])
        self.assertEqual(counts(rec, params), [5, 5])
        result = score([rec], params)
        self.assertEqual(result["early_T"]["exact5"], 1.0)
        self.assertEqual(result["mean_exact5"], 1.0)

    def test_impossible_rate(self):
        rows = [{"v": [5, 5]}, {"v": [6, 4]}, {"v": [3, 2]}, {"v": [5, 7]}]
        self.assertEqual(impossible_rate(rows, "v"), 0.5)


if __name__ == "__main__":
    unittest.main()
