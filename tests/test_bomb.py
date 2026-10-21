"""Bomb badge shape detection and dropped-candidate checks on generated frames."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stratscout.bomb import (
    BombStateTracker,
    carrier_rim,
    carrier_visible,
    dropped_candidates,
    find_bomb,
    write_bomb_scan,
)
from stratscout.config import Config
from stratscout.context import RunContext
from stratscout.segmentation import blob_shape, find_icons_teams

T_YELLOW = (255, 225, 0)  # OpenCV hue ~27, like spectator-radar T icons
CT_CYAN = (40, 200, 240)
BADGE_ORANGE = (240, 140, 30)  # hue ~15


def radar(badge=(150, 100), t_icons=((60, 60), (200, 160)), ct_icons=((90, 180),), carrier=False):
    """Grey map with round numbered icons and one rectangular C4 badge (RGB). With carrier,
    a T icon and its white arrow are drawn under the badge, as the spectator radar does."""
    f = np.full((240, 260, 3), 70, np.uint8)
    if carrier and badge is not None:
        x, y = badge
        cv2.circle(f, (x, y), 9, T_YELLOW, -1)
        arrow = np.array([[x - 18, y], [x - 9, y - 5], [x - 9, y + 5]], np.int32)
        cv2.fillPoly(f, [arrow], (255, 255, 255))
    for x, y in t_icons:
        cv2.circle(f, (x, y), 9, T_YELLOW, -1)
        cv2.putText(f, "7", (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
    for x, y in ct_icons:
        cv2.circle(f, (x, y), 9, CT_CYAN, -1)
    if badge is not None:
        x, y = badge
        cv2.rectangle(f, (x - 11, y - 7), (x + 10, y + 7), BADGE_ORANGE, -1)
        cv2.rectangle(f, (x + 1, y - 4), (x + 7, y), (150, 80, 20), -1)  # the badge's dark panel
    return f


class BombShapeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = RunContext(Config())

    def test_shape_separates_rectangle_from_circle(self):
        rect = np.ones((14, 22), bool)
        circle = np.zeros((19, 19), np.uint8)
        cv2.circle(circle, (9, 9), 9, 1, -1)
        r, c = blob_shape(rect), blob_shape(circle.astype(bool))
        self.assertGreater(r["corner_fill"], 0.9)
        self.assertLess(c["corner_fill"], 0.3)
        self.assertGreater(r["aspect"], 1.15)
        self.assertAlmostEqual(c["aspect"], 1.0)

    def test_find_bomb_returns_badge_not_player_icons(self):
        b = find_bomb(radar(), context=self.ctx)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["x"], 150, delta=2)
        self.assertAlmostEqual(b["y"], 100, delta=2)
        self.assertIsNone(find_bomb(radar(badge=None), context=self.ctx))

    def test_colour_split_finds_badge_touching_its_carrier(self):
        touching = radar(t_icons=((132, 100),))  # T circle overlaps the badge's left edge
        b = find_bomb(touching, context=self.ctx)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["x"], 150, delta=3)
        self.assertIsNone(find_bomb(touching, shape_only=True, context=self.ctx))

    def test_shape_only_still_finds_an_isolated_badge(self):
        self.assertIsNotNone(find_bomb(radar(), shape_only=True, context=self.ctx))

    def test_colour_baseline_no_longer_counts_badge_as_t(self):
        ctx = RunContext(Config(icon_area=(6, 450)))
        hues = {"T": 27.0, "CT": 94.0}
        pts = find_icons_teams(radar(), hues, context=ctx)
        self.assertEqual(sum(p[2] == 0 for p in pts), 2)
        self.assertEqual(sum(p[2] == 1 for p in pts), 1)

    def rim(self, frame):
        return carrier_rim(frame, find_bomb(frame, context=self.ctx), context=self.ctx)

    def test_carrier_under_badge_shows_at_its_rim(self):
        rim = self.rim(radar(carrier=True))
        self.assertTrue(carrier_visible(rim))
        self.assertGreater(rim["yellow"], 0)
        self.assertGreater(rim["white"], 0)

    def test_lone_badge_and_nearby_teammate_show_no_carrier(self):
        self.assertFalse(carrier_visible(self.rim(radar(t_icons=((40, 40),)))))
        # A teammate 30 px away is outside the rim band, unlike the old 40 px proximity rule.
        self.assertFalse(carrier_visible(self.rim(radar(t_icons=((150, 130),)))))


class DroppedCandidateTests(unittest.TestCase):
    @staticmethod
    def rows(spec):
        return [
            {"frame": i, "seconds": i / 5, "x": x, "y": y, "carried": int(c)}
            for i, (x, y, c) in enumerate(spec)
        ]

    def test_still_lone_badge_becomes_one_candidate(self):
        spec = [(100 + i, 50, True) for i in range(10)] + [(110, 50, False)] * 15
        runs = dropped_candidates(self.rows(spec), fps=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["start_frame"], runs[0]["end_frame"]), (10, 24))

    def test_still_carrier_and_short_or_moving_runs_are_ignored(self):
        carried = [(110, 50, True)] * 20
        short = [(110, 50, False)] * 6
        moving = [(100 + 5 * i, 50, False) for i in range(20)]
        for spec in (carried, short, moving):
            self.assertEqual(dropped_candidates(self.rows(spec), fps=5), [])

    def test_brief_detection_gaps_are_bridged(self):
        spec = [(110, 50, False)] * 6 + [(None, None, False)] * 2 + [(110, 50, False)] * 6
        self.assertEqual(len(dropped_candidates(self.rows(spec), fps=5)), 1)


class BombStateTrackerTests(unittest.TestCase):
    def test_states_follow_the_dropped_rule(self):
        tr = BombStateTracker(fps=5, min_seconds=2)
        self.assertEqual(tr.update(110, 50, True), "carried")
        states = [tr.update(110, 50, False) for _ in range(10)]
        self.assertEqual(states[:9], ["alone"] * 9)
        self.assertEqual(states[9], "dropped")
        self.assertEqual(tr.update(110, 50, True), "carried")

    def test_moving_or_long_missing_badge_restarts_the_still_timer(self):
        tr = BombStateTracker(fps=5, min_seconds=2)
        for _ in range(9):
            tr.update(110, 50, False)
        self.assertEqual(tr.update(120, 50, False), "alone")
        for _ in range(9):
            tr.update(120, 50, False)
        for _ in range(3):
            self.assertEqual(tr.update(None, None, False), "missing")
        self.assertEqual(tr.update(120, 50, False), "alone")


class BombScanOutputTests(unittest.TestCase):
    def test_scan_writes_new_folder_and_refuses_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "minimap.avi"
            frame = cv2.cvtColor(radar(t_icons=((40, 40),)), cv2.COLOR_RGB2BGR)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (260, 240))
            for _ in range(15):
                writer.write(frame)
            writer.release()
            out = Path(tmp) / "scan"
            ctx = RunContext(Config())
            with contextlib.redirect_stdout(io.StringIO()):
                summary = write_bomb_scan(video, out, context=ctx)
            self.assertEqual(summary["frames_with_badge"], 15)
            self.assertEqual(summary["dropped_candidates"], 1)
            self.assertTrue((out / "track.csv").exists())
            self.assertEqual(len(list(out.glob("candidate-*.png"))), 1)
            with self.assertRaises(ValueError):
                write_bomb_scan(video, out, context=ctx)


if __name__ == "__main__":
    unittest.main()
