"""Synthetic checks for the position features' contracts, not detector accuracy."""

import unittest

from stratscout.tactic_positions import (
    CHANNELS,
    GRID,
    TOTALS,
    channel_columns,
    feature_names,
    frame_positions,
)

WIDTH, HEIGHT = 494, 500


def cell(names, channel, row, col):
    return names.index(f"{channel}[{row},{col}]")


class FramePositionTests(unittest.TestCase):
    def setUp(self):
        self.names = feature_names()

    def test_layout_is_four_channel_grids_then_totals(self):
        self.assertEqual(len(self.names), len(CHANNELS) * GRID * GRID + len(TOTALS))
        self.assertEqual(self.names[0], "t_observed[0,0]")
        self.assertEqual(self.names[-4:], list(TOTALS))

    def test_counts_go_in_their_own_cells_and_channels(self):
        found = {
            "players": [(10, 10, "T"), (20, 20, "T"), (480, 490, "CT"), (250, 250, "CT")],
            "carrier": (130, 70),
            "bomb": {"x": 130, "y": 70},
        }
        row = frame_positions(found, WIDTH, HEIGHT)
        self.assertEqual(row[cell(self.names, "t_observed", 0, 0)], 2)
        self.assertEqual(row[cell(self.names, "ct_observed", 7, 7)], 1)
        self.assertEqual(row[cell(self.names, "ct_observed", 4, 4)], 1)
        self.assertEqual(row[cell(self.names, "t_inferred_carrier", 1, 2)], 1)
        self.assertEqual(row[cell(self.names, "bomb_badge", 1, 2)], 1)
        totals = dict(zip(TOTALS, row[-4:], strict=True))
        # The inferred carrier is not counted as a seen T.
        self.assertEqual(
            totals, {"t_seen": 2, "ct_seen": 2, "t_inferred_present": 1, "bomb_badge_present": 1}
        )
        self.assertEqual(row[: GRID * GRID].sum(), 2)

    def test_missing_players_are_not_invented(self):
        row = frame_positions(
            {"players": [(10, 10, "CT")], "carrier": None, "bomb": None}, WIDTH, HEIGHT
        )
        self.assertEqual(row.sum(), 2)  # One CT cell plus its total; nothing else is filled in.
        self.assertEqual(dict(zip(TOTALS, row[-4:], strict=True))["t_seen"], 0)

    def test_edges_stay_inside_the_grid(self):
        found = {"players": [(WIDTH, HEIGHT, "T"), (-1, -1, "CT")], "carrier": None, "bomb": None}
        row = frame_positions(found, WIDTH, HEIGHT)
        self.assertEqual(row[cell(self.names, "t_observed", 7, 7)], 1)
        self.assertEqual(row[cell(self.names, "ct_observed", 0, 0)], 1)

    def test_channel_columns_cover_a_channel_and_its_total(self):
        columns = channel_columns(("bomb_badge",))
        self.assertEqual(len(columns), GRID * GRID + 1)
        picked = {self.names[i] for i in columns}
        self.assertIn("bomb_badge_present", picked)
        self.assertTrue(all(n.startswith("bomb_badge") for n in picked))
        with self.assertRaisesRegex(ValueError, "Unknown channels"):
            channel_columns(("bombs",))


if __name__ == "__main__":
    unittest.main()
