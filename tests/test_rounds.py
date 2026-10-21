"""Round estimation from spawn-occupancy flags."""

import unittest

import numpy as np

from stratscout.rounds import CT_SPAWN, T_SPAWN, in_spawns, round_at, rounds_from_flags


def flags(freezes, duration, step=0.4):
    """Samples every step seconds, True inside any (start, end) freeze interval."""
    times = np.arange(0, duration, step)
    return [(float(t), any(a <= t <= b for a, b in freezes)) for t in times]


class RoundGroupingTests(unittest.TestCase):
    def test_each_freeze_starts_a_round(self):
        rounds = rounds_from_flags(flags([(5, 25), (120, 140), (250, 270)], 400), 400)
        self.assertEqual([r["round"] for r in rounds], [1, 2, 3])
        self.assertAlmostEqual(rounds[1]["start"], 120, delta=0.5)
        self.assertAlmostEqual(rounds[0]["freeze"], 20, delta=0.5)
        self.assertEqual(rounds[-1]["end"], 400)
        self.assertFalse(any(r["partial"] for r in rounds))

    def test_recording_that_starts_mid_round_gets_a_partial_round(self):
        rounds = rounds_from_flags(flags([(60, 80), (200, 220)], 300), 300)
        self.assertTrue(rounds[0]["partial"])
        self.assertEqual(rounds[0]["start"], 0.0)
        self.assertEqual(rounds[0]["end"], rounds[1]["start"])
        self.assertEqual(len(rounds), 3)

    def test_timeout_and_brief_gaps_stay_one_round(self):
        # A long timeout freeze with a short interruption is still one round start.
        f = flags([(5, 40), (44, 75), (200, 220)], 300)
        rounds = rounds_from_flags(f, 300)
        self.assertEqual(len(rounds), 2)
        self.assertGreater(rounds[0]["freeze"], 60)

    def test_momentary_spawn_coincidences_are_ignored(self):
        rounds = rounds_from_flags(flags([(5, 25), (100, 101), (200, 220)], 300), 300)
        self.assertEqual(len(rounds), 2)

    def test_round_at(self):
        rounds = rounds_from_flags(flags([(5, 25), (120, 140)], 300), 300)
        self.assertEqual(round_at(rounds, 50)["round"], 1)
        self.assertEqual(round_at(rounds, 130)["round"], 2)
        self.assertEqual(round_at(rounds, 1)["round"], 1)


class SpawnOccupancyTests(unittest.TestCase):
    @staticmethod
    def frame(t_at, ct_at, size=(500, 494)):
        h, w = size
        f = np.full((h, w, 3), 60, np.uint8)
        for (fx, fy), bgr in ((t_at, (0, 225, 255)), (ct_at, (240, 200, 40))):
            cx, cy = int(fx * w), int(fy * h)
            for dx in (-12, 0, 12):  # three icons per team
                f[cy - 7 : cy + 7, cx + dx - 5 : cx + dx + 5] = bgr
        return f

    def test_both_teams_in_their_spawns(self):
        background = np.full((500, 494, 3), 60, np.uint8)
        self.assertTrue(in_spawns(self.frame(T_SPAWN, CT_SPAWN), background, min_pixels=200))
        self.assertFalse(in_spawns(self.frame((0.5, 0.5), CT_SPAWN), background, min_pixels=200))


if __name__ == "__main__":
    unittest.main()
