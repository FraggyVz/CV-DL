import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stratscout.detector_check import (
    TICKRATE,
    CheckConfig,
    DemoPositions,
    PassMark,
    RadarMap,
    assign,
    calibrate,
    carrier_at,
    carrier_timeline,
    demo_rounds,
    find_demo,
    fit_radar_map,
    fit_time_map,
    refuse_held_out,
    run_detector_check,
    sample_times,
    score_frames,
    team_scores,
)


class HeldOutTests(unittest.TestCase):
    def test_navi_is_refused_by_any_name(self):
        for name in ("navi", "navi-falcons-mirage-cologne-major", "natus-vincere-vs-falcons.dem"):
            with self.assertRaises(ValueError):
                refuse_held_out(name)
        refuse_held_out("furia", "spirit-vs-falcons-m2-mirage.dem")

    def test_vitality_falcons_is_refused_but_aurora_vitality_is_not(self):
        for name in (
            "vitality-falcons",
            "vitality-falcons-mirage-esl-pro-league-season-24",
            "vitality-vs-falcons-m3-mirage.dem",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                refuse_held_out(name)
        refuse_held_out(
            "aurora",
            "aurora-vitality-mirage-esl-pro-league-season-24",
            "aurora-vs-vitality-m3-mirage.dem",
            "mouz-betboom-mirage-iem-bejing-2026",
            "mouz-vs-betboom-m2-mirage.dem",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "held-out"):
                run_detector_check(
                    ["vitality-falcons"], Path(tmp) / "out", demos_dir=Path(tmp) / "none"
                )
            self.assertFalse((Path(tmp) / "out").exists())

    def test_run_refuses_navi_before_reading_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "held-out"):
                run_detector_check(["navi"], Path(tmp) / "out", demos_dir=Path(tmp) / "none")
            self.assertFalse((Path(tmp) / "out").exists())

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run_detector_check(["furia"], Path(tmp))


class RoundAndTimeTests(unittest.TestCase):
    def test_demo_rounds_drop_warm_up_and_duplicate_ends(self):
        freeze = pd.DataFrame({"tick": [100, 1000, 2000]})
        ends = pd.DataFrame(
            {
                "tick": [50, 900, 905, 1900],
                "total_rounds_played": [0, 1, 1, 2],
                "reason": [None, "t_killed", "t_killed", "bomb_exploded"],
            }
        )
        self.assertEqual(demo_rounds(freeze, ends), [(100, 905), (1000, 1900)])

    def test_time_map_recovers_offset_and_speed(self):
        demo = np.array([10.0, 120.0, 260.0, 400.0])
        a, b, residuals = fit_time_map(demo, 12.5 + 1.0 * demo)
        self.assertAlmostEqual(a, 12.5, places=6)
        self.assertAlmostEqual(b, 1.0, places=6)
        self.assertLess(np.abs(residuals).max(), 1e-6)

    def test_sample_times_skip_partial_rounds_and_stop_at_the_demo_end(self):
        video_rounds = [
            {"start": 0.0, "freeze": 0.0, "partial": True, "end": 30.0},
            {"start": 30.0, "freeze": 3.0, "partial": False, "end": 100.0},
        ]
        rounds = [(0, 640), (2000, int(40 * TICKRATE))]  # second round ends at demo 40 s
        times = sample_times(video_rounds, rounds, (0.0, 1.0), every_s=1.0)
        self.assertEqual({t["round"] for t in times}, {2})
        self.assertEqual([t["video_s"] for t in times if t["phase"] == "freeze"], [31.0, 32.0])
        live = [t["video_s"] for t in times if t["phase"] == "live"]
        self.assertEqual(live[0], 33.0)
        self.assertLess(live[-1], 40.0)


class GeometryTests(unittest.TestCase):
    def test_radar_map_fit_recovers_parameters(self):
        rng = np.random.default_rng(0)
        uv = rng.uniform(0.1, 0.9, size=(50, 2))
        truth = RadarMap(480.0, 470.0, 12.0, -8.0)
        fitted = fit_radar_map(uv, truth.apply(uv))
        for name in ("sx", "sy", "tx", "ty"):
            self.assertAlmostEqual(getattr(fitted, name), getattr(truth, name), places=6)

    def test_assignment_is_optimal_not_greedy(self):
        truth = np.array([[0.0, 0.0], [10.0, 0.0]])
        dets = np.array([[6.0, 0.0], [16.0, 0.0]])
        pairs = sorted((i, j) for i, j, _ in assign(truth, dets, gate=36))
        self.assertEqual(pairs, [(0, 0), (1, 1)])  # greedy would take (1, 0) first

    def test_assignment_drops_pairs_beyond_the_gate_and_handles_uneven_counts(self):
        truth = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        dets = np.array([[1.0, 0.0], [150.0, 0.0]])
        self.assertEqual([(i, j) for i, j, _ in assign(truth, dets, gate=36)], [(0, 0)])
        self.assertEqual(assign(truth, np.zeros((0, 2)), gate=36), [])


class DemoPositionTests(unittest.TestCase):
    def positions(self):
        ticks = [0, 4, 8]
        xy = np.array(
            [
                [[0.0, 0.0], [500.0, 500.0], [0.0, 0.0]],
                [[40.0, 0.0], [500.0, 500.0], [0.0, 0.0]],
                [[80.0, 0.0], [500.0, 500.0], [0.0, 0.0]],
            ]
        )
        alive = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], bool)  # player 1 dies by tick 8
        team = np.array([[0, 1, -1]] * 3)  # player 2 is not playing
        return DemoPositions(ticks, ["a", "b", "c"], xy, alive, team)

    def test_positions_are_interpolated_with_speed(self):
        at = self.positions().at(2 / TICKRATE)
        self.assertEqual(at["steamid"], ["a", "b"])
        self.assertAlmostEqual(at["xy"][0, 0], 20.0)
        self.assertAlmostEqual(at["speed"][0], 40.0 / (4 / TICKRATE))

    def test_dying_players_and_non_players_are_left_out(self):
        at = self.positions().at(6 / TICKRATE)
        self.assertEqual(at["steamid"], ["a"])

    def test_outside_the_samples_gives_nothing(self):
        self.assertIsNone(self.positions().at(100.0))


class CarrierTests(unittest.TestCase):
    def test_carrier_follows_pickups_drops_and_plants(self):
        timeline = carrier_timeline(
            pd.DataFrame({"tick": [10, 30], "user_steamid": ["A", "B"]}),
            pd.DataFrame({"tick": [20]}),
            pd.DataFrame({"tick": [40]}),
        )
        found = [carrier_at(timeline, t) for t in (5, 15, 25, 35, 45)]
        self.assertEqual(found, [None, "A", None, "B", None])


class ScoreTests(unittest.TestCase):
    def test_p90_is_not_cut_at_the_match_distance(self):
        players = [
            {"distance_px": d, "found": d <= 12} for d in (1.0, 2.0, 3.0, float("nan"), 20.0)
        ]
        detections = [{"extra": False}] * 3 + [{"extra": True}]
        s = team_scores(players, detections, PassMark())
        self.assertEqual((s["found"], s["missed"]), (3, 2))
        self.assertAlmostEqual(s["extra_rate"], 0.25)
        self.assertGreater(s["p90_px"], 12.0)
        self.assertFalse(s["pass"])


def synthetic_match(true_offset=0.4, radar=RadarMap(480.0, 470.0, 10.0, -5.0), noise=0.3):
    """Ten players frozen for 15 s, then running in straight lines; detections are their
    true radar positions at a known time offset, plus centroid noise."""
    rng = np.random.default_rng(1)
    start = rng.uniform([-2400, -2000], [800, 800], size=(10, 2))
    heading = rng.uniform(-1, 1, size=(10, 2))
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    ticks = np.arange(0, 70 * TICKRATE, 4)
    moving = np.clip(ticks / TICKRATE - 15.0, 0, None)
    xy = start[None] + 200.0 * moving[:, None, None] * heading[None]
    team = np.repeat([[0] * 5 + [1] * 5], len(ticks), axis=0)
    positions = DemoPositions(ticks, list("abcdefghij"), xy, np.ones(team.shape, bool), team)
    time_map = (5.0, 1.0)  # video = 5 + demo
    frames = []
    for number, (phase, t0, t1) in enumerate(
        (("freeze", 6.0, 20.0), ("live", 20.0, 50.0), ("live", 50.0, 64.0)), start=1
    ):
        for video_s in np.arange(t0, t1, 1.0):
            truth = positions.at(video_s - 5.0 + true_offset)
            px = radar.apply(truth["uv"]) + rng.normal(0, noise, size=(10, 2))
            frames.append(
                {
                    "match": "synthetic",
                    "round": 1 if number < 3 else 2,
                    "phase": phase,
                    "video_s": float(video_s),
                    "dets": [px[truth["team"] == 0], px[truth["team"] == 1]],
                    "inferred": [[False] * 5, [False] * 5],
                }
            )
    return positions, time_map, frames


class CalibrationTests(unittest.TestCase):
    def test_live_frames_start_the_mapping_when_spawn_hides_an_icon(self):
        radar = RadarMap(480.0, 470.0, 10.0, -5.0)
        positions, time_map, frames = synthetic_match(0.4, radar)
        odd = [f for f in frames if f["round"] == 1]
        with_freeze = calibrate(odd, positions, time_map, CheckConfig(), PassMark())
        self.assertEqual(with_freeze["mapping_start"], "freeze")
        for frame in odd:
            if frame["phase"] == "freeze":  # One stacked CT icon is never separated.
                frame["dets"] = [frame["dets"][0], frame["dets"][1][:4]]
        fit = calibrate(odd, positions, time_map, CheckConfig(), PassMark())
        self.assertEqual(fit["mapping_start"], "live")
        self.assertAlmostEqual(fit["offset_s"], 0.4, delta=0.051)
        self.assertAlmostEqual(fit["radar"].sx, radar.sx, delta=2.0)
        self.assertAlmostEqual(fit["radar"].tx, radar.tx, delta=1.0)

    def test_calibration_recovers_known_offset_and_mapping(self):
        radar = RadarMap(480.0, 470.0, 10.0, -5.0)
        positions, time_map, frames = synthetic_match(0.4, radar)
        odd = [f for f in frames if f["round"] == 1]
        fit = calibrate(odd, positions, time_map, CheckConfig(), PassMark())
        self.assertAlmostEqual(fit["offset_s"], 0.4, delta=0.051)
        self.assertAlmostEqual(fit["radar"].sx, radar.sx, delta=2.0)
        self.assertAlmostEqual(fit["radar"].tx, radar.tx, delta=1.0)

        scored = [f for f in frames if f["round"] == 2]
        rounds = [(0, 70 * TICKRATE), (0, 70 * TICKRATE)]
        players, detections = score_frames(
            scored,
            positions,
            time_map,
            fit["offset_s"],
            fit["radar"],
            rounds,
            [],
            CheckConfig(),
            PassMark(),
        )
        s = team_scores(players, detections, PassMark())
        self.assertEqual(s["found_rate"], 1.0)
        self.assertLess(s["median_px"], 2.0)
        self.assertTrue(s["pass"])


class FindDemoTests(unittest.TestCase):
    def test_exactly_one_demo_per_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("furia-vs-falcons-m1-mirage.dem", "spirit-vs-falcons-m2-mirage.dem"):
                (Path(tmp) / name).write_bytes(b"")
            self.assertEqual(find_demo(Path(tmp), "furia").name, "furia-vs-falcons-m1-mirage.dem")
            with self.assertRaises(ValueError):
                find_demo(Path(tmp), "vitality")


if __name__ == "__main__":
    unittest.main()
