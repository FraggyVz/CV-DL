import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stratscout.clip_data import (
    NEVER_S,
    REGIONS_PATH,
    Regions,
    ResnetCompressor,
    assemble,
    build_clip_data,
    clip_summary,
    detector_channels,
    load_clip_data,
    pad,
)

REGIONS = Regions.load()
CENTRE = {
    name: ((rects[0][0] + rects[0][2]) / 2, (rects[0][1] + rects[0][3]) / 2)
    for name, rects in zip(REGIONS.names, REGIONS.rects)
}


def detections(path_by_frame, ct=(), bomb=None, fps=10.0):
    """path_by_frame: per frame a list of T (x, y); ct: CT (x, y) on every frame."""
    rows = []
    for k, pts in enumerate(path_by_frame):
        rows += [(k, x, y, 0, 0.0) for x, y in pts]
        rows += [(k, x, y, 1, 0.0) for x, y in ct]
    n = len(path_by_frame)
    return {
        "players": np.asarray(rows, np.float32).reshape(-1, 5),
        "bomb": np.zeros((n, 4), np.float32) if bomb is None else np.asarray(bomb, np.float32),
        "fps": np.float32(fps),
    }


def channel(x, names, name):
    return x[:, names.index(name)]


class RegionTests(unittest.TestCase):
    def test_regions_file_is_valid_and_first_match_wins(self):
        raw = json.loads(REGIONS_PATH.read_text())
        self.assertEqual(len(raw["regions"]), len(set(REGIONS.names)))
        for rects in REGIONS.rects:
            self.assertTrue(np.all(rects[:, 0] < rects[:, 2]) and np.all(rects[:, 1] < rects[:, 3]))
        idx = REGIONS.index(np.array([CENTRE["a_site"], CENTRE["b_site"], (0.01, 0.01)]))
        self.assertEqual([REGIONS.names[i] for i in idx[:2]], ["a_site", "b_site"])
        self.assertEqual(idx[2], -1)


class ChannelTests(unittest.TestCase):
    def test_split_route_reaches_both_approaches_with_their_gap(self):
        # Two Ts on ramp from the start; a third reaches connector after 4 s (40 frames).
        ramp, conn = CENTRE["a_ramp"], CENTRE["connector"]
        path = [[ramp, ramp] + ([conn] if k >= 40 else [CENTRE["mid"]]) for k in range(100)]
        frames, x, names = detector_channels(detections(path), REGIONS, (494, 500))
        self.assertEqual(len(frames), 50)  # every second frame: 5 rows per second
        self.assertAlmostEqual(float(channel(x, names, "t_in_a_ramp")[0]), 0.4)
        reached = channel(x, names, "t_reached_connector")
        self.assertEqual(int(np.argmax(reached)), 20)  # row 20 = 4.0 s
        self.assertTrue(np.all(np.diff(reached) >= 0))  # reached never switches back off
        s, s_names = clip_summary(x, names, REGIONS, rows_per_s=5.0)
        self.assertAlmostEqual(float(s[s_names.index("a_split_gap_s")]), 4.0)
        self.assertEqual(float(s[s_names.index("b_split_gap_s")]), NEVER_S)

    def test_one_frame_flicker_does_not_count_as_reaching_a_region(self):
        a = CENTRE["a_site"]
        path = [[CENTRE["t_spawn"]] for _ in range(40)]
        path[20] = [a]  # one sampled row in A, then gone
        _, x, names = detector_channels(detections(path), REGIONS, (494, 500))
        self.assertEqual(float(channel(x, names, "t_reached_a_site").max()), 0.0)

    def test_bomb_still_on_a_site_is_an_estimated_plant_and_stays_planted(self):
        a = CENTRE["a_site"]
        bomb = [(1, a[0], a[1], 1)] * 10 + [(1, a[0], a[1], 0)] * 40 + [(0, 0, 0, 0)] * 10
        path = [[CENTRE["a_ramp"]] for _ in range(60)]
        _, x, names = detector_channels(detections(path, bomb=bomb), REGIONS, (494, 500))
        planted = channel(x, names, "bomb_planted_est")
        self.assertEqual(planted[0], 0.0)
        self.assertEqual(planted[-1], 1.0)  # badge gone, plant kept
        first = int(np.argmax(planted))
        self.assertEqual(first, 10)  # carried to frame 9, uncarried on site 1 s: frame 20 = row 10
        self.assertEqual(float(channel(x, names, "bomb_carried")[0]), 1.0)
        s, s_names = clip_summary(x, names, REGIONS, rows_per_s=5.0)
        self.assertEqual((s[s_names.index("planted_a")], s[s_names.index("planted_b")]), (1, 0))

    def test_a_flickering_badge_on_site_still_counts_as_planted(self):
        a = CENTRE["a_site"]
        seen, gone = (1, a[0], a[1], 0), (0, 0, 0, 0)
        # Seen 2 frames, gone 3: longer gaps than the tracker tolerates, so it never settles.
        bomb = [seen if k % 5 < 2 else gone for k in range(60)]
        path = [[CENTRE["a_ramp"]] for _ in range(60)]
        _, x, names = detector_channels(detections(path, bomb=bomb), REGIONS, (494, 500))
        self.assertEqual(float(channel(x, names, "bomb_dropped").max()), 0.0)  # timer resets
        self.assertEqual(float(channel(x, names, "bomb_planted_est")[-1]), 1.0)

    def test_short_is_on_the_route_from_mid_to_b(self):
        # Points on the mid -> B catwalk route seen in B_SPLIT clips.
        idx = REGIONS.index(np.array([(0.46, 0.36), (0.40, 0.30)]))
        self.assertEqual([REGIONS.names[i] for i in idx], ["short", "short"])


class AssemblyTests(unittest.TestCase):
    def test_compressor_is_fitted_on_given_clips_and_padding_masks_rows(self):
        rng = np.random.default_rng(0)
        train = [rng.normal(size=(n, 512)).astype(np.float16) for n in (5, 7)]
        comp = ResnetCompressor(4).fit(train)
        self.assertEqual(comp(train[0]).shape, (5, 4))
        with self.assertRaises(ValueError):
            ResnetCompressor(4)(train[0])
        x, mask = pad([np.ones((2, 3)), np.ones((4, 3))])
        self.assertEqual(x.shape, (2, 4, 3))
        self.assertEqual(mask.tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])

    def test_build_writes_every_clip_and_loads_back(self):
        rows, dets = [], {}
        for k, (label, region) in enumerate((("A_DIRECT", "a_ramp"), ("B_DIRECT", "apartments"))):
            clip_id = f"m{k}.jx.s00{k}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "split": "train",
                    "match": f"m{k}",
                    "label": label,
                    "duration_s": 3.0,
                    "uncertain": k == 1,
                }
            )
            dets[clip_id] = detections([[CENTRE[region]] for _ in range(30)])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            build_clip_data(
                "synthetic", out, rows=rows, dets=dets, with_resnet=False, log=lambda m: None
            )
            clips = load_clip_data(out)
            self.assertIn("Does the data separate the classes?", (out / "report.md").read_text())
            with self.assertRaises(FileExistsError):
                build_clip_data("synthetic", out, rows=rows, dets=dets, with_resnet=False)
        self.assertEqual([c["y"] for c in clips], [0, 1])
        self.assertEqual([c["uncertain"] for c in clips], [False, True])
        x, names, s, s_names = assemble(clips[1])
        self.assertEqual(x.shape, (15, len(names)))
        self.assertEqual(float(s[s_names.index("first_t_apartments_s")]), 0.0)
        with self.assertRaises(ValueError):
            assemble(clips[0], ResnetCompressor(2))


if __name__ == "__main__":
    unittest.main()
