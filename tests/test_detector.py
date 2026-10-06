from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from helpers import bgr, frame, profile
from stratscout.calibration import sample_hue
from stratscout.config import load_profile, save_profile
from stratscout.detector import ColourDetector, draw_detections


class DetectorTests(unittest.TestCase):
    def test_offset_crop_normalisation_and_no_double_crop(self):
        source = frame()
        before = source.copy()
        found = ColourDetector(profile()).detect(source)
        self.assertEqual(set(found), {"T", "CT"})
        self.assertEqual([len(found[k]) for k in ("T", "CT")], [1, 1])
        self.assertAlmostEqual(found["T"][0]["x"], 30 / 200)
        self.assertAlmostEqual(found["T"][0]["y"], 50 / 160)
        self.assertAlmostEqual(found["CT"][0]["x"], 130 / 200)
        self.assertAlmostEqual(found["CT"][0]["y"], 110 / 160)
        overlay = draw_detections(source, profile(), found)
        self.assertFalse(np.array_equal(overlay, source))
        np.testing.assert_array_equal(source, before)

    def test_empty_outputs_are_explicit(self):
        found = ColourDetector(profile()).detect(np.zeros((240, 320, 3), np.uint8))
        self.assertEqual(found, {"T": [], "CT": []})

    def test_hue_wraparound_and_bgr_input(self):
        settings = profile()
        settings.teams["T"] = replace(settings.teams["T"], hue=0, hue_tolerance=3)
        source = np.zeros((240, 320, 3), np.uint8)
        for x, hue in ((110, 179), (140, 1), (170, 0)):
            cv2.circle(source, (x, 120), 4, bgr(hue), -1)
        found = ColourDetector(settings).detect(source)
        self.assertEqual(len(found["T"]), 3)
        self.assertEqual(found["CT"], [])

    def test_distractors_filtered_without_map_exclusion_zones(self):
        source = frame()
        cv2.rectangle(source, (100, 75), (155, 76), bgr(30), -1)  # elongated
        cv2.rectangle(source, (230, 80), (260, 110), bgr(105), -1)  # oversized
        cv2.circle(source, (180, 130), 5, (220, 220, 220), -1)  # desaturated
        cv2.circle(source, (120, 180), 4, bgr(60), -1)  # wrong hue
        source[150, 150] = bgr(30)  # undersized
        found = ColourDetector(profile()).detect(source)
        self.assertEqual([len(found[k]) for k in ("T", "CT")], [1, 1])

    def test_no_five_player_padding_or_truncation(self):
        source = np.zeros((240, 320, 3), np.uint8)
        detector = ColourDetector(profile())
        for number in range(1, 7):
            cv2.circle(source, (90 + 25 * number, 150), 4, bgr(30), -1)
            self.assertEqual(len(detector.detect(source)["T"]), number)
        self.assertEqual(detector.detect(source)["CT"], [])

    def test_unresolved_overlap_does_not_fabricate_two_icons(self):
        source = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(source, (120, 120), 4, bgr(30), -1)
        cv2.circle(source, (123, 120), 4, bgr(30), -1)
        self.assertEqual(len(ColourDetector(profile()).detect(source)["T"]), 1)

    def test_wrong_dimensions_or_dtype_rejected(self):
        detector = ColourDetector(profile())
        for source in (np.zeros((160, 200, 3), np.uint8), frame().astype(float)):
            with self.assertRaisesRegex(ValueError, "Recalibrate"):
                detector.detect(source)

    def test_hue_sampling_handles_wraparound_and_rejects_grey(self):
        patch = np.array([[bgr(179), bgr(1), bgr(0)]] * 3, np.uint8)
        self.assertEqual(sample_hue(patch, 1, 1), 0)
        with self.assertRaises(ValueError):
            sample_hue(np.zeros((3, 3, 3), np.uint8), 1, 1)


class ProfileTests(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_profile(profile(), path)
            before = path.read_bytes()
            self.assertEqual(load_profile(path), profile())
            with self.assertRaises(FileExistsError):
                save_profile(profile(), path)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_profiles(self):
        for changes in ({"crop": (0, 0, 400, 240)}, {"crop": (80, 60, 80, 90)},
                        {"crop": (0., 0, 20, 20)}, {"schema_version": 2}, {"teams": {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(profile(), **changes).validate()
        settings = profile()
        settings.teams["CT"] = replace(settings.teams["CT"], hue=35)
        with self.assertRaisesRegex(ValueError, "overlap"):
            settings.validate()
        for changes in ({"area_min": 300}, {"hue": -1}, {"aspect_max": float("nan")},
                        {"fill_min": 2}, {"saturation_min": 0}, {"hue": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(profile().teams["T"], **changes).validate()

    def test_malformed_json_has_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            for data in ("{", "[]", json.dumps({"teams": {}})):
                path.write_text(data)
                with self.assertRaisesRegex(ValueError, "Invalid profile"):
                    load_profile(path)


if __name__ == "__main__":
    unittest.main()
