import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stratscout.segment_labeller import SegmentLabeller, play_start, screen_work_area, wrap
from stratscout.segments import (
    EXCLUSION_REASONS,
    SegmentFile,
    match_identity,
    segments_json,
    validate_segment,
)

ROUNDS = [
    {"round": 1, "start": 0.0, "freeze": 0.0, "partial": True, "end": 46.0},
    {"round": 2, "start": 46.0, "freeze": 20.0, "partial": False, "end": 130.0},
    {"round": 3, "start": 130.0, "freeze": 20.0, "partial": False, "end": 300.0},
]


class SegmentTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "labels" / "furia.gin.segments.json"

    def open(self, match_id="abc", path=None):
        return SegmentFile(
            path or self.path,
            match_id=match_id,
            match_folder="furia",
            video="minimap-10fps.mp4",
            video_fps=10,
            reviewer="gin",
        )


class NamingTests(SegmentTestCase):
    def test_one_file_per_match_and_reviewer(self):
        a = segments_json(self.root / "furia-falcons", "Gin")
        b = segments_json(self.root / "furia-falcons", "ryan")
        self.assertEqual(a, Path("labels/furia-falcons.gin.segments.json"))
        self.assertNotEqual(a, b)

    def test_reviewer_must_be_file_safe(self):
        for bad in ("", "../x", "two words"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                segments_json("furia", bad)

    def test_match_identity_comes_from_the_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps({"match_id": "4cd1"}))
        self.assertEqual(match_identity(self.root), "4cd1")
        with self.assertRaises(ValueError):
            match_identity(self.root / "missing")


class ValidationTests(unittest.TestCase):
    def check(self, **seg):
        return validate_segment({"start_s": 10.0, "end_s": 20.0, "status": "pending", **seg})

    def test_attack_with_class_is_valid(self):
        self.assertEqual(self.check(label="A_SPLIT"), [])

    def test_boundaries_are_required_and_ordered(self):
        self.assertTrue(self.check(label="A_SPLIT", start_s=None))
        self.assertTrue(self.check(label="A_SPLIT", end_s=5.0))

    def test_class_must_be_chosen_explicitly(self):
        self.assertTrue(self.check(label=None))
        self.assertEqual(self.check(label=None, undecided=True), [])
        self.assertTrue(self.check(label="A_RUSH"))

    def test_reviewed_needs_a_class(self):
        self.assertTrue(self.check(label=None, undecided=True, status="reviewed"))

    def test_exclusion_needs_a_reason_and_no_class(self):
        self.assertTrue(self.check(label=None, status="excluded"))
        self.assertTrue(
            self.check(label="A_SPLIT", status="excluded", exclusion_reason=EXCLUSION_REASONS[0])
        )
        self.assertEqual(
            self.check(label=None, status="excluded", exclusion_reason=EXCLUSION_REASONS[0]), []
        )

    def test_retake_needs_a_site_and_attack_sites_must_match(self):
        self.assertTrue(self.check(label="RETAKE"))
        self.assertEqual(self.check(label="RETAKE", site="B"), [])
        self.assertTrue(self.check(label="A_DIRECT", site="B"))


class SegmentFileTests(SegmentTestCase):
    def test_add_saves_immediately_and_resumes(self):
        store = self.open()
        saved = store.add(
            {"start_s": 10.123, "end_s": 20.0, "status": "pending", "label": "B_SPLIT"}
        )
        self.assertEqual((saved["id"], saved["site"], saved["start_s"]), (1, "B", 10.12))
        self.assertFalse(list(self.path.parent.glob("*.tmp")))
        again = self.open()
        self.assertEqual(again.segments[0]["label"], "B_SPLIT")
        self.assertEqual(again.data["definition_version"], saved["definition_version"])

    def test_invalid_segments_are_not_saved(self):
        store = self.open()
        with self.assertRaises(ValueError):
            store.add({"start_s": 10.0, "end_s": 20.0, "status": "pending", "label": None})
        self.assertFalse(self.path.exists())

    def test_refuses_a_file_from_another_match(self):
        self.open().save()
        with self.assertRaises(ValueError):
            self.open(match_id="other")

    def test_undo_never_removes_reviewed_work(self):
        store = self.open()
        store.add({"start_s": 1.0, "end_s": 2.0, "status": "pending", "label": "A_DIRECT"})
        store.add({"start_s": 3.0, "end_s": 4.0, "status": "pending", "label": "B_DIRECT"})
        self.assertEqual(store.undo_last()["label"], "B_DIRECT")
        store.segments[0]["status"] = "reviewed"
        with self.assertRaises(ValueError):
            store.undo_last()
        self.assertEqual(len(store.segments), 1)

    def test_checked_rounds_and_bookmarks_persist(self):
        store = self.open()
        self.assertTrue(store.toggle_checked(4))
        store.add_bookmark(12.345)
        again = self.open()
        self.assertEqual(again.data["checked_rounds"], [4])
        self.assertEqual(again.data["bookmarks"], [12.35])
        self.assertFalse(again.toggle_checked(4))


class LabellerKeyTests(SegmentTestCase):
    def setUp(self):
        super().setUp()
        self.store = self.open()
        self.ui = SegmentLabeller(self.store, ROUNDS)

    def keys(self, *steps):
        for key, seconds in steps:
            self.ui.press(key, seconds)

    def test_marks_class_and_enter_save_a_segment(self):
        self.keys(("i", 80.0), ("o", 95.0), ("3", 95.0), ("enter", 95.0))
        seg = self.store.segments[0]
        self.assertEqual((seg["label"], seg["site"], seg["status"]), ("A_SPLIT", "A", "pending"))
        self.assertEqual((seg["estimated_round"], seg["round_play_start_s"]), (2, 66.0))
        self.assertIsNone(self.ui.draft["start_s"])

    def test_d_deletes_the_segment_being_watched(self):
        self.keys(("i", 80.0), ("o", 95.0), ("1", 95.0), ("s", 95.0))
        self.keys(("i", 160.0), ("o", 170.0), ("2", 170.0), ("s", 170.0))
        self.ui.press("d", 50.0)
        self.assertFalse(self.ui.ok)
        self.assertEqual(len(self.store.segments), 2)
        self.ui.press("d", 85.0)  # first press only asks
        self.assertEqual(len(self.store.segments), 2)
        self.assertIn("Press D again", self.ui.message)
        self.ui.press("d", 85.0)
        self.assertEqual([s["label"] for s in self.store.segments], ["B_DIRECT"])
        self.assertIn("Deleted #1", self.ui.message)
        self.store.segments[0]["status"] = "reviewed"
        self.keys(("d", 165.0), ("d", 165.0))
        self.assertEqual(len(self.store.segments), 1)
        self.assertFalse(self.ui.ok)

    def test_delete_keys_need_two_presses_and_any_other_key_cancels(self):
        self.keys(("i", 80.0), ("o", 95.0), ("1", 95.0), ("s", 95.0))
        self.keys(("i", 160.0), ("o", 170.0), ("2", 170.0), ("s", 170.0))
        self.keys(("z", 0.0), ("x", 0.0), ("z", 0.0))  # X between the presses cancels
        self.assertEqual(len(self.store.segments), 2)
        self.ui.press("z", 0.0)
        self.assertEqual([s["label"] for s in self.store.segments], ["A_DIRECT"])
        self.keys(("z", 0.0), ("d", 85.0))  # a Z press does not arm D
        self.assertEqual(len(self.store.segments), 1)

    def test_s_saves_like_enter(self):
        self.keys(("i", 80.0), ("o", 95.0), ("2", 95.0), ("s", 95.0))
        self.assertEqual(self.store.segments[0]["label"], "B_DIRECT")
        self.assertIn("Saved #1", self.ui.message)

    def test_enter_without_a_class_keeps_the_draft(self):
        self.keys(("i", 80.0), ("o", 95.0), ("enter", 95.0))
        self.assertEqual(self.store.segments, [])
        self.assertFalse(self.ui.ok)
        self.assertEqual(self.ui.draft["start_s"], 80.0)

    def test_retake_needs_a_site_key(self):
        self.keys(("i", 100.0), ("o", 120.0), ("5", 120.0), ("enter", 120.0))
        self.assertEqual(self.store.segments, [])
        self.keys(("b", 120.0), ("enter", 120.0))
        self.assertEqual(self.store.segments[0]["site"], "B")

    def test_exclusion_cycles_reasons_and_keeps_the_proposal(self):
        self.keys(("i", 80.0), ("o", 95.0), ("1", 95.0), ("e", 95.0), ("e", 95.0), ("enter", 95.0))
        seg = self.store.segments[0]
        self.assertEqual(seg["status"], "excluded")
        self.assertEqual(seg["exclusion_reason"], "no retake attempt (save)")
        self.assertIsNone(seg["label"])
        self.assertEqual(seg["proposed_label"], "A_DIRECT")

    def test_e_cycles_fake_save_unusable_other(self):
        for reason in (*EXCLUSION_REASONS, EXCLUSION_REASONS[0]):
            self.ui.press("e", 0.0)
            self.assertEqual(self.ui.reason, reason)
        self.assertEqual(EXCLUSION_REASONS[1], "no retake attempt (save)")

    def test_undecided_and_uncertain_are_recorded(self):
        self.keys(("i", 80.0), ("o", 95.0), ("0", 95.0), ("u", 95.0), ("enter", 95.0))
        seg = self.store.segments[0]
        self.assertTrue(seg["undecided"] and seg["boundary_uncertain"])
        self.assertIsNone(seg["label"])

    def test_round_keys_jump_to_estimated_play_starts(self):
        self.assertEqual(self.ui.press("]", 10.0), play_start(ROUNDS[1]))
        self.assertEqual(self.ui.press("[", 100.0), play_start(ROUNDS[1]))
        self.assertEqual(self.ui.press("[", 67.0), play_start(ROUNDS[0]))
        self.assertIsNone(self.ui.press("]", 200.0))

    def test_bookmarks_and_checked_round(self):
        self.assertIsNone(self.ui.press("j", 10.0))
        self.keys(("m", 30.0), ("m", 90.0), ("c", 90.0))
        self.assertEqual(self.ui.press("j", 40.0), 90.0)
        self.assertEqual(self.ui.press("j", 95.0), 30.0)
        self.assertEqual(self.store.data["checked_rounds"], [2])

    def test_next_and_previous_saved_segment(self):
        self.assertIsNone(self.ui.press("n", 10.0))
        self.keys(("i", 160.0), ("o", 170.0), ("2", 170.0), ("enter", 170.0))
        self.keys(("i", 80.0), ("o", 95.0), ("3", 95.0), ("enter", 95.0))
        self.assertEqual(self.ui.press("n", 10.0), 80.0)
        self.assertEqual(self.ui.press("n", 80.0), 160.0)
        self.assertEqual(self.ui.press("n", 160.0), 80.0)
        self.assertEqual(self.ui.press("p", 160.0), 80.0)
        self.assertEqual(self.ui.press("p", 80.0), 160.0)
        self.assertIn("#1 B_DIRECT", self.ui.message)
        line = self.ui.status_lines(85.0, "PAUSED")[3][0]
        self.assertIn(">#2 A_SPLIT", line)

    def test_display_lines_and_timeline_render_without_a_window(self):
        self.keys(("i", 80.0), ("o", 95.0), ("2", 95.0), ("enter", 95.0), ("i", 140.0))
        lines = [text for text, _ in self.ui.status_lines(150.0, "PAUSED")]
        self.assertIn("Round 3/3", lines[1])
        self.assertIn("02:30.0", lines[1])
        bar = self.ui.timeline(400, 300.0, 150.0)
        self.assertEqual(bar.shape, (22, 400, 3))

    def test_long_lines_wrap_at_separators(self):
        fits = lambda s: len(s) <= 12  # noqa: E731
        self.assertEqual(wrap("short", fits), ["short"])
        self.assertEqual(wrap("aaaa | bbbb | cccc", fits), ["aaaa | bbbb", "cccc"])
        self.assertEqual(wrap("#1 aaaa, #2 bbbb", fits), ["#1 aaaa", "#2 bbbb"])
        self.assertEqual(wrap("unbreakable-word-here", fits), ["unbreakable-word-here"])

    def test_screen_work_area_is_a_usable_rectangle(self):
        left, top, width, height = screen_work_area()
        self.assertGreater(width, 200)
        self.assertGreater(height, 200)

    def test_window_image_matches_the_window_size(self):
        frame = np.zeros((500, 494, 3), np.uint8)
        for size in ((1850, 900), (1100, 938), (700, 1000), (500, 600)):
            with self.subTest(size=size):
                img = self.ui.compose(frame, 150.0, "PAUSED", 300.0, size=size)
                self.assertEqual(img.shape[:2], (size[1], size[0]))
        self.assertEqual(self.ui.compose(frame, 150.0, "PAUSED", 300.0).shape[1], 988)


if __name__ == "__main__":
    unittest.main()
