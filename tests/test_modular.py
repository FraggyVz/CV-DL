"""Regression and wiring checks for the notebook-to-package migration."""

import ast
import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from stratscout import workflow
from stratscout.annotations import ReviewTool
from stratscout.config import Config
from stratscout.context import RunContext
from stratscout.contracts import Round
from stratscout.detector import MinimapDetector, gaussian_heatmap
from stratscout.ingest.config import DemCfg
from stratscout.ingest.context import IngestContext
from stratscout.ingest.coordinates import world_to_radar
from stratscout.ingest.video import radar_only_clip
from stratscout.preprocessing import MinimapPreprocessor
from stratscout.storage import save_checkpoint, write_new_text
from stratscout.synthetic import build_background, make_round, render_frames
from stratscout.tracking import FixedSlotTracker
from stratscout.video import extract_clip

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def synthetic_context():
    ctx = RunContext(Config(minimap_px=32, seq_len=8, map_image="missing.png"))
    ctx.state["MAP_BG_SOURCE"] = None
    with contextlib.redirect_stdout(io.StringIO()):
        workflow.prepare_map(ctx, preview=False)
    return ctx


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT / "tests/fixtures/baseline.npz") as fixture:
            cls.expected = {k: fixture[k] for k in fixture.files}

    def test_synthetic_trajectories_and_frames_match_original(self):
        ctx = synthetic_context()
        rng = np.random.default_rng(17)
        pos = make_round("A_SPLIT", rng, 8, context=ctx)
        bg = build_background(32, np.random.default_rng(1234), context=ctx)
        frames = render_frames(pos, bg, 32, rng, context=ctx)
        np.testing.assert_array_equal(pos, self.expected["positions"])
        np.testing.assert_array_equal(frames, self.expected["frames"])

    def test_tracker_matches_original(self):
        # This preserves the old coasting behaviour; it does not endorse it.
        detections = [
            np.array([[0.2, 0.3, 1], [0.7, 0.6, 1]], np.float32),
            np.array([[0.21, 0.3, 1], [0.69, 0.6, 1]], np.float32),
            np.empty((0, 3), np.float32),
        ]
        np.testing.assert_array_equal(
            FixedSlotTracker(2, 0.09, 1).run(detections), self.expected["tracked"]
        )

    def test_detector_logits_and_heatmaps_match_original(self):
        ctx = synthetic_context()
        torch.manual_seed(7)
        model = MinimapDetector(context=ctx).eval()
        with torch.no_grad():
            actual = model(MinimapPreprocessor(32)(self.expected["frames"][:2])).numpy()
        np.testing.assert_allclose(actual, self.expected["logits"], atol=1e-6)
        np.testing.assert_array_equal(
            gaussian_heatmap(self.expected["positions"][0], 32, 1, context=ctx).numpy(),
            self.expected["heatmap"],
        )


class ModuleTests(unittest.TestCase):
    def test_all_definitions_are_available(self):
        inventory = json.loads((ROOT / "docs/archive/module-inventory.json").read_text())
        for entries in inventory.values():
            for item in entries:
                with self.subTest(name=item["name"]):
                    module = importlib.import_module("stratscout." + item["module"])
                    self.assertTrue(callable(getattr(module, item["name"])))

    def test_contexts_do_not_share_configuration_or_caches(self):
        a, b = RunContext(), RunContext()
        a.config.minimap_px = 64
        a.state["_GRID_CACHE"][64] = "example"
        self.assertEqual(b.config.minimap_px, 128)
        self.assertEqual(b.state["_GRID_CACHE"], {})
        x, y = IngestContext(), IngestContext()
        x.state["DEM_ROUNDS"] = [1]
        self.assertNotIn("DEM_ROUNDS", y.state)

    def test_round_phase_contract(self):
        r = Round("match", 1, "DEFAULT", plant_site="A", ct_label="SAVE", winner="T")
        self.assertEqual(r.label_of("post"), "SAVE")
        self.assertTrue(r.tactic_worked)
        self.assertFalse(r.response_worked)

    def test_notebooks_contain_no_implementation_definitions(self):
        for path in ROOT.glob("*.ipynb"):
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                if cell["cell_type"] == "code":
                    nodes = ast.walk(ast.parse("".join(cell["source"])))
                    self.assertFalse(
                        any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in nodes)
                    )

    def test_imports_do_not_execute_processing(self):
        code = """
import importlib, pkgutil, subprocess
from pathlib import Path
from unittest.mock import patch
import cv2, numpy, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stratscout
def forbidden(*args, **kwargs):
    raise AssertionError("Processing during import")
with patch.object(subprocess,"Popen",forbidden), patch.object(cv2,"VideoCapture",forbidden), patch.object(torch,"save",forbidden), patch.object(numpy,"save",forbidden), patch.object(numpy,"savez_compressed",forbidden), patch.object(Path,"write_text",forbidden), patch.object(plt,"show",forbidden):
    for item in pkgutil.walk_packages(stratscout.__path__, "stratscout."):
        importlib.import_module(item.name)
"""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_local_dependency_graph_has_no_cycles(self):
        graph = {}
        for path in (ROOT / "stratscout").rglob("*.py"):
            name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            package = name.rsplit(".", 1)[0]
            edges = set()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    edges.add(importlib.util.resolve_name("." * node.level + node.module, package))
            graph[name] = edges

        def walk(node, active, done):
            self.assertNotIn(node, active, f"Circular imports: {active} -> {node}")
            if node in done:
                return
            for edge in graph.get(node, []):
                walk(edge, active | {node}, done)
            done.add(node)

        done = set()
        for node in graph:
            walk(node, set(), done)


class FileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def video(self):
        path = self.folder / "input.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(20):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[8:40, 8:40] = (0, 0, 255)
            writer.write(frame)
        writer.release()
        return path

    def test_video_crop_and_rgb_contract(self):
        frames = extract_clip(str(self.video()), 0, 0.8, (8, 8, 40, 40), 4)
        self.assertEqual(frames.shape, (4, 32, 32, 3))
        self.assertGreater(float(frames[..., 0].mean()), 240)
        self.assertLess(float(frames[..., 2].mean()), 10)

    def test_ingest_radar_export_rate_and_dimensions(self):
        output = self.folder / "radar.mp4"
        radar_only_clip(self.video(), (8, 8, 40, 40), output, fps=10, context=IngestContext())
        cap = cv2.VideoCapture(str(output))
        self.addCleanup(cap.release)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FPS)), 10)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 32)

    def test_review_constructor_preserves_existing_labels(self):
        path = self.folder / "reviewed.json"
        content = json.dumps({"points": {"0": [[0.2, 0.3, 0]], "1": []}})
        path.write_text(content)
        with contextlib.redirect_stdout(io.StringIO()):
            review = ReviewTool(np.zeros((2, 32, 32, 3), np.uint8), str(path), context=RunContext())
        self.assertEqual(review.labels[1], [])
        self.assertEqual(path.read_text(), content)

    def test_outputs_refuse_overwrites(self):
        path = self.folder / "model.pt"
        save_checkpoint({"x": torch.tensor([1])}, path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_checkpoint({}, path)
        self.assertEqual(path.read_bytes(), before)
        text_path = self.folder / "summary.json"
        write_new_text(text_path, "original")
        with self.assertRaises(FileExistsError):
            write_new_text(text_path, "replacement")
        self.assertEqual(text_path.read_text(), "original")

    def test_world_conversion_keeps_missing_players(self):
        ctx = IngestContext(DemCfg())
        world = np.array([[-3230, 1713], [1890, -3407], [np.nan, np.nan]], np.float32)
        result = world_to_radar(world, {"s": 1, "ox": 0, "oy": 0}, context=ctx)
        np.testing.assert_allclose(result[:2], [[0, 0], [1, 1]])
        self.assertTrue(np.isnan(result[2]).all())


if __name__ == "__main__":
    unittest.main()
