"""Synthetic checks of the final-model prediction contract (gru_predict), not accuracy."""

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from stratscout import gru_predict, tcn_predict
from stratscout.__main__ import main
from stratscout.clip_data import REGIONS_PATH, Regions, detector_channel_names
from stratscout.config import TACTIC_CLASSES
from stratscout.context import RunContext
from stratscout.segment_export import COLUMNS
from stratscout.tactic_features import match_background
from stratscout.tactic_tcn import TacticConfig, run_tactic_experiment
from stratscout.tactic_temporal import TIME_SCALE_S, clip_data_inputs, load_clip_rows
from stratscout.tactic_training import BaselineConfig

REGIONS = Regions.load(REGIONS_PATH)
DETECTOR = detector_channel_names(REGIONS)  # The real 72-channel layout.
RESNET = 40
MATCHES = (
    ("aurora", [0, 1, 2, 3, 4, 4]),
    ("furia", [0, 1, 1, 2, 3, 4]),
    ("mouz", [0, 1, 2, 3, 3, 4]),
)
FRAMES = 12  # 10 fps video frames per clip -> 6 rows at step 2


def clip_arrays(generator, label, rows):
    detector = generator.normal(0, 0.1, (rows, len(DETECTOR))).astype(np.float32)
    detector[:, label] += 1.0
    resnet = generator.normal(size=(rows, RESNET)).astype(np.float16)
    return detector, resnet


def fake_det(frames=FRAMES, fps=10.0):
    return {
        "players": np.zeros((0, 5), np.float32),
        "bomb": np.zeros((frames, 4), np.float32),
        "fps": np.float32(fps),
    }


def write_dataset(root):
    """A train-only export (tiny videos) plus a train-only clip-data folder for it."""
    train = root / "dataset" / "train"
    train.mkdir(parents=True)
    clip_data = root / "clip-data"
    (clip_data / "clips").mkdir(parents=True)
    rows, index = [], []
    for match, labels in MATCHES:
        for i, label in enumerate(labels):
            clip_id = f"{match}.jx.s{i:03d}"
            path = train / f"{clip_id}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
            try:
                for frame in range(FRAMES):
                    writer.write(np.full((32, 32, 3), (label * 40, frame * 9, 90 + i), np.uint8))
            finally:
                writer.release()
            row = dict.fromkeys(COLUMNS, "")
            row.update(
                clip_id=clip_id,
                clip=path.name,
                split="train",
                label=TACTIC_CLASSES[label],
                match_id=match,
                match_folder=match,
                start_s=1,
                end_s=2.2,
                duration_s=1.2,
                frames=FRAMES,
                status="reviewed",
                boundary_uncertain=False,
                definition_version="synthetic-test",
                source_video="unused.mp4",
            )
            rows.append(row)
            generator = np.random.default_rng(len(rows))
            n = 6 + i  # Clips differ in length.
            detector, resnet = clip_arrays(generator, label, n)
            np.savez(
                clip_data / "clips" / f"{clip_id}.npz",
                frames=np.arange(n) * 2,
                t_s=(np.arange(n) * 0.2).astype(np.float32),
                detector=detector,
                summary=np.zeros(1, np.float32),
                resnet=resnet,
            )
            index.append([clip_id, "train", match, TACTIC_CLASSES[label], "False", n, n * 0.2])
    with (train / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset" / "test").mkdir()
    # A test manifest that is never opened: reading it would fail.
    (root / "dataset" / "test" / "manifest.csv").write_text("not a manifest", encoding="utf-8")
    (clip_data / "channels.json").write_text(
        json.dumps(
            {
                "data_version": "clip-data-test",
                "regions_version": REGIONS.version,
                "detector": DETECTOR,
                "summary": ["s0"],
                "rows_per_second": 5.0,
            }
        ),
        encoding="utf-8",
    )
    with (clip_data / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip_id", "split", "match", "label", "uncertain", "rows", "duration_s"])
        writer.writerows(index)
    return train / "manifest.csv", clip_data


def train_tcn(folder):
    """A tiny clip-data TCN saved as train-tactics --evaluate-test saves the final one,
    on the same 72 + 40 numbers per row (synthetic matches only)."""
    generator = np.random.default_rng(7)
    rows, dets = [], {}
    for split, match in (("train", "t1"), ("train", "t2"), ("test", "t3")):
        for y in range(5):
            for k in range(2):
                clip_id = f"{match}.{y}.{k}"
                detector, resnet = clip_arrays(generator, y, int(generator.integers(6, 10)))
                dets[clip_id] = {"detector": detector, "resnet": resnet}
                rows.append(
                    dict(clip_id=clip_id, split=split, label=TACTIC_CLASSES[y], y=y, match=match)
                    | {"duration_s": len(detector) / 5}
                )
    cfg = TacticConfig(
        epochs=4,
        eval_every=2,
        seeds=(0, 1),
        channels=8,
        dilations=(1, 2),
        input="clip-data",
        resnet_pcs=3,
    )
    run_tactic_experiment(
        "synthetic", folder, cfg, evaluate_test=True, rows=rows, dets=dets, log=lambda m: None
    )
    return folder / "models"


def new_clip_features(seed=11, clips=3):
    """clip_features-style inputs for new clips: detector, raw ResNet and row times."""
    generator = np.random.default_rng(seed)
    out = {}
    for i in range(clips):
        detector, resnet = clip_arrays(generator, i % 5, FRAMES // 2)
        out[f"new.jx.s{i:03d}"] = {
            "detector": detector,
            "resnet": resnet,
            "t_s": (np.arange(FRAMES // 2) * 0.2).astype(np.float32),
        }
    return out


def quiet(*_):
    return None


class FinalGruTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.manifest, cls.clip_data = write_dataset(cls.root)
        cls.config = BaselineConfig(seeds=(0, 1), epochs=4, patience=4)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = gru_predict.train_final_gru(
                cls.manifest,
                cls.clip_data,
                cls.root / "gru",
                context=RunContext(),
                config=cls.config,
            )
            cls.tcn_models = train_tcn(cls.root / "tcn")
        cls.gru_models = cls.root / "gru" / "models"

    def setUp(self):
        gru_predict.clear_model_cache()
        self.addCleanup(gru_predict.clear_model_cache)

    def predict(self, models, **kwargs):
        kwargs.setdefault("features", new_clip_features())
        return gru_predict.predict_clips(
            [],
            models,
            tcn_models=self.tcn_models,
            gru_models=self.gru_models,
            log=quiet,
            **kwargs,
        )

    def test_epoch_count_is_the_best_mean_leave_one_match_out_epoch(self):
        curves = self.report["curves"]
        self.assertEqual(sorted(curves), ["aurora", "furia", "mouz"])
        per_match = [np.mean(c, 0) for c in curves.values()]
        mean = np.mean(per_match, 0)
        self.assertEqual(self.report["chosen_epoch"], int(np.argmax(mean)) + 1)
        self.assertEqual(gru_predict.final_epoch({"a": [[0.1, 0.5, 0.5]]})[0], 2)  # Earliest tie.
        saved = torch.load(self.gru_models / "gru-seed0.pt", weights_only=True)
        self.assertEqual(saved["epochs"], self.report["chosen_epoch"])
        self.assertEqual(saved["classes"], list(TACTIC_CLASSES))
        self.assertEqual(saved["regions_version"], REGIONS.version)
        self.assertEqual(len(saved["train_clip_ids"]), 18)
        self.assertTrue(torch.equal(saved["resnet_pca"]["mean"], saved["state_dict"]["pca_mean"]))
        self.assertEqual(saved["settings"]["gru"]["readout"], "mean-max")  # Compact pooled GRU.

    def test_a_saved_and_reloaded_gru_predicts_identically(self):
        saved = torch.load(self.gru_models / "gru-seed0.pt", weights_only=True)
        clip_ids = saved["train_clip_ids"]
        index = {r["clip_id"]: r for r in self._index_rows()}
        frames = load_clip_rows(self.clip_data, clip_ids)
        targets = torch.tensor([TACTIC_CLASSES.index(index[c]["label"]) for c in clip_ids])
        inputs, build_head, _ = clip_data_inputs("gru", frames, TACTIC_CLASSES)
        fitted = gru_predict.fit_all(
            inputs,
            targets,
            epochs=3,
            classes=TACTIC_CLASSES,
            config=self.config,
            context=RunContext(),
            build_head=build_head,
            match_ids=[index[c]["match"] for c in clip_ids],
        )
        folder = self.root / "reload" / "models"
        provenance = {k: saved[k] for k in ("regions_version", "data_version", "detector_names")}
        provenance |= {"train_clip_ids": [], "excluded_clip_ids": [], "manifest_sha256": "x"}
        for seed, head, _ in fitted:
            checkpoint = gru_predict.gru_checkpoint(
                head, seed=seed, epochs=3, settings=saved["settings"], provenance=provenance
            )
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, folder / f"gru-seed{seed}.pt")
        reloaded = gru_predict.load_gru(folder)
        features = list(new_clip_features().values())
        with torch.no_grad():
            for (seed, head, _), (again, info) in zip(fitted, reloaded):
                self.assertEqual(info["seed"], seed)
                for f in features:
                    rows = gru_predict.gru_rows(f)
                    self.assertTrue(torch.equal(head(rows), again(rows)))
        # The saved run reproduces its own training-fit probabilities after reloading.
        members = gru_predict.load_gru(self.gru_models)
        clips = {
            c: {
                "detector": frames["features"][i, :n, : len(DETECTOR)].numpy(),
                "resnet": frames["features"][i, :n, len(DETECTOR) :].numpy(),
                "t_s": frames["timestamps_s"][i, :n].numpy().astype(np.float32),
            }
            for i, (c, n) in enumerate((s["clip_id"], s["rows"]) for s in frames["samples"])
        }
        for member in members:
            probs = gru_predict.gru_probs([member], list(clips.values()))
            expected = self.report["training_fit_probabilities"][str(member[1]["seed"])]
            np.testing.assert_allclose(probs, [expected[c] for c in clips], atol=1e-6)

    def _index_rows(self):
        with (self.clip_data / "index.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_reloading_checks_regions_settings_and_format(self):
        other = self.root / "regions-other.json"
        raw = json.loads(REGIONS_PATH.read_text(encoding="utf-8"))
        other.write_text(json.dumps({**raw, "version": "mirage-other"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "map regions"):
            gru_predict.load_gru(self.gru_models, regions_path=other)
        mixed = self.root / "mixed"
        shutil.copytree(self.gru_models, mixed)
        changed = torch.load(mixed / "gru-seed1.pt", weights_only=True)
        changed["epochs"] += 1
        (mixed / "gru-seed1.pt").unlink()
        torch.save(changed, mixed / "gru-seed1.pt")
        with self.assertRaisesRegex(ValueError, "mixes"):
            gru_predict.load_gru(mixed)
        changed["format"] = "something-else"
        (mixed / "gru-seed1.pt").unlink()
        torch.save(changed, mixed / "gru-seed1.pt")
        with self.assertRaisesRegex(ValueError, "not a final GRU"):
            gru_predict.load_gru(mixed)
        with self.assertRaisesRegex(ValueError, "no gru-seed"):
            gru_predict.load_gru(self.root)

    def test_the_gru_and_tcn_receive_identical_features(self):
        features = new_clip_features(clips=1)
        name, clip = next(iter(features.items()))
        with (
            patch.object(
                tcn_predict, "ensemble_probs", wraps=tcn_predict.ensemble_probs
            ) as tcn_call,
            patch.object(gru_predict, "gru_rows", wraps=gru_predict.gru_rows) as gru_call,
        ):
            self.predict(("tcn", "gru"), features=features)
        (tcn_input,) = tcn_call.call_args.args[1]
        self.assertIs(gru_call.call_args.args[0], clip)  # The GRU packs this clip's features.
        rows = gru_predict.gru_rows(clip)[0]
        width = len(DETECTOR)
        # Before their own PCA and scaling, both see the same detector and raw ResNet rows.
        np.testing.assert_array_equal(tcn_input["detector"], rows[:, :width].numpy())
        np.testing.assert_array_equal(
            np.asarray(tcn_input["resnet"], np.float32), rows[:, width : width + RESNET].numpy()
        )
        np.testing.assert_allclose(rows[:, width + RESNET].numpy(), clip["t_s"] / TIME_SCALE_S)
        self.assertEqual(sorted(tcn_input), ["detector", "resnet"])

    def test_the_tcn_result_is_the_same_with_or_without_the_gru(self):
        tcn_only = self.predict(("tcn",))
        both = self.predict(("tcn", "gru"))
        gru_only = self.predict(("gru",))
        self.assertEqual(
            [r["predictions"]["tcn"] for r in tcn_only], [r["predictions"]["tcn"] for r in both]
        )
        self.assertEqual(
            [r["predictions"]["gru"] for r in gru_only], [r["predictions"]["gru"] for r in both]
        )
        self.assertEqual(list(tcn_only[0]["predictions"]), ["tcn"])
        # The default is the TCN alone: the GRU folder is not even opened.
        default = gru_predict.predict_clips(
            [],
            tcn_models=self.tcn_models,
            gru_models=self.root / "missing",
            features=new_clip_features(),
            log=quiet,
        )
        self.assertEqual(default, tcn_only)
        with self.assertRaisesRegex(ValueError, "distinct names"):
            self.predict(("tcn", "lstm"))

    def test_results_are_plain_json(self):
        results = self.predict(("tcn", "gru"), min_confidence=0.99)
        self.assertEqual(json.loads(json.dumps(results)), results)
        for clip in results:
            self.assertEqual(sorted(clip), ["clip", "file", "predictions", "warnings"])
            for prediction in clip["predictions"].values():
                probs = prediction["probabilities"]
                self.assertEqual(list(probs), list(TACTIC_CLASSES))
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
                self.assertEqual(
                    probs[prediction["tactic"]], max(probs.values())
                )  # Rounded ties allowed.
                self.assertIs(prediction["uncertain"], prediction["confidence"] < 0.99)

    def video_clips(self, count=3):
        folder = self.root / f"uploads-{count}"
        folder.mkdir(exist_ok=True)
        paths = []
        for i in range(count):
            path = folder / f"new.jx.s{i:03d}.mp4"
            path.write_bytes(b"")
            paths.append(path)
        return paths

    def fake_inputs(self):
        """Stand-ins for video reading: detections and tcn_predict.clip_inputs rows."""
        features = new_clip_features()

        def detect_clips(paths, background_from=None, log=print):
            return {p.stem: fake_det() for p in paths}

        def clip_inputs(paths, dets, members, regions_path=None, device="cpu", log=print):
            return {
                p.stem: {
                    "detector": features[p.stem]["detector"],
                    "resnet": features[p.stem]["resnet"],
                }
                for p in paths
            }

        return (
            patch.object(tcn_predict, "detect_clips", side_effect=detect_clips),
            patch.object(tcn_predict, "clip_inputs", side_effect=clip_inputs),
        )

    def test_predict_clips_matches_predict_tcn_and_predict_gru(self):
        paths = self.video_clips()
        first, second = self.fake_inputs()
        with first, second, contextlib.redirect_stdout(io.StringIO()):
            results = gru_predict.predict_clips(
                paths,
                ("tcn", "gru"),
                tcn_models=self.tcn_models,
                gru_models=self.gru_models,
                log=quiet,
            )
            tcn_csv, gru_csv = self.root / "tcn.csv", self.root / "gru.csv"
            main(
                ["predict-tcn", *map(str, paths), "--models", str(self.tcn_models)]
                + ["--output", str(tcn_csv)]
            )
            main(
                ["predict-gru", *map(str, paths), "--models", str(self.gru_models)]
                + ["--with-tcn", "--tcn-models", str(self.tcn_models), "--output", str(gru_csv)]
            )
        tcn_rows = list(csv.DictReader(tcn_csv.read_text(encoding="utf-8").splitlines()))
        gru_rows = list(csv.DictReader(gru_csv.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(list(gru_rows[0])[0], "model")
        self.assertEqual([r["model"] for r in gru_rows], ["tcn"] * 3 + ["gru"] * 3)
        for kind, rows in (("tcn", tcn_rows), ("tcn", gru_rows[:3]), ("gru", gru_rows[3:])):
            for result, row in zip(results, rows, strict=True):
                self.assertEqual(result["clip"], row["clip"])
                got = result["predictions"][kind]
                self.assertEqual(got["tactic"], row["predicted"])
                self.assertEqual(str(got["uncertain"]), row["uncertain"])
                for c in TACTIC_CLASSES:
                    self.assertEqual(got["probabilities"][c], float(row[f"p_{c}"]))
        # Three clips of one match build their own background: below five, they are warned.
        self.assertIn("background from only 3", results[0]["warnings"][0])

    def test_a_single_clip_works_with_a_saved_background(self):
        dataset = self.manifest.parent.parent
        png = self.root / "aurora-background.png"
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                ["save-background", "--dataset", str(dataset), "--match", "aur"]
                + ["--output", str(png)]
            )
        training = sorted((dataset / "train").glob("aurora.*.mp4"))
        expected = match_background(training, per_clip=8)  # As the training detections did.
        np.testing.assert_array_equal(gru_predict.load_background(png), expected)
        for match, output, message in (
            ("aur", png, "already exists"),
            ("aur", self.root / "b.jpg", "png"),
            ("navi", self.root / "b.png", "training matches"),
        ):
            with self.assertRaisesRegex((ValueError, FileExistsError), message):
                gru_predict.save_background(dataset, match, output, log=quiet)
        upload = training[0]  # One clip on its own, of the same layout.
        seen = []

        def detect_clip(path, background):
            seen.append(background)
            return fake_det()

        features = new_clip_features(clips=1)
        clip_inputs = patch.object(
            tcn_predict,
            "clip_inputs",
            side_effect=lambda paths, *a, **k: {
                paths[0].stem: {k2: features["new.jx.s000"][k2] for k2 in ("detector", "resnet")}
            },
        )
        with patch("stratscout.tactic_features.detect_clip", side_effect=detect_clip), clip_inputs:
            from_path = self.predict_upload([upload], background=png)
            from_array = self.predict_upload(upload, background=expected)
            with self.assertRaisesRegex(ValueError, "same layout"):
                self.predict_upload([upload], background=expected[:16])
            with self.assertRaisesRegex(ValueError, "not both"):
                self.predict_upload([upload], background=png, background_from=[upload])
        for results in (from_path, from_array):
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["clip"], upload.stem)
            self.assertEqual(results[0]["warnings"], [])
        self.assertEqual(from_path, from_array)
        self.assertEqual(len(seen), 2)
        for background in seen:
            np.testing.assert_array_equal(background, expected)

    def predict_upload(self, paths, **kwargs):
        return gru_predict.predict_clips(
            paths,
            ("tcn", "gru"),
            tcn_models=self.tcn_models,
            gru_models=self.gru_models,
            log=quiet,
            **kwargs,
        )

    def test_saved_backgrounds_read_training_clips_only(self):
        dataset = self.manifest.parent.parent
        clips = gru_predict.training_match_clips(dataset, "furia")
        self.assertEqual(len(clips), 6)
        bad = self.root / "bad"
        (bad / "train").mkdir(parents=True)
        text = self.manifest.read_text(encoding="utf-8").replace(",train,", ",test,", 1)
        (bad / "train" / "manifest.csv").write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-training clip"):
            gru_predict.training_match_clips(bad, "furia")


if __name__ == "__main__":
    unittest.main()
