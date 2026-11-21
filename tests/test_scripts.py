"""The helper scripts in scripts/ still import, and the tactic_tcn code they call exists."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from stratscout import tactic_tcn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# Everything the scripts take from tactic_tcn (ri3n's module); renaming any breaks them.
USED = {
    "shuffle_tcn.py": ("TacticConfig", "_load_clip_inputs", "cross_validate"),
    "nested_tcn.py": (
        "Normaliser",
        "TacticConfig",
        "_eval_features",
        "_load_clip_inputs",
        "_raw_features",
        "class_weights",
        "classification_metrics",
        "cross_validate",
        "predict",
        "train_model",
    ),
    "tcn_curves.py": (
        "TacticConfig",
        "_eval_features",
        "_load_clip_inputs",
        "class_weights",
        "classification_metrics",
        "pad_batch",
        "train_model",
    ),
}


def load(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Importing runs nothing: work happens in main().
    return module


class ScriptTests(unittest.TestCase):
    def test_python_scripts_import_without_running_and_have_a_main(self):
        for name in USED:
            with self.subTest(script=name):
                module = load(name)
                self.assertTrue(callable(module.main))
                self.assertIn(f"python scripts/{name}", module.__doc__)  # Usage line.

    def test_the_tactic_tcn_functions_they_use_still_exist(self):
        for name, functions in USED.items():
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            for function in functions:
                with self.subTest(script=name, function=function):
                    self.assertTrue(hasattr(tactic_tcn, function))
                    self.assertIn(function, source)
        cross_validate_result = ("final_epoch", "cross_fold_macro_f1", "pooled", "runs")
        source = (Path(tactic_tcn.__file__)).read_text(encoding="utf-8")
        for key in cross_validate_result:  # Keys the scripts read from cross_validate.
            self.assertIn(f'"{key}"', source)

    def test_tcn_curves_relies_on_the_validate_hook_running_every_eval_every_epochs(self):
        # tcn_curves.py sets eval_every=1 to record every epoch without editing tactic_tcn.
        source = (Path(tactic_tcn.__file__)).read_text(encoding="utf-8")
        self.assertIn("epoch % cfg.eval_every == 0", source)
        self.assertIn("eval_every", tactic_tcn.TacticConfig.__dataclass_fields__)
        self.assertIn(
            "replace(cfg, eval_every=1)", (SCRIPTS / "tcn_curves.py").read_text(encoding="utf-8")
        )

    def test_the_shell_script_uses_the_clip_data_input(self):
        text = (SCRIPTS / "clipdata_temporal_20261120.sh").read_text(encoding="utf-8")
        for part in ("--inputs clip-data", "--epoch-mode", "2026-11-16-train-only", "Usage"):
            self.assertIn(part, text)
        self.assertIn("set -euo pipefail", text)  # Stops at the first failed run.
        self.assertIn("TAG=${1:-20261120}", text)  # New outputs for a rerun.

    def test_tcn_curves_reads_training_clips_only(self):
        module = load("tcn_curves.py")
        header = "clip_id,split,label,match_folder,duration_s,boundary_uncertain,clip\n"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").mkdir()
            (root / "train" / "manifest.csv").write_text(
                header + "m.jx.s001,train,A_SPLIT,m,10.0,False,clips/a.mp4\n", encoding="utf-8"
            )
            rows = module.training_rows(root)
            self.assertEqual(
                [(r["clip_id"], r["y"], r["match"]) for r in rows], [("m.jx.s001", 2, "m")]
            )
            (root / "test").mkdir()  # A test manifest is never opened, even if malformed.
            (root / "test" / "manifest.csv").write_text("not a manifest", encoding="utf-8")
            self.assertEqual(len(module.training_rows(root)), 1)
            (root / "train" / "manifest.csv").write_text(
                header + "m.jx.s002,test,A_SPLIT,m,10.0,False,clips/b.mp4\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-training clip"):
                module.training_rows(root)
            index = root / "clips-data"
            index.mkdir()
            (index / "index.csv").write_text(
                "clip_id,split\nm.jx.s001,train\nm.jx.s009,test\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not training clips"):
                module.check_training_index(index, rows)


if __name__ == "__main__":
    unittest.main()
