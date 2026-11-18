"""Classify new tactical clips with models saved by train-tactics --evaluate-test.

Clips must be prepared like the training clips: the spectator-radar crop at 10 fps, cut
from the start to the end of one tactic (export-segments output is ready to use). Every
saved seed of the chosen model kind is run and their probabilities are averaged, as in
the report's seed ensemble. A prediction whose top probability is below min_confidence
is flagged as uncertain rather than trusted.

The empty-map background is the median of frames from one match's clips, as in training.
Clips are grouped into matches by the part of the file name before the first dot (how
export-segments names them), unless background_from gives the frames to use instead.
Importing this module reads nothing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import TACTIC_CLASSES
from .tactic_features import feature_dim
from .tactic_tcn import Normaliser, TacticConfig, _eval_features, _write_csv, build_model, predict

TRAINED_FPS = 10.0
BACKGROUND_FRAMES = 160  # about 8 frames from each of ~20 clips, as for a training match
FEW_BACKGROUND_CLIPS = 5


def load_ensemble(models_dir, kind: str = "tcn", device: str = "cpu"):
    """Every <kind>-seed*.pt in models_dir -> [(model, normaliser, config)]."""
    import torch

    paths = sorted(Path(models_dir).glob(f"{kind}-seed*.pt"))
    if not paths:
        raise ValueError(
            f"no {kind}-seed*.pt in {models_dir}; train with --evaluate-test to save models"
        )
    members = []
    for path in paths:
        # The checkpoint holds numpy arrays, so it is not weights-only: load only
        # checkpoints this project wrote.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if list(ckpt["classes"]) != list(TACTIC_CLASSES):
            raise ValueError(f"{path} was trained on different classes: {ckpt['classes']}")
        raw = ckpt["config"]
        cfg = TacticConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        model = build_model(ckpt["kind"], feature_dim(cfg.grid), cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        members.append((model, Normaliser(ckpt["norm_mean"], ckpt["norm_std"]), cfg))
    return members


def ensemble_probs(members, dets: Sequence[Dict], device: str = "cpu") -> np.ndarray:
    """Mean class probabilities over the ensemble, (clips, classes)."""
    return np.mean(
        [predict(model, _eval_features(dets, norm, cfg), device) for model, norm, cfg in members],
        0,
    )


def find_clips(paths: Sequence[Path]) -> List[Path]:
    """Video files given directly, plus every .mp4 under any folder given."""
    clips = []
    for path in map(Path, paths):
        if path.is_dir():
            # By name, not by folder, so an exported dataset's class folders do not
            # order (and so hint at) the predictions.
            clips += sorted(path.rglob("*.mp4"), key=lambda p: p.name)
        elif path.is_file():
            clips.append(path)
        else:
            raise ValueError(f"{path} does not exist")
    if not clips:
        raise ValueError("no .mp4 clips found")
    names = [c.stem for c in clips]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"clip names must be unique; repeated: {', '.join(duplicates)}")
    return clips


def match_of(clip: Path) -> str:
    return clip.name.split(".", 1)[0]


def detect_clips(
    clips: Sequence[Path], background_from: Optional[Sequence[Path]] = None, log=print
) -> Dict[str, Dict]:
    """Detections per clip name, with one background per match (or one from
    background_from for every clip)."""
    from .tactic_features import detect_clip, match_background

    if background_from:
        groups = {"(--background-from)": list(clips)}
        sources = {"(--background-from)": find_clips(background_from)}
    else:
        groups = defaultdict(list)
        for clip in clips:
            groups[match_of(clip)].append(clip)
        sources = groups
    out = {}
    for name, members in groups.items():
        source = sources[name]
        if len(source) < FEW_BACKGROUND_CLIPS:
            log(
                f"  warning: background for {name} uses only {len(source)} clip(s); players "
                "who stand still may be mistaken for the map. Pass more clips from the same "
                "match, or the prepared recording with --background-from."
            )
        per_clip = max(8, math.ceil(BACKGROUND_FRAMES / len(source)))
        log(f"  background for {name} from {len(source)} video(s)")
        background = match_background(list(source), per_clip=per_clip)
        for i, clip in enumerate(members, 1):
            log(f"  detecting {name} {i}/{len(members)}: {clip.name}")
            det = detect_clip(clip, background)
            if abs(float(det["fps"]) - TRAINED_FPS) > 0.5:
                log(
                    f"  warning: {clip.name} is {float(det['fps']):.1f} fps; the models were "
                    f"trained on {TRAINED_FPS:.0f} fps clips"
                )
            out[clip.stem] = det
    return out


def prediction_rows(
    names: Sequence[str], groups: Sequence[str], probs: np.ndarray, min_confidence: float
) -> List[Dict]:
    rows = []
    for name, group, p in zip(names, groups, probs):
        first, second = np.argsort(p)[::-1][:2]
        rows.append(
            {
                "clip": name,
                "match": group,
                "predicted": TACTIC_CLASSES[first],
                "confidence": round(float(p[first]), 4),
                "runner_up": TACTIC_CLASSES[second],
                "runner_up_p": round(float(p[second]), 4),
                "uncertain": bool(p[first] < min_confidence),
                **{f"p_{c}": round(float(v), 4) for c, v in zip(TACTIC_CLASSES, p)},
            }
        )
    return rows


def format_rows(rows: List[Dict]) -> str:
    width = max(len(r["clip"]) for r in rows)
    lines = []
    for r in rows:
        line = f"{r['clip']:<{width}}  {r['predicted']:<8}  {r['confidence']:.2f}"
        if r["uncertain"]:
            line += f"  uncertain (next: {r['runner_up']} {r['runner_up_p']:.2f})"
        lines.append(line)
    counts = {c: sum(r["predicted"] == c for r in rows) for c in TACTIC_CLASSES}
    unsure = sum(r["uncertain"] for r in rows)
    lines += [
        "",
        f"{len(rows)} clips: " + ", ".join(f"{c} {n}" for c, n in counts.items() if n),
        f"{unsure} below the confidence threshold",
    ]
    return "\n".join(lines)


def run_predict(
    clips: Sequence[Path],
    models_dir,
    *,
    kind: str = "tcn",
    output: Optional[Path] = None,
    background_from: Optional[Sequence[Path]] = None,
    min_confidence: float = 0.6,
    device: str = "cpu",
    log=print,
    dets: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """Predict every clip; print a table and, with output, write a new CSV.
    dets (clip name -> detections) skips video reading, for tests."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"{output} already exists; choose a new file name")
    members = load_ensemble(models_dir, kind, device)
    log(f"Loaded {len(members)} {kind} model(s) from {models_dir}")
    if dets is None:
        paths = find_clips(clips)
        dets = detect_clips(paths, background_from, log=log)
        names = [p.stem for p in paths]
        groups = ["" if background_from else match_of(p) for p in paths]
    else:
        names = list(dets)
        groups = [name.split(".", 1)[0] for name in names]
    probs = ensemble_probs(members, [dets[n] for n in names], device)
    rows = prediction_rows(names, groups, probs, min_confidence)
    log(format_rows(rows))
    if output is not None:
        _write_csv(Path(output), rows)
        log(f"Predictions: {output}")
    return rows
