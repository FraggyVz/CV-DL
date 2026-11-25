"""TCN prediction (predict-tcn): classify new tactical clips with the TCN seeds saved by
train-tactics --evaluate-test. The GRU and temporal models have their own loader in
tactic_temporal.py; this module serves the TCN only.

Works for both TCN inputs: the shared clip data (--clip-data, the current model) and the
original detection features. For clip-data models, each new clip goes through the same
detector, map regions and ResNet-18 as build-clip-data, and each seed applies its own
saved ResNet compression and scaling. The map-regions file must be the version the
models were trained with.

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
        compressor = None
        if ckpt.get("resnet_pca") is not None:
            from .clip_data import ResnetCompressor

            compressor = ResnetCompressor(len(ckpt["resnet_pca"]["components"]))
            compressor.mean = np.asarray(ckpt["resnet_pca"]["mean"], np.float32)
            compressor.components = np.asarray(ckpt["resnet_pca"]["components"], np.float32)
        norm = Normaliser(ckpt["norm_mean"], ckpt["norm_std"], compressor)
        model = build_model(ckpt["kind"], len(norm.mean), cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        members.append((model, norm, cfg))
    inputs = {cfg.input for _, _, cfg in members}
    if len(inputs) != 1:
        raise ValueError(f"{models_dir} mixes models trained on different inputs: {inputs}")
    return members


def trained_regions_version(models_dir) -> Optional[str]:
    """The map-regions version a clip-data model was trained with, from the run's
    report.json next to its models/ folder, or None if the report is missing."""
    import json

    report = Path(models_dir).parent / "report.json"
    if not report.exists():
        return None
    info = json.loads(report.read_text(encoding="utf-8")).get("clip_data") or {}
    return info.get("regions_version")


def clip_inputs(
    paths: Sequence[Path],
    dets: Dict[str, Dict],
    members,
    regions_path=None,
    device: str = "cpu",
    log=print,
) -> Dict[str, Dict]:
    """Detections -> the shared clip-data rows a clip-data model reads: the detector
    channels per map region and, if the models use them, ResNet rows of the same frames.
    Built exactly as build-clip-data does, so a clip predicts as it would from that data."""
    from .clip_data import (
        REGIONS_PATH,
        Regions,
        _crop_size,
        detector_channels,
        resnet_encoder,
        resnet_rows,
    )

    regions = Regions.load(regions_path or REGIONS_PATH)
    cfg = members[0][2]
    use_resnet = any(norm.compressor is not None for _, norm, _ in members)
    encode = resnet_encoder(device) if use_resnet else None
    out = {}
    for i, path in enumerate(paths, 1):
        det = dets[path.stem]
        frames = np.arange(0, len(det["bomb"]), cfg.step)
        resnet, size = None, _crop_size(path, (494, 500))
        if use_resnet:
            log(f"  ResNet {i}/{len(paths)}: {path.name}")
            resnet, size = resnet_rows(path, frames, encode, device=device)
        _, x, _ = detector_channels(det, regions, size, cfg.step)
        out[path.stem] = {"detector": x, "resnet": resnet}
    return out


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
    regions_path=None,
) -> List[Dict]:
    """Predict every clip; print a table and, with output, write a new CSV.
    dets (clip name -> the model's input: detections, or clip-data rows for a clip-data
    model) skips video reading, for tests."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"{output} already exists; choose a new file name")
    members = load_ensemble(models_dir, kind, device)
    clip_data = members[0][2].input == "clip-data"
    log(
        f"Loaded {len(members)} {kind} model(s) from {models_dir} "
        f"(input: {'shared clip data' if clip_data else 'original detection features'})"
    )
    if clip_data:
        from .clip_data import REGIONS_PATH, Regions

        trained = trained_regions_version(models_dir)
        current = Regions.load(regions_path or REGIONS_PATH).version
        if trained is None:
            log(f"  warning: no report.json beside {models_dir}; cannot check the map regions")
        elif trained != current:
            raise ValueError(
                f"the models were trained with map regions {trained}, but the regions file "
                f"is {current}; use the regions file the models were trained with"
            )
    if dets is None:
        paths = find_clips(clips)
        dets = detect_clips(paths, background_from, log=log)
        if clip_data:
            dets = clip_inputs(paths, dets, members, regions_path, device, log)
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
