"""Prediction with the final models on shared clip features, for the web app and the CLI.

The TCN (``runs/tactic-tcn-clipdata-final``) is the selected model and the default. The
Compact pooled GRU (``ClipDataHead`` with the GRU, ``tactic_temporal.GRU_DEFAULTS``) is an
optional second opinion or backup, run only when asked for.

* ``train_final_gru``: choose the GRU's epoch count by leave-one-match-out
  cross-validation on the training matches only (the best mean macro-F1, earliest on
  ties, as train-tactics does for the TCN), then fit each seed on every training clip for
  exactly that many epochs and save it. Only a train-only manifest and clip-data copy are
  read; the test split is never opened.
* ``load_gru``: reload those checkpoints, refusing a different map-regions version,
  channel layout, class order or mixed settings.
* ``predict_clips``: compute each clip's features once with tcn_predict's detection and
  clip-feature code, run each requested model (seeds averaged) and return plain,
  JSON-ready results. The TCN's numbers never depend on whether the GRU also runs.
* ``save_background`` / ``load_background``: an empty-map background image built from
  the training clips of one match, so a single uploaded clip is detected as in training.

Models are loaded once and cached. Importing this module reads and trains nothing.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import tcn_predict
from .config import TACTIC_CLASSES

REPO = Path(__file__).resolve().parents[1]
TCN_MODELS = REPO / "runs" / "tactic-tcn-clipdata-final" / "models"
GRU_MODELS = REPO / "runs" / "tactic-gru-clipdata-final-20261121" / "models"
MODEL_KINDS = ("tcn", "gru")
FORMAT = "stratscout-clipdata-gru-1"
DISPLAY_NAMES = {"tcn": "TCN", "gru": "Compact pooled GRU"}
BACKGROUND_CLIPS_PER_FRAME = 8  # match_background's default, as for the training detections


# --------------------------------------------------------------------------- training


def gru_curves(inputs, targets, folds, *, classes, config, context, build_head, names=None) -> Dict:
    """Validation macro-F1 after every epoch for each held-out match and seed: every fold
    trained for all epochs, with no early stopping. {match: [[per epoch] per seed]}, keyed
    by ``names[match_id]`` (e.g. the match folder) when given."""
    from .tactic_training import fit_classifier

    full = replace(config, patience=config.epochs, fixed_epochs=None)
    curves = {}
    for fold in folds:
        match = (names or {}).get(fold.validation_match_id, fold.validation_match_id)
        curves[match] = []
        for seed in config.seeds:
            _, run, _ = fit_classifier(
                inputs,
                targets,
                fold,
                classes=classes,
                config=full,
                seed=seed,
                context=context,
                build_head=build_head,
            )
            curves[match].append([row["validation_macro_f1"] for row in run["history"]])
            print(
                f"  held out {match}, seed {seed}: best macro-F1 {max(curves[match][-1]):.3f}",
                flush=True,
            )
    return curves


def final_epoch(curves: Dict) -> tuple:
    """(1-based epoch, mean curve): each match's curve averaged over seeds, then the
    matches averaged equally; the best epoch, earliest on ties (tactic_tcn.cross_validate)."""
    per_match = [np.mean(np.asarray(c, np.float64), 0) for c in curves.values()]
    mean = np.mean(per_match, 0)
    return int(np.argmax(mean)) + 1, mean


def fit_all(inputs, targets, *, epochs, classes, config, context, build_head, match_ids):
    """One head per seed, trained on every clip for exactly ``epochs`` epochs with the
    protocol's own fit_classifier. Returns [(seed, head, probabilities on the training
    clips)]; those probabilities describe the fit, they are not a validation score."""
    from .tactic_data import MatchFold
    from .tactic_training import fit_classifier

    every = tuple(range(len(targets)))
    # fit_classifier scores its validation clips only after the last update in fixed-epoch
    # mode, so pointing them at the training clips changes nothing about the fit.
    fold = MatchFold(every, every, tuple(sorted(set(match_ids))), "all-training-matches")
    out = []
    for seed in config.seeds:
        head, _, probabilities = fit_classifier(
            inputs,
            targets,
            fold,
            classes=classes,
            config=replace(config, fixed_epochs=epochs),
            seed=seed,
            context=context,
            build_head=build_head,
        )
        out.append((seed, head.cpu().eval(), probabilities))
    return out


def gru_checkpoint(head, *, seed: int, epochs: int, settings: Dict, provenance: Dict) -> Dict:
    """Everything needed to rebuild and check one GRU seed. Tensors and plain values only,
    so it loads with torch.load(weights_only=True)."""
    if not bool(head.fitted):
        raise ValueError("The head's ResNet PCA and scaling were never fitted")
    return {
        "format": FORMAT,
        "kind": "gru",
        "display_name": DISPLAY_NAMES["gru"],
        "classes": list(TACTIC_CLASSES),
        "seed": int(seed),
        "epochs": int(epochs),
        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        # Also inside state_dict; kept by name so readers see the fitted transform.
        "resnet_pca": {
            "mean": head.pca_mean.detach().cpu().clone(),
            "components": head.components.detach().cpu().clone(),
        },
        "scaling": {
            "mean": head.row_mean.detach().cpu().clone(),
            "std": head.row_std.detach().cpu().clone(),
        },
        "settings": settings,
        **provenance,
    }


def train_final_gru(
    manifest,
    clip_data,
    output,
    *,
    context,
    config=None,
    exclude_uncertain: bool = True,
):
    """Choose the epoch by leave-one-match-out CV, fit every seed on all training clips and
    save models/gru-seed<N>.pt, report.json and report.md in a new ``output`` folder.

    ``manifest`` is a train/manifest.csv; ``clip_data`` a train-only build-clip-data copy
    (a folder listing any other clip is refused before a clip file is opened).
    """
    from .storage import save_checkpoint, write_new_text
    from .tactic_model import file_sha256
    from .tactic_temporal import GRU_DEFAULTS, TIME_SCALE_S, clip_data_inputs, load_clip_rows
    from .tactic_training import BaselineConfig, prepare_run

    config = config or BaselineConfig()
    output = Path(output)
    dataset, targets, folds, _ = prepare_run(
        manifest,
        output,
        context=context,
        config=config,
        exclude_uncertain=exclude_uncertain,
        epoch_mode="other-folds",  # Outer folds only; no fold's epoch is reported here.
    )
    classes = dataset.classes
    if list(classes) != list(TACTIC_CLASSES):
        raise ValueError(f"class order {classes} differs from {TACTIC_CLASSES}")
    clip_ids = [r.clip_id for r in dataset.records]
    frames = load_clip_rows(clip_data, clip_ids)  # Read before any output is written.
    info = frames["clip_data"]
    channels = json.loads((Path(clip_data) / "channels.json").read_text(encoding="utf-8"))
    step = int(round(tcn_predict.TRAINED_FPS / float(channels["rows_per_second"])))
    inputs, build_head, description = clip_data_inputs("gru", frames, classes, GRU_DEFAULTS)
    output.mkdir(parents=True, exist_ok=False)
    write_new_text(output / "input-manifest.csv", dataset.manifest.read_text(encoding="utf-8-sig"))
    started = time.time()
    print(
        f"Compact pooled GRU on {len(clip_ids)} training clips from {len(folds)} matches; "
        f"excluded {list(dataset.excluded_clip_ids)}",
        flush=True,
    )
    print(f"Leave-one-match-out curves ({config.epochs} epochs, seeds {config.seeds})", flush=True)
    curves = gru_curves(
        inputs,
        targets,
        folds,
        classes=classes,
        config=config,
        context=context,
        build_head=build_head,
        names={r.match_id: r.match_folder for r in dataset.records},
    )
    epochs, mean = final_epoch(curves)
    print(f"Chosen epoch count: {epochs} (mean macro-F1 {mean[epochs - 1]:.3f})", flush=True)
    match_ids = [r.match_id for r in dataset.records]
    fitted = fit_all(
        inputs,
        targets,
        epochs=epochs,
        classes=classes,
        config=config,
        context=context,
        build_head=build_head,
        match_ids=match_ids,
    )
    first = fitted[0][1]
    settings = {
        "gru": dict(GRU_DEFAULTS),
        "resnet_pcs": int(first.pcs),
        "detector_width": int(first.detector_width),
        "resnet_width": int(first.width - first.detector_width),
        "time_scale_s": TIME_SCALE_S,
        "step": step,
        "rows_per_second": float(channels["rows_per_second"]),
        "training": {**asdict(config), "fixed_epochs": epochs},
    }
    provenance = {
        "regions_version": info["regions_version"],
        "data_version": info["data_version"],
        "detector_names": list(frames["feature_names"][: frames["detector_width"]]),
        "train_clip_ids": clip_ids,
        "excluded_clip_ids": list(dataset.excluded_clip_ids),
        "manifest_sha256": file_sha256(dataset.manifest),
    }
    for seed, head, _ in fitted:
        save_checkpoint(
            gru_checkpoint(
                head, seed=seed, epochs=epochs, settings=settings, provenance=provenance
            ),
            output / "models" / f"gru-seed{seed}.pt",
        )
    report = {
        "model": DISPLAY_NAMES["gru"],
        "method": "Epoch count: leave-one-match-out on the training matches only, every fold "
        "trained for all epochs; each match's curve averaged over seeds, matches averaged "
        "equally, best epoch (earliest on ties), as train-tactics chooses the TCN's. Then each "
        "seed trained on every training clip for exactly that many epochs.",
        "manifest": str(dataset.manifest),
        "clip_data": info,
        "clips": len(clip_ids),
        "excluded_clip_ids": list(dataset.excluded_clip_ids),
        "settings": settings,
        "description": description,
        "trainable_parameters": sum(p.numel() for p in first.parameters()),
        "chosen_epoch": epochs,
        "chosen_epoch_mean_macro_f1": float(mean[epochs - 1]),
        "note": "The mean macro-F1 at the chosen epoch is optimistic: the epoch was picked on "
        "the same held-out matches. It is not a test score; no test clip was read.",
        "mean_curve": [float(v) for v in mean],
        "curves": {m: [[float(v) for v in c] for c in seeds] for m, seeds in curves.items()},
        "training_fit_probabilities": {
            str(seed): {c: [round(float(v), 7) for v in p] for c, p in zip(clip_ids, probs)}
            for seed, _, probs in fitted
        },
        "seconds": round(time.time() - started, 1),
    }
    write_new_text(output / "report.json", json.dumps(report, indent=1) + "\n")
    write_new_text(output / "report.md", render_report(report))
    print(
        f"Saved {len(fitted)} seeds in {output / 'models'} ({report['seconds']:.0f}s)", flush=True
    )
    return report


def render_report(report: Dict) -> str:
    curve = report["mean_curve"]
    best = report["chosen_epoch"]
    return "\n".join(
        [
            "# Final Compact pooled GRU on shared clip features",
            "",
            f"Trained on {report['clips']} training clips (excluded: "
            f"{', '.join(report['excluded_clip_ids']) or 'none'}); clip data "
            f"{report['clip_data']['data_version']}, regions {report['clip_data']['regions_version']}.",
            f"Settings: {report['settings']['gru']}, {report['settings']['resnet_pcs']} ResNet PCA "
            f"numbers, {report['trainable_parameters']} trainable weights, seeds "
            f"{report['settings']['training']['seeds']}.",
            "",
            report["method"],
            "",
            f"Chosen epoch count: **{best}** (mean leave-one-match-out macro-F1 "
            f"{curve[best - 1]:.3f}; at epochs 25/50/100/200: "
            + ", ".join(f"{curve[e - 1]:.3f}" for e in (25, 50, 100, 200) if e <= len(curve))
            + ").",
            "",
            report["note"],
            "",
        ]
    )


# --------------------------------------------------------------------------- loading


def _plain(value):
    """Tuples to lists, so saved settings compare equal to JSON-style values."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def load_gru(models_dir, *, device: str = "cpu", regions_path=None) -> List:
    """Every gru-seed*.pt in models_dir -> [(head, checkpoint info)], each head in eval mode.

    Refuses checkpoints of another format or class order, a regions file whose version or
    channel layout differs from training, seeds with different settings, and weights that
    do not fit the saved settings.
    """
    import torch

    from .clip_data import REGIONS_PATH, Regions, detector_channel_names
    from .tactic_temporal import GRU_DEFAULTS, GRU_READOUTS, ClipDataHead

    paths = sorted(Path(models_dir).glob("gru-seed*.pt"))
    if not paths:
        raise ValueError(f"no gru-seed*.pt in {models_dir}; run train-gru-final first")
    regions = Regions.load(regions_path or REGIONS_PATH)
    names = detector_channel_names(regions)
    members, shared = [], None
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != FORMAT or ckpt.get("kind") != "gru":
            raise ValueError(f"{path} is not a final GRU checkpoint ({FORMAT})")
        if list(ckpt["classes"]) != list(TACTIC_CLASSES):
            raise ValueError(f"{path} was trained on different classes: {ckpt['classes']}")
        if ckpt["regions_version"] != regions.version:
            raise ValueError(
                f"{path} was trained with map regions {ckpt['regions_version']}, but the regions "
                f"file is {regions.version}; use the regions file the model was trained with"
            )
        if list(ckpt["detector_names"]) != names:
            raise ValueError(f"{path}: the regions file's channels differ from training")
        settings = _plain(ckpt["settings"])
        same = {
            "settings": settings,
            "epochs": ckpt["epochs"],
            "data_version": ckpt["data_version"],
            "train_clip_ids": list(ckpt["train_clip_ids"]),
        }
        if shared is None:
            shared = same
        elif same != shared:
            raise ValueError(f"{models_dir} mixes GRU seeds with different settings or data")
        sizes = settings["gru"]
        if set(sizes) != set(GRU_DEFAULTS) or sizes["readout"] not in GRU_READOUTS:
            raise ValueError(f"{path}: unexpected GRU settings {sizes}")
        if settings["detector_width"] != len(names):
            raise ValueError(f"{path}: {settings['detector_width']} detector channels saved")
        head = ClipDataHead(
            "gru",
            settings["detector_width"],
            settings["resnet_width"],
            len(TACTIC_CLASSES),
            pcs=settings["resnet_pcs"],
            gru=sizes,
        )
        try:
            head.load_state_dict(ckpt["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"{path}: the weights do not match the saved settings") from exc
        state = ckpt["state_dict"]
        named = {
            "pca_mean": ckpt["resnet_pca"]["mean"],
            "components": ckpt["resnet_pca"]["components"],
            "row_mean": ckpt["scaling"]["mean"],
            "row_std": ckpt["scaling"]["std"],
        }
        if not bool(state["fitted"]) or any(not torch.equal(state[k], v) for k, v in named.items()):
            raise ValueError(f"{path}: the saved ResNet PCA or scaling is missing or inconsistent")
        info = {k: v for k, v in ckpt.items() if k not in ("state_dict", "resnet_pca", "scaling")}
        members.append((head.to(device).eval(), {**info, "settings": settings, "path": str(path)}))
    return members


_CACHE: Dict[tuple, List] = {}


def _cache_key(kind, models_dir, device, regions_path):
    folder = Path(models_dir).resolve()
    files = tuple(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(folder.glob(f"{kind}-seed*.pt"))
    )
    regions = str(Path(regions_path).resolve()) if regions_path else None
    return kind, str(folder), files, device, regions


def load_models(kind: str, models_dir=None, *, device: str = "cpu", regions_path=None) -> List:
    """The checked ensemble for one model kind, loaded once and then served from a cache
    (reloaded if the checkpoint files change). TCN members are (model, normaliser, config);
    GRU members are (head, checkpoint info)."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model {kind!r}; choose from {', '.join(MODEL_KINDS)}")
    models_dir = Path(models_dir or (TCN_MODELS if kind == "tcn" else GRU_MODELS))
    key = _cache_key(kind, models_dir, device, regions_path)
    if key not in _CACHE:
        if kind == "gru":
            _CACHE[key] = load_gru(models_dir, device=device, regions_path=regions_path)
        else:
            members = tcn_predict.load_ensemble(models_dir, "tcn", device)
            if members[0][2].input != "clip-data":
                raise ValueError(f"{models_dir}: predict_clips serves clip-data TCN models only")
            _check_tcn_regions(models_dir, regions_path)
            _CACHE[key] = members
    return _CACHE[key]


def clear_model_cache() -> None:
    _CACHE.clear()


def _check_tcn_regions(models_dir, regions_path) -> None:
    """The check run_predict makes: the regions file must be the one the TCN used."""
    from .clip_data import REGIONS_PATH, Regions

    trained = tcn_predict.trained_regions_version(models_dir)
    current = Regions.load(regions_path or REGIONS_PATH).version
    if trained is not None and trained != current:
        raise ValueError(
            f"the TCN was trained with map regions {trained}, but the regions file is "
            f"{current}; use the regions file the models were trained with"
        )


# --------------------------------------------------------------------------- background


def training_match_clips(dataset, match: str) -> List[Path]:
    """Clip paths of one training match, from <dataset>/train/manifest.csv only (the test
    manifest is never opened). ``match`` is a match folder name or a unique part of one."""
    manifest = Path(dataset) / "train" / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(r.get("split", "train") != "train" for r in rows):
        raise ValueError(f"{manifest} lists a non-training clip")
    matches = sorted({r["match_folder"] for r in rows})
    chosen = [m for m in matches if m == match] or [m for m in matches if match in m]
    if len(chosen) != 1:
        raise ValueError(
            f"{match!r} names {len(chosen)} training matches; choose one of: {', '.join(matches)}"
        )
    return [Path(dataset) / "train" / r["clip"] for r in rows if r["match_folder"] == chosen[0]]


def save_background(dataset, match: str, output, *, log=print) -> np.ndarray:
    """Build the empty-map background from one training match's clips, as the training
    detections were built (median of 8 frames per clip), and save it as a lossless PNG."""
    import cv2

    from .tactic_features import match_background

    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("save the background as a .png file (lossless)")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new file name")
    clips = training_match_clips(dataset, match)
    log(f"Background from {len(clips)} training clips of {clips[0].name.split('.', 1)[0]}")
    background = match_background(clips, per_clip=BACKGROUND_CLIPS_PER_FRAME)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode the background as PNG")
    with output.open("xb") as stream:
        stream.write(encoded.tobytes())
    log(f"Saved {output} ({background.shape[1]}x{background.shape[0]})")
    return background


def load_background(background) -> np.ndarray:
    """A saved background image (path) or an RGB uint8 array (height, width, 3)."""
    if isinstance(background, np.ndarray):
        image = background
    else:
        import cv2

        path = Path(background)
        raw = np.fromfile(path, np.uint8) if path.is_file() else np.zeros(0, np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
        if bgr is None:
            raise ValueError(f"cannot read the background image {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("the background must be an RGB uint8 image of shape (height, width, 3)")
    return image


# --------------------------------------------------------------------------- features


def _feature_spec(step: int):
    """What tcn_predict.clip_inputs reads from its model list: the row step and whether any
    model uses ResNet numbers (both final models do)."""
    return [(None, SimpleNamespace(compressor=True), SimpleNamespace(step=step))]


def detect(paths, *, background=None, background_from=None, log=print) -> tuple:
    """Detections per clip name, and warnings per clip name. Without a background this is
    tcn_predict.detect_clips (one background per match, from that match's clips)."""
    from . import tactic_features
    from .clip_data import _crop_size

    warnings = {p.stem: [] for p in paths}
    if background is not None:
        image = load_background(background)
        dets = {}
        for i, path in enumerate(paths, 1):
            size = _crop_size(path, (0, 0))
            if size != (image.shape[1], image.shape[0]):
                raise ValueError(
                    f"{path.name} is {size[0]}x{size[1]} but the background is "
                    f"{image.shape[1]}x{image.shape[0]}; use a background of the same layout"
                )
            log(f"  detecting {i}/{len(paths)}: {path.name} (saved background)")
            dets[path.stem] = tactic_features.detect_clip(path, image)
    else:
        dets = tcn_predict.detect_clips(paths, background_from, log=log)
        if background_from:
            sources = len(tcn_predict.find_clips(background_from))
            groups = {p.stem: sources for p in paths}
        else:
            counts: Dict[str, int] = {}
            for p in paths:
                counts[tcn_predict.match_of(p)] = counts.get(tcn_predict.match_of(p), 0) + 1
            groups = {p.stem: counts[tcn_predict.match_of(p)] for p in paths}
        for name, n in groups.items():
            if n < tcn_predict.FEW_BACKGROUND_CLIPS:
                warnings[name].append(
                    f"background from only {n} video(s); still players may be mistaken for the "
                    "map. Pass a saved background (background=) or more clips of the match."
                )
    for name, det in dets.items():
        fps = float(det["fps"])
        if abs(fps - tcn_predict.TRAINED_FPS) > 0.5:
            warnings[name].append(f"clip is {fps:.1f} fps; the models were trained on 10 fps")
    return dets, warnings


def clip_features(paths, dets, *, step: int, regions_path=None, device="cpu", log=print):
    """Clip name -> {"detector", "resnet", "t_s"}: tcn_predict.clip_inputs's rows (built as
    build-clip-data builds them) plus each row's time in seconds, as build-clip-data saves."""
    rows = tcn_predict.clip_inputs(paths, dets, _feature_spec(step), regions_path, device, log)
    for path in paths:
        det = dets[path.stem]
        frames = np.arange(0, len(det["bomb"]), step)
        rows[path.stem]["t_s"] = (frames / float(det["fps"])).astype(np.float32)
        if len(rows[path.stem]["detector"]) != len(frames):
            raise ValueError(f"{path.name}: detector rows and frame times differ in number")
    return rows


def gru_rows(features: Dict):
    """One clip's features -> the GRU's input, (1, rows, detector + 512 + 3): the same raw
    rows the TCN reads (detector channels, then raw ResNet numbers) with elapsed time,
    packed exactly as tactic_temporal.load_clip_rows and sequence_inputs pack training clips."""
    import torch

    from .tactic_temporal import sequence_inputs

    detector = np.asarray(features["detector"], np.float32)
    resnet = np.asarray(features["resnet"], np.float32)
    if len(detector) != len(resnet) or len(detector) != len(features["t_s"]):
        raise ValueError("detector, ResNet and time rows differ in number")
    rows = torch.as_tensor(np.concatenate([detector, resnet], 1))[None]
    times = torch.as_tensor(np.asarray(features["t_s"], np.float32), dtype=torch.float64)[None]
    padding = torch.zeros(1, len(detector), dtype=torch.bool)
    return sequence_inputs(rows, times, padding)


def gru_probs(members, features: Sequence[Dict], device: str = "cpu") -> np.ndarray:
    """Mean class probabilities over the GRU seeds, one clip at a time, (clips, classes)."""
    import torch

    out = []
    with torch.no_grad():
        for head, _ in members:
            head.eval()
            out.append(
                np.concatenate(
                    [
                        head(gru_rows(f).to(device)).softmax(1).cpu().numpy().astype(np.float64)
                        for f in features
                    ]
                )
            )
    return np.mean(out, 0)


def tcn_probs(members, features: Sequence[Dict], device: str = "cpu") -> np.ndarray:
    """tcn_predict's seed ensemble, on the TCN's own keys only (never the GRU's)."""
    return tcn_predict.ensemble_probs(
        members, [{"detector": f["detector"], "resnet": f["resnet"]} for f in features], device
    )


# --------------------------------------------------------------------------- prediction


def _models(models) -> tuple:
    models = (models,) if isinstance(models, str) else tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(MODEL_KINDS):
        raise ValueError(f"models must be distinct names from {', '.join(MODEL_KINDS)}")
    return models


def _paths(paths) -> List[Path]:
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return tcn_predict.find_clips(paths)


def predict_probabilities(
    paths,
    models=("tcn",),
    *,
    background=None,
    background_from=None,
    tcn_models=None,
    gru_models=None,
    regions_path=None,
    device: str = "cpu",
    log=print,
    features: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """The shared core of predict_clips and predict-gru: {names, files, groups, warnings,
    probs: {model: (clips, classes)}, seconds}. ``features`` (clip name -> clip_features
    output) skips video reading, for tests."""
    models = _models(models)
    if background is not None and background_from:
        raise ValueError("pass either a saved background or background_from, not both")
    loaded = {
        kind: load_models(
            kind,
            tcn_models if kind == "tcn" else gru_models,
            device=device,
            regions_path=regions_path,
        )
        for kind in models
    }
    steps = {
        kind: (members[0][2].step if kind == "tcn" else members[0][1]["settings"]["step"])
        for kind, members in loaded.items()
    }
    if len(set(steps.values())) != 1:
        raise ValueError(f"the models read rows at different steps: {steps}")
    step = next(iter(steps.values()))
    seconds = {}
    if features is None:
        clips = _paths(paths)
        started = time.perf_counter()
        dets, warnings = detect(
            clips, background=background, background_from=background_from, log=log
        )
        seconds["detection"] = time.perf_counter() - started
        started = time.perf_counter()
        features = clip_features(clips, dets, step=step, regions_path=regions_path, device=device)
        seconds["clip_features"] = time.perf_counter() - started
        names = [p.stem for p in clips]
        files = [str(p) for p in clips]
        own_background = background is None and not background_from
        groups = [tcn_predict.match_of(p) if own_background else "" for p in clips]
    else:
        names = list(features)
        files = [None] * len(names)
        groups = [n.split(".", 1)[0] for n in names]
        warnings = {n: [] for n in names}
    probs = {}
    for kind in models:  # The TCN always runs first and on its own copy of the inputs.
        started = time.perf_counter()
        run = tcn_probs if kind == "tcn" else gru_probs
        probs[kind] = run(loaded[kind], [features[n] for n in names], device)
        seconds[kind] = time.perf_counter() - started
    return {
        "names": names,
        "files": files,
        "groups": groups,
        "warnings": warnings,
        "probs": probs,
        "seconds": seconds,
    }


def predict_clips(
    paths,
    models=("tcn",),
    background=None,
    background_from=None,
    min_confidence: float = 0.6,
    *,
    tcn_models=None,
    gru_models=None,
    regions_path=None,
    device: str = "cpu",
    log=print,
    features: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """Classify prepared clips (radar crop, 10 fps, one tactic each) and return JSON-ready
    results, one dict per clip:

        {"clip": name, "file": path, "warnings": [...],
         "predictions": {"tcn": {"tactic", "confidence", "uncertain", "runner_up",
                                 "probabilities": {class: p}}, "gru": {...}}}

    ``models``: ("tcn",) by default (the selected model); ("gru",) or ("tcn", "gru") for the
    Compact pooled GRU as well. ``background``: a saved empty-map image (path or RGB array)
    of the same broadcast layout, needed for a single clip; ``background_from``: videos of
    the same match to build it from instead; with neither, each match's clips build their
    own (as predict-tcn does). A prediction whose top probability is below
    ``min_confidence`` is flagged as uncertain.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    result = predict_probabilities(
        paths,
        models,
        background=background,
        background_from=background_from,
        tcn_models=tcn_models,
        gru_models=gru_models,
        regions_path=regions_path,
        device=device,
        log=log,
        features=features,
    )
    rows = {
        kind: tcn_predict.prediction_rows(result["names"], result["groups"], p, min_confidence)
        for kind, p in result["probs"].items()
    }
    out = []
    for i, (name, file) in enumerate(zip(result["names"], result["files"])):
        predictions = {}
        for kind, kind_rows in rows.items():
            r = kind_rows[i]
            predictions[kind] = {
                "tactic": r["predicted"],
                "confidence": r["confidence"],
                "uncertain": r["uncertain"],
                "runner_up": r["runner_up"],
                "probabilities": {c: r[f"p_{c}"] for c in TACTIC_CLASSES},
            }
        out.append(
            {
                "clip": name,
                "file": file,
                "warnings": list(result["warnings"][name]),
                "predictions": predictions,
            }
        )
    return out


def run_predict_gru(
    clips,
    models_dir=None,
    *,
    with_tcn: bool = False,
    tcn_models=None,
    output: Optional[Path] = None,
    background=None,
    background_from=None,
    min_confidence: float = 0.6,
    regions_path=None,
    device: str = "cpu",
    log=print,
) -> List[Dict]:
    """predict-gru: predict-tcn's table and CSV for the GRU (and the TCN with with_tcn),
    with a model column first."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"{output} already exists; choose a new file name")
    models = ("tcn", "gru") if with_tcn else ("gru",)
    result = predict_probabilities(
        clips,
        models,
        background=background,
        background_from=background_from,
        tcn_models=tcn_models,
        gru_models=models_dir,
        regions_path=regions_path,
        device=device,
        log=log,
    )
    rows = []
    for kind in models:
        kind_rows = tcn_predict.prediction_rows(
            result["names"], result["groups"], result["probs"][kind], min_confidence
        )
        log(f"\n{DISPLAY_NAMES[kind]} ({kind}):")
        log(tcn_predict.format_rows(kind_rows))
        rows += [{"model": kind, **r} for r in kind_rows]
    for name, notes in result["warnings"].items():
        for note in notes:
            log(f"  warning, {name}: {note}")
    if output is not None:
        tcn_predict._write_csv(Path(output), rows)
        log(f"Predictions: {output}")
    return rows
