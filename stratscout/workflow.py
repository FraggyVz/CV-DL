"""Workflow: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import set_seed
from .contracts import check_contract
from .datasets import IconDataset, load_icon_labels, synthetic_icon_dataset
from .detector import MinimapDetector
from .evaluation import detector_metrics
from .pilot import _pilot_sources_match, extract_pilot_frames
from .preprocessing import MinimapPreprocessor
from .segmentation import (
    DEFAULT_TEAM_HUES,
    find_icons_teams,
    learn_background,
    learn_team_hues,
)
from .storage import save_checkpoint, write_new_text
from .synthetic import (
    MAP_POINTS_FILE,
    _nearest_walkable,
    build_background,
    generate_split,
    make_round,
    render_frames,
    walkable_mask,
)
from .tracking import positional_error, positions_from_video
from .training import train_detector
from .video import detect_crops, find_radar_box, list_videos, sample_probe_frames
from .visualisation import (
    show_counts,
    show_crop_on_screen,
    show_detections,
    show_detector_output,
    show_mask_pipeline,
    show_pipeline,
    show_training_curve,
    verify_crop,
)


def discover_videos(context):
    """Explicit workflow step migrated from original notebook cell 2."""
    print("device:", context.device, "| torch", torch.__version__)
    context.state["N_PLAYERS"] = context.config.n_per_team * 2
    set_seed(context.config.seed)
    context.state["VIDEOS"] = list_videos(context=context)
    if context.config.video_path is None and context.state["VIDEOS"]:
        context.config.video_path = str(context.state["VIDEOS"][0])
    context.state["HAVE_VIDEO"] = (
        cv2 is not None
        and context.config.video_path is not None
        and Path(context.config.video_path).exists()
    )
    print(f"data dir: {context.config.data_dir}/  ->  {len(context.state['VIDEOS'])} video(s)")
    for v in context.state["VIDEOS"]:
        print(f"   {v.name:<40} {v.stat().st_size / 1000000000.0:6.2f} GB")
    if not context.state["VIDEOS"]:
        print(
            f"   (empty — drop rendered demo replays into {Path(context.config.data_dir).resolve()})"
        )
    print(
        "primary video:",
        context.config.video_path,
        "->",
        "FOUND (real mode)" if context.state["HAVE_VIDEO"] else "missing (synthetic mode)",
    )


def show_contracts(context):
    """Explicit workflow step migrated from original notebook cell 4."""
    print("contracts defined")
    show_pipeline()


def locate_crops(context):
    """Explicit workflow step migrated from original notebook cell 7."""
    if context.state["HAVE_VIDEO"]:
        context.state["CROPS"] = detect_crops(context.state["VIDEOS"], context=context)
        dropped = [v.name for v in context.state["VIDEOS"] if v.name not in context.state["CROPS"]]
        context.state["VIDEOS"] = [
            v for v in context.state["VIDEOS"] if v.name in context.state["CROPS"]
        ]
        if dropped:
            print("left out (no radar found):", dropped)
        if Path(context.config.video_path).name not in context.state["CROPS"]:
            context.config.video_path = (
                str(context.state["VIDEOS"][0]) if context.state["VIDEOS"] else None
            )
        context.state["HAVE_VIDEO"] = bool(context.state["VIDEOS"])
    if context.state["HAVE_VIDEO"]:
        context.state["probe"] = sample_probe_frames(
            context.config.video_path, context.config.probe_frames
        )
        context.state["CROP"] = (
            context.state["CROPS"].get(Path(context.config.video_path).name)
            or context.config.crop_box
            or find_radar_box(context.state["probe"], context.config.search_frac)
        )
        print(
            f"\nprimary screen {context.state['probe'].shape[2]}x{context.state['probe'].shape[1]}  ->  CROP = {context.state['CROP']}"
        )
    else:
        context.state["CROPS"] = {}
        context.state["CROP"] = (
            0,
            0,
            context.config.minimap_px,
            context.config.minimap_px,
        )
        print("synthetic mode — crop box not needed")


def verify_crops(context):
    """Explicit workflow step migrated from original notebook cell 9."""
    VERIFY_ALL = False
    if context.state["HAVE_VIDEO"]:
        show_crop_on_screen(
            context.state["probe"][len(context.state["probe"]) // 2],
            context.state["CROP"],
            Path(context.config.video_path).stem,
        )
        verify_crop(context.state["probe"], context.state["CROP"])
        if VERIFY_ALL:
            for v in context.state["VIDEOS"]:
                if str(v) != str(Path(context.config.video_path)):
                    print(f"--- {v.name}")
                    verify_crop(
                        sample_probe_frames(str(v), context.config.probe_frames),
                        context.state["CROPS"][v.name],
                    )


def prepare_pilot(context):
    """Explicit workflow step migrated from original notebook cell 12."""
    if not context.state["HAVE_VIDEO"]:
        context.state["pilot_frames"] = None
        print("synthetic mode — no frames extracted")
    elif (
        Path(context.config.icon_labels_json).exists() and Path(context.config.frames_npy).exists()
    ):
        context.state["pilot_frames"] = np.load(context.config.frames_npy)
        print(
            f"{context.config.icon_labels_json} exists — reusing {context.config.frames_npy} {context.state['pilot_frames'].shape}. New videos are NOT added to the icon set; delete both files to rebuild it."
        )
    elif Path(context.config.frames_npy).exists() and _pilot_sources_match(context=context):
        context.state["pilot_frames"] = np.load(context.config.frames_npy)
        print(
            f"reusing {context.config.frames_npy} {context.state['pilot_frames'].shape} — same videos as last time"
        )
    else:
        if Path(context.config.frames_npy).exists():
            bak = Path(context.config.frames_npy).with_suffix(
                f".bak-{datetime.now():%Y%m%d-%H%M%S}.npy"
            )
            Path(context.config.frames_npy).rename(bak)
            print(f"videos changed — old frames kept as {bak.name}")
        context.state["pilot_frames"] = extract_pilot_frames(context=context)


def prepare_background(context):
    """Explicit workflow step migrated from original notebook cell 15."""
    if context.state["HAVE_VIDEO"]:
        context.state["BG"] = learn_background(context.state["pilot_frames"])
        context.state["HUES"] = learn_team_hues(context.state["pilot_frames"], context=context)
        _d = find_icons_teams(
            context.state["pilot_frames"][0],
            context.state["HUES"],
            context.state["BG"],
            context=context,
        )
        print(f"frame 0: {sum((p[2] == 0 for p in _d))} T, {sum((p[2] == 1 for p in _d))} CT")
    else:
        context.state["BG"], context.state["HUES"] = (None, DEFAULT_TEAM_HUES)
        print("synthetic mode — icon finder will be exercised in Section 7")
    context.state["MAP_BG_SOURCE"] = (
        context.state["BG"]
        if context.state["BG"] is not None
        else context.config.map_image
        if Path(context.config.map_image).exists()
        else None
    )
    print(
        "synthetic background:",
        "median of your footage"
        if isinstance(context.state["MAP_BG_SOURCE"], np.ndarray)
        else f"map image {context.config.map_image}"
        if context.state["MAP_BG_SOURCE"]
        else "drawn abstraction (no real map available)",
    )
    print(
        "plot background:     ",
        f"{context.config.map_image}"
        if Path(context.config.map_image).exists()
        else "same as synthetic background",
    )


def preview_segmentation(context):
    """Explicit workflow step migrated from original notebook cell 18."""
    if context.state["HAVE_VIDEO"]:
        show_mask_pipeline(
            context.state["pilot_frames"],
            0,
            context.state["HUES"],
            context.state["BG"],
            context=context,
        )
        show_detections(
            context.state["pilot_frames"],
            team_hues=context.state["HUES"],
            background=context.state["BG"],
            context=context,
        )
        show_counts(
            context.state["pilot_frames"],
            context.state["HUES"],
            context.state["BG"],
            step=4,
            context=context,
        )


def prepare_map(context, *, preview=True):
    """Explicit workflow step migrated from original notebook cell 20."""
    context.state["MAP_POINTS"] = {
        "t_spawn": (0.88, 0.36),
        "ct_spawn": (0.297, 0.697),
        "a_site": (0.545, 0.77),
        "b_site": (0.23, 0.276),
        "mid": (0.52, 0.47),
        "a_ramp": (0.8, 0.63),
        "a_approach": (0.66, 0.745),
        "connector": (0.62, 0.58),
        "b_apps": (0.52, 0.215),
        "b_short": (0.38, 0.33),
    }
    if Path(MAP_POINTS_FILE).exists():
        context.state["MAP_POINTS"] = {
            k: tuple(v) for k, v in json.loads(Path(MAP_POINTS_FILE).read_text()).items()
        }
        print(f"loaded callouts from {MAP_POINTS_FILE}")
    context.state["A_SITE"] = np.array(context.state["MAP_POINTS"]["a_site"], np.float32)
    context.state["B_SITE"] = np.array(context.state["MAP_POINTS"]["b_site"], np.float32)
    context.state["_GRID_CACHE"]: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    context.state["MAP_MASK"] = walkable_mask(
        build_background(
            context.config.minimap_px,
            np.random.default_rng(1234),
            context.state["MAP_BG_SOURCE"],
            context=context,
        )
    )
    context.state["_NEAREST"] = _nearest_walkable(context.state["MAP_MASK"])
    print(f"walkable map: {context.state['MAP_MASK'].mean() * 100:.0f}% of the radar crop")
    if not preview:
        return
    _rng = np.random.default_rng(context.config.seed)
    _bg = build_background(
        context.config.minimap_px,
        np.random.default_rng(1234),
        context.state["MAP_BG_SOURCE"],
        context=context,
    )
    _demo = render_frames(
        make_round("A_SPLIT", _rng, context.config.seq_len, context=context),
        _bg,
        context.config.minimap_px,
        _rng,
        context=context,
    )
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, f in zip(axes, np.linspace(0, len(_demo) - 1, 4).astype(int)):
        ax.imshow(_demo[f])
        ax.set_title(f"t={f}")
        ax.axis("off")
    fig.suptitle("Synthetic radar — A_SPLIT (two attacker routes converging on A)")
    plt.tight_layout()
    plt.show()
    show_detections(
        _demo,
        idxs=np.linspace(0, len(_demo) - 1, 5).astype(int),
        title="icon finder on synthetic frames",
        context=context,
    )


def prepare_synthetic_rounds(context):
    """Rendered synthetic rounds: frames and true positions for training and checking the
    detector and tracker. The tactic labels they carry are not used by any model here."""
    context.state["test_rounds"] = generate_split(
        context.config.test_matches,
        "test",
        context.config.seed + 3,
        render=True,
        rounds_per_match=context.config.rounds_per_match,
        context=context,
    )
    context.state["det_pool"] = generate_split(
        context.config.detector_matches,
        "detpool",
        context.config.seed + 4,
        render=True,
        rounds_per_match=context.config.rounds_per_match,
        context=context,
    )
    context.state["det_pool_val"] = generate_split(
        1,
        "detval",
        context.config.seed + 5,
        render=True,
        rounds_per_match=context.config.rounds_per_match,
        context=context,
    )
    ids = {
        n: {r.match_id for r in context.state[k]}
        for n, k in [
            ("detector", "det_pool"),
            ("detector val", "det_pool_val"),
            ("test", "test_rounds"),
        ]
    }
    assert not (
        ids["detector"] & ids["detector val"]
        or ids["detector"] & ids["test"]
        or ids["detector val"] & ids["test"]
    ), "match-id leakage between splits"
    print(
        f"detector {len(context.state['det_pool'])} | detector val {len(context.state['det_pool_val'])} | test {len(context.state['test_rounds'])} rounds - no match-id overlap"
    )


def prepare_preprocessor(context):
    """Explicit workflow step migrated from original notebook cell 24."""
    context.state["PRE"] = MinimapPreprocessor(context.config.minimap_px)
    check_contract(
        "frames",
        context.state["PRE"](context.state["test_rounds"][0].frames),
        (
            context.config.seq_len,
            3,
            context.config.minimap_px,
            context.config.minimap_px,
        ),
    )


def describe_detector(context):
    """Explicit workflow step migrated from original notebook cell 26."""
    print(
        f"detector parameters: {sum((p.numel() for p in MinimapDetector(context=context).parameters())):,}"
    )


def prepare_detector_data(context):
    """Explicit workflow step migrated from original notebook cell 28."""
    context.state["USING_REAL_ICONS"] = (
        Path(context.config.icon_labels_json).exists() and Path(context.config.frames_npy).exists()
    )
    if context.state["USING_REAL_ICONS"]:
        f_all, p_all = load_icon_labels(
            np.load(context.config.frames_npy), context.config.icon_labels_json
        )
        cut = int(len(f_all) * 0.8)
        context.state["det_train"] = IconDataset(f_all[:cut], p_all[:cut], context=context)
        det_val = IconDataset(f_all[cut:], p_all[cut:], context=context)
        print(
            f"REAL icon labels: {len(f_all)} frames -> train {len(context.state['det_train'])} / val {len(det_val)}"
        )
    else:
        context.state["det_train"] = synthetic_icon_dataset(
            context.state["det_pool"], context.config.detector_train_frames, 10, context=context
        )
        det_val = synthetic_icon_dataset(
            context.state["det_pool_val"], context.config.detector_val_frames, 11, context=context
        )
        print(f"SYNTHETIC icons: train {len(context.state['det_train'])} / val {len(det_val)}")
    context.state["det_train_dl"] = DataLoader(
        context.state["det_train"],
        batch_size=context.config.det_batch,
        shuffle=True,
        drop_last=True,
    )
    context.state["det_val_dl"] = DataLoader(det_val, batch_size=context.config.det_batch)


def prepare_detection_mask(context):
    """Explicit workflow step migrated from original notebook cell 30."""
    context.state["DETECT_MASK"] = cv2.dilate(
        context.state["MAP_MASK"].astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)


def fit_detector(context, *, preview=True):
    """Explicit workflow step migrated from original notebook cell 32."""
    output = Path(context.config.out_dir) / "detector.pt"
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    context.state["DET_HISTORY"]: List[Dict[str, float]] = []
    context.state["detector"] = train_detector(context=context)
    if preview:
        show_training_curve(context.state["DET_HISTORY"], context=context)
    context.state["DET_METRICS"] = detector_metrics(
        context.state["detector"], context.state["det_val_dl"], context=context
    )
    print(
        "\nfinal detector:",
        {k: round(v, 4) for k, v in context.state["DET_METRICS"].items()},
    )
    save_checkpoint(
        context.state["detector"].state_dict(),
        Path(context.config.out_dir) / "detector.pt",
    )


def preview_detector(context):
    """Explicit workflow step migrated from original notebook cell 34."""
    _vis_frames = (
        np.load(context.config.frames_npy)
        if context.state["USING_REAL_ICONS"]
        else context.state["test_rounds"][0].frames
    )
    show_detector_output(context.state["detector"], _vis_frames, context=context)


def check_tracking(context):
    """Explicit workflow step migrated from original notebook cell 36."""
    _est = positions_from_video(
        context.state["detector"],
        context.state["test_rounds"][0].frames,
        context=context,
    )
    check_contract("positions", _est, (context.config.seq_len, context.state["N_PLAYERS"], 2))
    context.state["errs"] = [
        positional_error(
            positions_from_video(context.state["detector"], r.frames, context=context),
            r.positions,
            context=context,
        )
        for r in context.state["test_rounds"][:20]
    ]
    print(
        f"detector alone : {context.state['DET_METRICS']['mean_pos_err_norm'] * context.config.minimap_px:5.2f} px"
    )
    print(f"+ tracker      : {np.mean(context.state['errs']) * context.config.minimap_px:5.2f} px")
    print(
        "These errors describe this run only; synthetic results do not establish real-footage accuracy."
    )


def describe_icon_review(context):
    """Explicit workflow step migrated from original notebook cell 38."""
    print("to label real footage:")
    print("Import auto_label_icons and ReviewTool from stratscout.annotations.")
    print(
        "Pass context=context and context.state['pilot_frames']; see docs/archive/architecture.md."
    )
    print(
        "Keep proposal output separate from reviewed labels: legacy proposals overwrite their path."
    )


def describe_round_review(context):
    """Round labels (start, plant, end in video seconds) are kept for the future time-aware
    tactic model; nothing in this package turns them into a training set yet."""
    print("Import RoundLabeller from stratscout.annotations; pass context=context.")
    print(
        "                   -> saves",
        Path(context.config.labels_dir) / "<match_id>.rounds.json",
    )


def save_results(context):
    """Write the vision summary next to detector.pt in a fresh output directory."""
    out = Path(context.config.out_dir)
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    summary = {
        "mode": "mixed" if context.state["USING_REAL_ICONS"] else "synthetic",
        "detector_data": "real" if context.state["USING_REAL_ICONS"] else "synthetic",
        "crop_box": list(context.state["CROP"]),
        "detector": {k: float(v) for k, v in context.state["DET_METRICS"].items()},
        "pipeline_pos_err_px": float(np.mean(context.state["errs"]) * context.config.minimap_px),
        "config": {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(context.config).items()
        },
    }
    write_new_text(out / "summary.json", json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


def run_prototype(context, *, plots=False):
    """Run the vision experiment explicitly: crop, colour baseline, detector and tracker.

    The CLI reserves a fresh output directory. Individual notebook stages are also
    available above; their inputs and outputs live in this run's context.
    """
    steps = [discover_videos, locate_crops, prepare_pilot, prepare_background]
    for step in steps:
        step(context)
    prepare_map(context, preview=plots)
    for step in (
        prepare_synthetic_rounds,
        prepare_preprocessor,
        prepare_detector_data,
        prepare_detection_mask,
        fit_detector,
        check_tracking,
    ):
        if step is fit_detector:
            step(context, preview=plots)
        else:
            step(context)
        if not plots:
            plt.close("all")
    if plots:
        for step in (
            show_contracts,
            verify_crops,
            preview_segmentation,
            describe_detector,
            preview_detector,
        ):
            step(context)
    save_results(context)
    return context
