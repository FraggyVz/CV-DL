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

from .annotations import load_real_rounds
from .config import CT_CLASSES, PHASES, T_CLASSES, set_seed
from .contracts import check_contract
from .datasets import (
    IconDataset,
    SequenceDataset,
    load_icon_labels,
    synthetic_icon_dataset,
)
from .detector import MinimapDetector
from .evaluation import detector_metrics, evaluate
from .features import ct_post_signals, to_features, to_sequence
from .models import MODEL_ZOO
from .pilot import _pilot_sources_match, extract_pilot_frames
from .pipeline import StratScoutPipeline
from .preprocessing import MinimapPreprocessor
from .reports import (
    plot_routes,
    plot_win_rates,
    rounds_needed,
    scouting_report,
    show_side_occupancy,
    show_tactic_occupancy,
    show_trails,
    win_rate_report,
)
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
from .training import train_classifier, train_detector
from .video import detect_crops, find_radar_box, list_videos, sample_probe_frames
from .visualisation import (
    plot_confusion,
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
    print("attacker tactics (before the plant):", T_CLASSES)
    print("defender responses (after the plant):", CT_CLASSES)


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
    """Explicit workflow step migrated from original notebook cell 22."""
    context.state["train_rounds"] = generate_split(
        context.config.train_matches,
        "train",
        context.config.seed + 1,
        render=False,
        rounds_per_match=context.config.rounds_per_match,
        context=context,
    )
    context.state["val_rounds"] = generate_split(
        context.config.val_matches,
        "val",
        context.config.seed + 2,
        render=False,
        rounds_per_match=context.config.rounds_per_match,
        context=context,
    )
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
        n: {r.match_id for r in rs}
        for n, rs in [
            ("train", context.state["train_rounds"]),
            ("val", context.state["val_rounds"]),
            ("test", context.state["test_rounds"]),
        ]
    }
    assert (
        not ids["train"] & ids["val"]
        and (not ids["train"] & ids["test"])
        and (not ids["val"] & ids["test"])
    ), "match-id leakage between splits"
    print(
        f"train {len(context.state['train_rounds'])} | val {len(context.state['val_rounds'])} | test {len(context.state['test_rounds'])} rounds - no match-id overlap"
    )
    print(
        "attacker tactics (train):",
        {c: sum((r.label == c for r in context.state["train_rounds"])) for c in T_CLASSES},
    )
    print(
        f"bomb planted in {sum((r.planted for r in context.state['train_rounds']))}/{len(context.state['train_rounds'])} train rounds; defender responses:",
        {c: sum((r.ct_label == c for r in context.state["train_rounds"])) for c in CT_CLASSES},
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
    print("Pass context=context and context.state['pilot_frames']; see docs/architecture.md.")
    print(
        "Keep proposal output separate from reviewed labels: legacy proposals overwrite their path."
    )


def check_features(context):
    """Explicit workflow step migrated from original notebook cell 40."""
    context.state["FEATURE_NAMES"] = [
        f"{t}_{n}"
        for t in context.config.teams
        for n in ["cx", "cy", "spread", "dist_A", "dist_B", "near_A", "near_B", "speed"]
    ]
    check_contract(
        "sequence",
        to_sequence(context.state["train_rounds"][0].positions, context=context),
        (context.config.seq_len, 16),
    )


def preview_features(context):
    """Explicit workflow step migrated from original notebook cell 42."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, (fi, name) in zip(
        axes, [(3, "T distance to A site"), (2, "T spread"), (7, "T mean speed")]
    ):
        for c in T_CLASSES:
            curves = np.stack(
                [
                    to_features(r.positions, context=context)[:, fi]
                    for r in context.state["train_rounds"]
                    if r.label == c
                ]
            )
            ax.plot(curves.mean(0), lw=2, label=c)
        ax.set_title(name)
        ax.set_xlabel("timestep (before the plant)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Question 1 - attacker tactics, before the plant")
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    titles = [
        "CT mean distance to the planted site",
        "CTs standing on the site",
        "closest CT to the site",
    ]
    for c in CT_CLASSES:
        sig = [
            ct_post_signals(r, context=context)
            for r in context.state["train_rounds"]
            if r.ct_label == c
        ]
        for k, ax in enumerate(axes):
            ax.plot(np.stack([s[k] for s in sig]).mean(0), lw=2, label=c)
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("timestep (after the plant)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Question 2 - defender responses, after the plant")
    plt.tight_layout()
    plt.show()


def describe_classifiers(context):
    """Explicit workflow step migrated from original notebook cell 44."""
    for k, v in MODEL_ZOO.items():
        n_t = sum(
            (
                p.numel()
                for p in v(
                    16, context.config.cls_hidden, len(T_CLASSES), context=context
                ).parameters()
            )
        )
        n_ct = sum(
            (
                p.numel()
                for p in v(
                    16, context.config.cls_hidden, len(CT_CLASSES), context=context
                ).parameters()
            )
        )
        print(f"{k:<13} {n_t:>8,} params (attacker head)   {n_ct:>8,} (defender head)")


def fit_classifiers(context):
    """Explicit workflow step migrated from original notebook cell 47."""
    context.state["REPR"] = "features"
    context.state["DATA"]: Dict[str, Dict[str, SequenceDataset]] = {}
    for phase in ("pre", "post"):
        tr = SequenceDataset(
            context.state["train_rounds"],
            context.state["REPR"],
            "gt",
            phase=phase,
            context=context,
        )
        context.state["DATA"][phase] = {
            "train": tr,
            "val": SequenceDataset(
                context.state["val_rounds"],
                context.state["REPR"],
                "gt",
                normaliser=tr.normaliser,
                phase=phase,
                context=context,
            ),
            "test": SequenceDataset(
                context.state["test_rounds"],
                context.state["REPR"],
                "gt",
                normaliser=tr.normaliser,
                phase=phase,
                context=context,
            ),
        }
    context.state["train_ds"], val_ds, context.state["test_ds"] = (
        context.state["DATA"]["pre"][k] for k in ("train", "val", "test")
    )
    context.state["RESULTS"]: Dict[str, Dict] = {}
    for phase, title in (
        ("pre", "QUESTION 1 - attacker tactic, before the plant"),
        ("post", "QUESTION 2 - defender response, after the plant"),
    ):
        d = context.state["DATA"][phase]
        print(
            f"\n{title}   (train {len(d['train'])} / val {len(d['val'])} / test {len(d['test'])} rounds)"
        )
        context.state["RESULTS"][phase] = {}
        for name in MODEL_ZOO:
            model, vm = train_classifier(name, d["train"], d["val"], verbose=False, context=context)
            tm = evaluate(model, d["test"], context=context)
            context.state["RESULTS"][phase][name] = {
                "model": model,
                "val": vm,
                "test": tm,
            }
            print(
                f"  {name:<12} val macro-F1 {vm['macro_f1']:.3f}   TEST acc {tm['accuracy']:.3f}  macro-F1 {tm['macro_f1']:.3f}"
            )
    context.state["BEST"] = {
        ph: max(
            context.state["RESULTS"][ph],
            key=lambda k: context.state["RESULTS"][ph][k]["val"]["macro_f1"],
        )
        for ph in context.state["RESULTS"]
    }
    context.state["results"], context.state["best_name"] = (
        context.state["RESULTS"]["pre"],
        context.state["BEST"]["pre"],
    )
    context.state["best_model"] = context.state["results"][context.state["best_name"]]["model"]
    context.state["ct_best_name"] = context.state["BEST"]["post"]
    context.state["ct_model"] = context.state["RESULTS"]["post"][context.state["ct_best_name"]][
        "model"
    ]
    print(
        f"\nbest on validation: attacker question -> {context.state['best_name']} | defender question -> {context.state['ct_best_name']}"
    )


def preview_classifiers(context):
    """Explicit workflow step migrated from original notebook cell 48."""
    plot_confusion(
        context.state["results"][context.state["best_name"]]["test"]["confusion"],
        f"Q1 attacker tactic - {context.state['best_name']} - oracle positions (test)",
    )
    for c, f1 in zip(
        T_CLASSES,
        context.state["results"][context.state["best_name"]]["test"]["per_class_f1"],
    ):
        print(f"  {c:<9} F1 {f1:.3f}")
    ct_test = context.state["RESULTS"]["post"][context.state["ct_best_name"]]["test"]
    plot_confusion(
        ct_test["confusion"],
        f"Q2 defender response - {context.state['ct_best_name']} - oracle positions (test)",
        CT_CLASSES,
    )
    for c, f1 in zip(CT_CLASSES, ct_test["per_class_f1"]):
        print(f"  {c:<9} F1 {f1:.3f}")


def compare_oracle_and_vision(context, *, run_bprime=False, preview=True):
    """Explicit workflow step migrated from original notebook cell 50."""
    vision_test_ds = SequenceDataset(
        context.state["test_rounds"],
        context.state["REPR"],
        "vision",
        detector=context.state["detector"],
        normaliser=context.state["train_ds"].normaliser,
        context=context,
    )
    context.state["exp_a"], context.state["exp_b"] = (
        evaluate(context.state["best_model"], context.state["test_ds"], context=context),
        evaluate(context.state["best_model"], vision_test_ds, context=context),
    )
    ct_vision_test = SequenceDataset(
        context.state["test_rounds"],
        context.state["REPR"],
        "vision",
        detector=context.state["detector"],
        normaliser=context.state["DATA"]["post"]["train"].normaliser,
        phase="post",
        context=context,
    )
    context.state["ct_a"], context.state["ct_b"] = (
        evaluate(
            context.state["ct_model"],
            context.state["DATA"]["post"]["test"],
            context=context,
        ),
        evaluate(context.state["ct_model"], ct_vision_test, context=context),
    )
    print(f"{'':<34}{'accuracy':>10}{'macro-F1':>10}")
    for q, a, b in (
        ("Q1 attacker", context.state["exp_a"], context.state["exp_b"]),
        ("Q2 defender", context.state["ct_a"], context.state["ct_b"]),
    ):
        print(f"{q + ' A - oracle positions':<34}{a['accuracy']:>10.3f}{a['macro_f1']:>10.3f}")
        print(f"{q + ' B - vision pipeline':<34}{b['accuracy']:>10.3f}{b['macro_f1']:>10.3f}")
        print(
            f"{q + ' gap':<34}{a['accuracy'] - b['accuracy']:>10.3f}{a['macro_f1'] - b['macro_f1']:>10.3f}"
        )
    if preview:
        plot_confusion(
            context.state["exp_b"]["confusion"],
            f"Q1 attacker tactic - {context.state['best_name']} - full vision pipeline (test)",
        )
    # The inherited B-prime experiment leaks test rounds into validation.
    # Retained for reproducibility, explicitly disabled in normal runs.
    RUN_BPRIME = run_bprime
    if RUN_BPRIME:
        print("WARNING: legacy B-prime uses test rounds for selection; its score is invalid.")
        rng_bp = np.random.default_rng(context.config.seed + 9)
        bg_bp = build_background(
            context.config.minimap_px,
            np.random.default_rng(1234),
            context.state["MAP_BG_SOURCE"],
            context=context,
        )
        bp_rounds = context.state["train_rounds"][:60]
        for r in bp_rounds:
            if r.frames is None:
                r.frames = render_frames(
                    r.positions,
                    bg_bp,
                    context.config.minimap_px,
                    rng_bp,
                    context=context,
                )
        bp_train = SequenceDataset(
            bp_rounds,
            context.state["REPR"],
            "vision",
            detector=context.state["detector"],
            context=context,
        )
        bp_val = SequenceDataset(
            context.state["test_rounds"][:20],
            context.state["REPR"],
            "vision",
            detector=context.state["detector"],
            normaliser=bp_train.normaliser,
            context=context,
        )
        bp_model, _ = train_classifier(
            context.state["best_name"], bp_train, bp_val, verbose=False, context=context
        )
        bp_test = SequenceDataset(
            context.state["test_rounds"],
            context.state["REPR"],
            "vision",
            detector=context.state["detector"],
            normaliser=bp_train.normaliser,
            context=context,
        )
        context.state["exp_bp"] = evaluate(bp_model, bp_test, context=context)
        print(
            f"{chr(66) + chr(39) + ' - trained on vision':<24}{context.state['exp_bp']['accuracy']:>10.3f}{context.state['exp_bp']['macro_f1']:>10.3f}   (n_train={len(bp_rounds)})"
        )
    else:
        context.state["exp_bp"] = None


def describe_round_review(context):
    """Explicit workflow step migrated from original notebook cell 52."""
    print("Import RoundLabeller from stratscout.annotations; pass context=context.")
    print(
        "                   -> saves",
        Path(context.config.labels_dir) / "<match_id>.rounds.json",
    )
    print("then: build_rounds_dataset(context=context)   # all labelled videos -> one npz")
    print(
        "then:              real_rounds = load_real_rounds()   # Round.match_id = video file name"
    )


def prepare_pipeline(context):
    """Explicit workflow step migrated from original notebook cell 54."""
    context.state["pipeline"] = StratScoutPipeline(
        context.state["detector"],
        {
            "pre": (
                context.state["best_model"],
                context.state["DATA"]["pre"]["train"].normaliser,
            ),
            "post": (
                context.state["ct_model"],
                context.state["DATA"]["post"]["train"].normaliser,
            ),
        },
        context=context,
    )
    demo_round = next((r for r in context.state["test_rounds"] if r.planted))
    one = context.state["pipeline"].predict_window(demo_round.frames, "pre")
    two = context.state["pipeline"].predict_window(demo_round.post_frames, "post")
    print(
        f"before the plant: true {demo_round.label:<8} -> predicted {one['label']} ({one['confidence']:.2f})"
    )
    print(
        f"after the plant at {demo_round.plant_site}: true {demo_round.ct_label:<8} -> predicted {two['label']} ({two['confidence']:.2f})"
    )


def preview_report(context):
    """Explicit workflow step migrated from original notebook cell 55."""
    report = scouting_report(context.state["pipeline"], context.state["test_rounds"][:40])
    print(
        f"rounds analysed {report['n_used']}   attacker-tactic agreement {report['agreement']:.3f}"
    )
    for c, f in sorted(report["tendencies"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<9} {f * 100:5.1f}%  {'#' * int(f * 40)}")
    print(
        f"\nplanted rounds {report['n_planted']}   defender-response agreement {report['ct_agreement']:.3f}"
    )
    for c, f in sorted(report["ct_tendencies"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<9} {f * 100:5.1f}%  {'#' * int(f * 40)}")
    plot_routes(context.state["test_rounds"], "A_SPLIT", use_vision=False, context=context)
    plot_routes(context.state["test_rounds"], "A_SPLIT", use_vision=True, context=context)


def prepare_occupancy(context):
    """Explicit workflow step migrated from original notebook cell 57."""
    if Path(context.config.rounds_npz).exists():
        context.state["OBJ2_ROUNDS"], context.state["OBJ2_VISION"] = (
            load_real_rounds(context=context),
            True,
        )
    else:
        context.state["OBJ2_ROUNDS"], context.state["OBJ2_VISION"] = (
            context.state["test_rounds"],
            False,
        )
    print(
        f"Objective 2 on {len(context.state['OBJ2_ROUNDS'])} rounds ({('vision' if context.state['OBJ2_VISION'] else 'oracle')} positions)"
    )
    print(
        "attacker tactics:",
        {c: sum((r.label == c for r in context.state["OBJ2_ROUNDS"])) for c in T_CLASSES},
    )
    print(
        "defender responses (planted rounds):",
        {c: sum((r.ct_label == c for r in context.state["OBJ2_ROUNDS"])) for c in CT_CLASSES},
    )
    context.state["_POS_CACHE"]: Dict[Tuple[int, str], np.ndarray] = {}


def preview_side_occupancy(context):
    """Explicit workflow step migrated from original notebook cell 59."""
    show_side_occupancy(
        context.state["OBJ2_ROUNDS"],
        use_vision=context.state["OBJ2_VISION"],
        detector=context.state["detector"],
        context=context,
    )


def preview_tactic_occupancy(context):
    """Explicit workflow step migrated from original notebook cell 61."""
    show_tactic_occupancy(
        context.state["OBJ2_ROUNDS"],
        T_CLASSES,
        "pre",
        use_vision=context.state["OBJ2_VISION"],
        detector=context.state["detector"],
        context=context,
    )
    show_tactic_occupancy(
        context.state["OBJ2_ROUNDS"],
        CT_CLASSES,
        "post",
        use_vision=context.state["OBJ2_VISION"],
        detector=context.state["detector"],
        context=context,
    )


def preview_trails(context):
    """Explicit workflow step migrated from original notebook cell 63."""
    for lab in T_CLASSES + CT_CLASSES:
        show_trails(
            context.state["OBJ2_ROUNDS"],
            lab,
            use_vision=context.state["OBJ2_VISION"],
            detector=context.state["detector"],
            context=context,
        )


def preview_win_rates(context, *, preview=True):
    """Explicit workflow step migrated from original notebook cell 65."""
    context.state["WR"] = win_rate_report(
        context.state["train_rounds"] + context.state["val_rounds"] + context.state["test_rounds"]
    )
    if preview:
        plot_win_rates(context.state["WR"])
    print("\nrounds needed per group for a given precision (at p=0.5):")
    for hw in (0.2, 0.15, 0.1, 0.05):
        print(f"  +/- {hw * 100:4.0f} pp  ->  {rounds_needed(hw):>4} rounds")
    need = rounds_needed(0.1)
    plant_rate = np.mean(
        [
            r.planted
            for r in context.state["train_rounds"]
            + context.state["val_rounds"]
            + context.state["test_rounds"]
        ]
    )
    print(
        f"\nAt +/-10pp each group needs ~{need} rounds. Attacker tactics: {len(T_CLASSES)} x {need} = {need * len(T_CLASSES)} rounds.\nDefender responses: {len(CT_CLASSES)} x {need} = {need * len(CT_CLASSES)} planted rounds, which come from the same rounds (~{int(need * len(CT_CLASSES) / plant_rate)} at a {plant_rate:.0%} plant rate).\nSo ~{need * len(T_CLASSES)} labelled rounds (~{need * len(T_CLASSES) // 25} matches) answers both. Plan collection against this, not against a guess."
    )


def save_results(context):
    """Explicit workflow step migrated from original notebook cell 67."""
    out = Path(context.config.out_dir)
    for name in ("stratscout.pt", "summary.json"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    save_checkpoint(
        {
            "detector": context.state["detector"].state_dict(),
            "heads": {
                ph: {
                    "classifier": context.state["RESULTS"][ph][context.state["BEST"][ph]][
                        "model"
                    ].state_dict(),
                    "classifier_name": context.state["BEST"][ph],
                    "classes": PHASES[ph],
                    "normaliser": [t.cpu() for t in context.state["DATA"][ph]["train"].normaliser],
                }
                for ph in ("pre", "post")
            },
            "repr_mode": context.state["REPR"],
            "crop_box": context.state["CROP"],
            "config": asdict(context.config),
        },
        out / "stratscout.pt",
    )
    as_lists = lambda d: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
    summary = {
        "mode": "mixed" if context.state["USING_REAL_ICONS"] else "synthetic",
        "detector_data": "real" if context.state["USING_REAL_ICONS"] else "synthetic",
        "classifier_data": "synthetic",
        "crop_box": list(context.state["CROP"]),
        "detector": {k: float(v) for k, v in context.state["DET_METRICS"].items()},
        "pipeline_pos_err_px": float(np.mean(context.state["errs"]) * context.config.minimap_px),
        "classifiers": {
            ph: {
                n: {
                    "val_macro_f1": r["val"]["macro_f1"],
                    "test_macro_f1": r["test"]["macro_f1"],
                    "test_accuracy": r["test"]["accuracy"],
                }
                for n, r in context.state["RESULTS"][ph].items()
            }
            for ph in context.state["RESULTS"]
        },
        "experiment_A_oracle": {
            "attacker": {
                "accuracy": context.state["exp_a"]["accuracy"],
                "macro_f1": context.state["exp_a"]["macro_f1"],
            },
            "defender": {
                "accuracy": context.state["ct_a"]["accuracy"],
                "macro_f1": context.state["ct_a"]["macro_f1"],
            },
        },
        "experiment_B_vision": {
            "attacker": {
                "accuracy": context.state["exp_b"]["accuracy"],
                "macro_f1": context.state["exp_b"]["macro_f1"],
            },
            "defender": {
                "accuracy": context.state["ct_b"]["accuracy"],
                "macro_f1": context.state["ct_b"]["macro_f1"],
            },
        },
        "experiment_Bprime_vision_trained": {
            "accuracy": context.state["exp_bp"]["accuracy"],
            "macro_f1": context.state["exp_bp"]["macro_f1"],
        }
        if context.state["exp_bp"]
        else None,
        "objective3_win_rates": {
            "overall": as_lists(context.state["WR"].get("overall", {})),
            "by_tactic": {
                c: as_lists(d) for c, d in context.state["WR"].get("by_tactic", {}).items()
            },
            "by_response": {
                c: as_lists(d) for c, d in context.state["WR"].get("by_response", {}).items()
            },
        },
    }
    write_new_text(out / "summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_prototype(context, *, plots=False):
    """Run the extracted experiment explicitly. Classifier rounds remain synthetic.

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
        check_features,
        fit_classifiers,
        compare_oracle_and_vision,
        prepare_pipeline,
        prepare_occupancy,
        preview_win_rates,
    ):
        if step in (fit_detector, compare_oracle_and_vision, preview_win_rates):
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
            preview_features,
            describe_classifiers,
            preview_classifiers,
            preview_report,
            preview_side_occupancy,
            preview_tactic_occupancy,
            preview_trails,
        ):
            step(context)
    save_results(context)
    return context
