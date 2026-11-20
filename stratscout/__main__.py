"""Command-line entry points. Argument parsing never starts a workflow on import."""

import argparse
import json
import subprocess
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="StratScout modular prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    prototype = sub.add_parser(
        "prototype", help="Run the vision experiment: crop, detector and tracker"
    )
    prototype.add_argument("--config", type=Path, help="JSON object of Config fields")
    prototype.add_argument("--output", type=Path, required=True, help="New run directory")
    prototype.add_argument("--smoke", action="store_true", help="Small synthetic CPU wiring check")
    prototype.add_argument("--plots", action="store_true", help="Show interactive plots")
    prototype.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    ingest = sub.add_parser("ingest", help="Run the companion demo-development workflow")
    ingest.add_argument("--config", type=Path, help="JSON object of DemCfg fields")
    ingest.add_argument("--stage", choices=["radar", "parse", "map", "export"], required=True)
    ingest.add_argument("--plots", action="store_true")
    crop = sub.add_parser("video-profile", help="Select and preview a native minimap crop")
    crop.add_argument("--video", type=Path, required=True)
    crop.add_argument("--output", type=Path, required=True)
    crop.add_argument("--at", type=float, default=0, help="Preview source timestamp in seconds")
    crop.add_argument("--crop", type=int, nargs=4, help="Explicit x0 y0 x1 y1; bypass GUI")
    prepare = sub.add_parser("prepare-video", help="Encode aligned minimap variants once")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--fps", type=int, nargs="+", default=[5, 10, 30])
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--start", type=float, default=0, help="Trim start in raw-video seconds")
    prepare.add_argument("--duration", type=float, help="Retained duration in raw-video seconds")
    prepare.add_argument("--playback-speed", type=float, default=1, help="Demo playback multiplier")
    prepare.add_argument("--match-id")
    compare = sub.add_parser(
        "compare-fps", help="Compare visual quality, timing, size and decode cost"
    )
    compare.add_argument("--input", type=Path, required=True)
    compare.add_argument(
        "--windows", nargs="+", required=True, help="Source start:duration in seconds"
    )
    compare.add_argument("--output", type=Path, help="Defaults to INPUT/comparison; must not exist")
    batch = sub.add_parser("process-recordings", help="Process all finished videos in an inbox")
    batch.add_argument("--input", type=Path, default=Path("data/raw/inbox"))
    batch.add_argument("--profile", type=Path, default=Path("configs/recording-video.json"))
    batch.add_argument("--output", type=Path, default=Path("data/processed/recordings"))
    batch.add_argument("--at", type=float, default=10, help="First crop preview time in seconds")
    batch.add_argument(
        "--playback-speed", type=float, default=2, help="Default speed for new recordings (2x)"
    )
    batch.add_argument(
        "--non-interactive",
        action="store_true",
        help="Keep whole videos without prompts; requires a crop profile",
    )
    bomb = sub.add_parser("bomb-scan", help="Find the C4 badge by shape; list dropped candidates")
    bomb.add_argument("--video", type=Path, required=True, help="Prepared minimap video")
    bomb.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    bomb.add_argument("--rim", type=int, default=4, help="Carrier rim band around the badge (px)")
    bomb.add_argument("--min-seconds", type=float, default=2.0, help="Shortest still run")
    bomb.add_argument("--t-hue", type=float, default=27.0, help="T icon hue (OpenCV 0-180)")
    bomb.add_argument(
        "--shape-only",
        action="store_true",
        help="Skip the orange colour split (old rule, for comparison)",
    )
    live = sub.add_parser("bomb-preview", help="Play a minimap video with live detections")
    live.add_argument("--video", type=Path, help="Prepared minimap video (else pick a match)")
    live.add_argument("--match", help="Match folder name fragment, e.g. spirit")
    live.add_argument("--fps", type=int, default=10, help="Variant to play when picking a match")
    live.add_argument("--round", type=int, help="Start at this estimated round")
    live.add_argument("--start", type=float, default=0, help="Start time in seconds")
    live.add_argument("--rim", type=int, default=4, help="Carrier rim band around the badge (px)")
    live.add_argument("--min-seconds", type=float, default=2.0, help="Still time before dropped")
    live.add_argument("--t-hue", type=float, default=27.0, help="T icon hue (OpenCV 0-180)")
    live.add_argument("--scale", type=int, default=2, help="Display magnification")
    seg = sub.add_parser("label-segments", help="Label attack/retake segments with the keyboard")
    seg.add_argument("--match", help="Match folder name fragment, e.g. furia (else pick one)")
    seg.add_argument("--reviewer", help="Your short name; each reviewer gets their own file")
    seg.add_argument("--round", type=int, help="Start at this estimated round")
    seg.add_argument("--start", type=float, default=0, help="Start time in seconds")
    seg.add_argument("--fps", type=int, default=10, help="Prepared variant to label on")
    seg.add_argument("--labels-dir", type=Path, default=Path("labels"))
    seg.add_argument("--scale", type=int, default=2, help="Display magnification")
    review = sub.add_parser("review-segments", help="Mark agreed segments as reviewed")
    review.add_argument("--match", required=True, help="Match folder name fragment")
    review.add_argument("--reviewer", required=True, help="Whose label file to update")
    review.add_argument("--by", required=True, help="Who agreed the labels, e.g. gin")
    pick = review.add_mutually_exclusive_group(required=True)
    pick.add_argument("--ids", type=int, nargs="+", help="Segment numbers to mark reviewed")
    pick.add_argument(
        "--all-labelled", action="store_true", help="Every pending segment that has a class"
    )
    review.add_argument("--except", dest="skip", type=int, nargs="*", default=[])
    review.add_argument("--labels-dir", type=Path, default=Path("labels"))
    export = sub.add_parser("export-segments", help="Cut reviewed segments into a clip dataset")
    export.add_argument("--test", nargs="+", required=True, help="Held-out match fragments")
    export.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    export.add_argument("--reviewer", help="Use this reviewer's files only")
    export.add_argument("--include-pending", action="store_true", help="Also export pending")
    export.add_argument("--fps", type=int, default=10, help="Prepared variant to cut from")
    export.add_argument("--no-clips", action="store_true", help="Write the manifest only")
    export.add_argument("--labels-dir", type=Path, default=Path("labels"))
    tactic = sub.add_parser(
        "train-tactics", help="Cross-validate the tactic TCN on an exported clip dataset"
    )
    tactic.add_argument("--dataset", type=Path, required=True, help="export-segments folder")
    tactic.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    tactic.add_argument("--epochs", type=int, default=150, help="Most epochs to try")
    tactic.add_argument("--seeds", type=int, default=5, help="Training seeds per setting")
    tactic.add_argument("--grid", type=int, default=8, help="Occupancy grid cells per side")
    tactic.add_argument("--models", nargs="+", default=["tcn", "control"])
    tactic.add_argument("--cache", type=Path, default=Path("runs/tactic-cache"))
    tactic.add_argument(
        "--clip-data",
        type=Path,
        help="Train on shared clip data from build-clip-data instead of the original features",
    )
    tactic.add_argument(
        "--resnet-pcs",
        type=int,
        default=32,
        help="With --clip-data: ResNet numbers per row, compressed per fold (0 = detector only)",
    )
    tactic.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    tactic.add_argument(
        "--exclude-uncertain",
        action="store_true",
        help="Omit clips with uncertain boundaries (listed in the report) instead of refusing",
    )
    tactic.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also train on all training matches and score the test match (do this once)",
    )
    shared = sub.add_parser(
        "build-clip-data",
        help="Build the shared model-ready data (detector channels + ResNet) for every clip",
    )
    shared.add_argument("--dataset", type=Path, required=True, help="export-segments folder")
    shared.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    shared.add_argument("--cache", type=Path, default=Path("runs/tactic-cache"))
    shared.add_argument("--no-resnet", action="store_true", help="Detector channels only")
    shared.add_argument("--device", default="cpu")
    guess = sub.add_parser(
        "predict-tcn", help="Classify new prepared clips with the saved TCN seeds"
    )
    guess.add_argument("clips", nargs="+", type=Path, help="Clip files or folders of .mp4 clips")
    guess.add_argument(
        "--models",
        type=Path,
        default=Path("runs/tactic-tcn-clipdata-final/models"),
        help="models/ folder written by train-tactics --evaluate-test",
    )
    guess.add_argument("--kind", default="tcn", choices=["tcn", "control"])
    guess.add_argument("--output", type=Path, help="Also write a new CSV of predictions")
    guess.add_argument(
        "--background-from",
        nargs="+",
        type=Path,
        help="Videos of the same match for the empty-map background (default: the clips)",
    )
    guess.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Flag predictions whose top probability is below this",
    )
    guess.add_argument(
        "--regions",
        type=Path,
        help="Map-regions JSON for clip-data models (default: stratscout/mirage_regions.json)",
    )
    guess.add_argument("--device", default="cpu")
    counts = sub.add_parser(
        "icon-count-eval", help="Fit icon splitting on some matches; score team counts on others"
    )
    counts.add_argument("--train", nargs="+", required=True, help="Match name fragments to fit on")
    counts.add_argument("--test", nargs="+", required=True, help="Held-out match name fragments")
    counts.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    counts.add_argument("--fps", type=int, default=5, help="Prepared variant to evaluate")
    tactics = sub.add_parser("tactic-baseline", help="Train Plan A using whole-match validation")
    tactics.add_argument("--manifest", type=Path, required=True, help="Exported train/manifest.csv")
    tactics.add_argument("--output", type=Path, required=True, help="New run directory")
    tactics.add_argument("--exclude-uncertain", action="store_true")
    tactics.add_argument(
        "--size", type=int, default=224, help="Whole-map square size, e.g. 224 or 320"
    )
    tactics.add_argument("--frames", type=int, default=16)
    tactics.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    tactics.add_argument("--epochs", type=int, default=200)
    tactics.add_argument("--patience", type=int, default=30)
    epoch_choice = tactics.add_mutually_exclusive_group()
    epoch_choice.add_argument(
        "--fixed-epochs", type=int, metavar="N", help="Train exactly N epochs; no early stopping"
    )
    epoch_choice.add_argument(
        "--cross-fold",
        dest="fixed_epochs",
        action="store_const",
        const=None,
        help="Choose each epoch without the scored match (default): the other fold with two "
        "matches, nested with three or more",
    )
    tactics.add_argument(
        "--epoch-mode",
        choices=["nested", "other-folds"],
        default="nested",
        help="nested (default): each match's epoch from its training matches only; "
        "other-folds: the TCN's rule, the epoch best on the other folds (slightly optimistic)",
    )
    tactics.add_argument("--device", default="cpu")
    temporal = sub.add_parser(
        "tactic-temporal", help="Train temporal average pooling and GRU on the baseline's protocol"
    )
    temporal.add_argument(
        "--manifest", type=Path, required=True, help="Exported train/manifest.csv"
    )
    temporal.add_argument("--output", type=Path, required=True, help="New run directory")
    temporal.add_argument("--exclude-uncertain", action="store_true")
    temporal.add_argument(
        "--models",
        nargs="+",
        default=["mean", "thirds", "gru"],
        choices=["mean", "thirds", "random-thirds", "gru"],
        help="thirds = temporal average pooling (3 segments) + linear classifier; "
        "random-thirds = random-segment average pooling + linear classifier (control)",
    )
    temporal.add_argument(
        "--inputs",
        choices=["images", "positions", "clip-data"],
        default="images",
        help="Per-frame features: ResNet-18 images, detector positions or the shared clip data",
    )
    temporal.add_argument(
        "--clip-data",
        type=Path,
        help="With --inputs clip-data: a train-only build-clip-data folder",
    )
    temporal.add_argument(
        "--no-class-weights",
        dest="class_weights",
        action="store_false",
        help="Ablation: train without balanced class weights in the loss (default: weighted)",
    )
    temporal.add_argument(
        "--drop-channels",
        nargs="+",
        default=[],
        choices=["t_observed", "ct_observed", "t_inferred_carrier", "bomb_badge"],
        help="Position channels to remove, for an ablation",
    )
    temporal.add_argument(
        "--features", type=Path, help="Reuse an earlier frame- or position-features.pt"
    )
    temporal.add_argument(
        "--shuffles", type=int, default=0, help="Shuffled-label refits per model (0 skips)"
    )
    temporal.add_argument("--size", type=int, default=224)
    temporal.add_argument("--frames", type=int, default=16)
    temporal.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    temporal.add_argument("--epochs", type=int, default=200)
    temporal.add_argument("--patience", type=int, default=30)
    temporal.add_argument(
        "--fixed-epochs", type=int, metavar="N", help="Train exactly N epochs; no early stopping"
    )
    temporal.add_argument(
        "--gru-projection", type=int, default=8, help="GRU frame projection size (first GRU: 32)"
    )
    temporal.add_argument("--gru-hidden", type=int, default=16, help="GRU state size (first: 32)")
    temporal.add_argument(
        "--gru-dropout", type=float, default=0.3, help="GRU dropout in training (first: 0)"
    )
    temporal.add_argument(
        "--gru-readout",
        choices=["mean-max", "last"],
        default="mean-max",
        help="GRU readout: mean+max over all real frames (default) or the last state (earlier GRUs)",
    )
    temporal.add_argument(
        "--epoch-mode",
        choices=["nested", "other-folds"],
        default="nested",
        help="nested (default): each match's epoch from its training matches only; "
        "other-folds: the TCN's rule, the epoch best on the other folds (slightly optimistic)",
    )
    temporal.add_argument("--device", default="cpu")
    check = sub.add_parser(
        "detector-check", help="Score detector positions against demo positions (Furia/Spirit)"
    )
    check.add_argument("--matches", nargs="+", required=True, help="e.g. furia spirit; never navi")
    check.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    check.add_argument("--demos-dir", type=Path, default=Path("demos"))
    check.add_argument("--every", type=float, default=1.0, help="Seconds between scored frames")
    check.add_argument("--fps", type=int, default=10, help="Prepared variant to read")
    args = parser.parse_args(argv)
    if args.command not in ("prototype", "ingest"):
        try:
            if args.command == "tactic-baseline":
                from .context import RunContext
                from .tactic_training import BaselineConfig, run_baseline

                run_baseline(
                    args.manifest,
                    args.output,
                    context=RunContext(device=args.device),
                    config=BaselineConfig(
                        size=args.size,
                        n_frames=args.frames,
                        seeds=tuple(args.seeds),
                        epochs=args.epochs,
                        patience=args.patience,
                        fixed_epochs=args.fixed_epochs,
                    ),
                    exclude_uncertain=args.exclude_uncertain,
                    epoch_mode=args.epoch_mode,
                )
            elif args.command == "tactic-temporal":
                from .context import RunContext
                from .tactic_temporal import run_temporal
                from .tactic_training import BaselineConfig

                run_temporal(
                    args.manifest,
                    args.output,
                    context=RunContext(device=args.device),
                    config=BaselineConfig(
                        size=args.size,
                        n_frames=args.frames,
                        seeds=tuple(args.seeds),
                        epochs=args.epochs,
                        patience=args.patience,
                        fixed_epochs=args.fixed_epochs,
                        class_weights=args.class_weights,
                    ),
                    exclude_uncertain=args.exclude_uncertain,
                    models=tuple(args.models),
                    features=args.features,
                    shuffles=args.shuffles,
                    inputs=args.inputs,
                    drop_channels=tuple(args.drop_channels),
                    epoch_mode=args.epoch_mode,
                    gru={
                        "projection": args.gru_projection,
                        "hidden": args.gru_hidden,
                        "dropout": args.gru_dropout,
                        "readout": args.gru_readout,
                    },
                    clip_data=args.clip_data,
                )
            elif args.command == "detector-check":
                from .detector_check import CheckConfig, run_detector_check

                run_detector_check(
                    args.matches,
                    args.output,
                    demos_dir=args.demos_dir,
                    config=CheckConfig(every_s=args.every, fps=args.fps),
                )
            elif args.command == "bomb-scan":
                from .bomb import write_bomb_scan
                from .context import RunContext

                summary = write_bomb_scan(
                    args.video,
                    args.output,
                    band=args.rim,
                    min_seconds=args.min_seconds,
                    t_hue=args.t_hue,
                    shape_only=args.shape_only,
                    context=RunContext(),
                )
                print(
                    f"Badge found in {summary['frames_with_badge']}/{summary['frames']} frames; "
                    f"{summary['dropped_candidates']} dropped candidates: {args.output}"
                )
            elif args.command == "icon-count-eval":
                from .count_eval import run_count_eval
                from .live_preview import choose_match, list_matches, match_video

                matches = list_matches()
                pick = lambda names: [
                    match_video(choose_match(matches, n), args.fps) for n in names
                ]  # noqa: E731
                report = run_count_eval(pick(args.train), pick(args.test), args.output)
                print(f"Fitted parameters: {report['params']}")
                for video, r in report["results"].items():
                    b, s = r["baseline"]["mean_exact5"], r["discs_markers_capped"]["mean_exact5"]
                    print(
                        f"{r['split']:5} {Path(video).parent.name}: frames with 5 T and 5 CT "
                        f"cells {b:.1%} -> {s:.1%}"
                    )
                print(f"Report: {args.output / 'report.json'}")
            elif args.command == "bomb-preview":
                from .live_preview import choose_match, list_matches, live_preview, match_video

                video, rounds_video = args.video, None
                if video is None:
                    folder = choose_match(list_matches(), args.match)
                    video, rounds_video = match_video(folder, args.fps), match_video(folder, 5)
                print(f"Opening {video}")
                live_preview(
                    video,
                    start=args.start,
                    round_number=args.round,
                    rounds_video=rounds_video,
                    band=args.rim,
                    min_seconds=args.min_seconds,
                    t_hue=args.t_hue,
                    scale=args.scale,
                )
            elif args.command == "label-segments":
                from .live_preview import choose_match, list_matches, match_video
                from .rounds import cached_rounds
                from .segment_labeller import SegmentLabeller, clock, play_start
                from .segments import SegmentFile, match_identity, segments_json

                folder = choose_match(list_matches(), args.match)
                reviewer = args.reviewer or input("Your short name (e.g. gin): ")
                video = match_video(folder, args.fps)
                path = segments_json(folder, reviewer, args.labels_dir)
                store = SegmentFile(
                    path,
                    match_id=match_identity(folder),
                    match_folder=folder,
                    video=video,
                    video_fps=int(video.stem[8:-3]),  # minimap-<n>fps; may fall back from --fps
                    reviewer=reviewer,
                )
                print("Estimating rounds (the first run for a match scans the video)...")
                rounds = cached_rounds(match_video(folder, 5))
                print("Estimated rounds (navigation aids, not labels):")
                for r in rounds:
                    note = "  started before recording" if r["partial"] else ""
                    print(f"  round {r['round']:2}  play from {clock(play_start(r))}{note}")
                start = args.start
                if args.round is not None:
                    if not 1 <= args.round <= len(rounds):
                        raise ValueError(f"--round must be 1-{len(rounds)} for this match")
                    start = play_start(rounds[args.round - 1])
                print(f"Opening {video}\nSegments save on S to {path}")
                SegmentLabeller(store, rounds).run(video, start=start, scale=args.scale)
                print(f"{len(store.segments)} segments in {path}: {store.summary()}")
            elif args.command == "review-segments":
                from .live_preview import choose_match, list_matches, match_video
                from .segments import SegmentFile, match_identity, segments_json

                folder = choose_match(list_matches(), args.match)
                path = segments_json(folder, args.reviewer, args.labels_dir)
                if not path.exists():
                    raise ValueError(f"no label file {path}")
                fps = json.loads(path.read_text(encoding="utf-8"))["video_fps"]
                store = SegmentFile(
                    path,
                    match_id=match_identity(folder),
                    match_folder=folder,
                    video=match_video(folder, fps),
                    video_fps=fps,
                    reviewer=args.reviewer,
                )
                ids = args.ids or [
                    s["id"]
                    for s in store.segments
                    if s["status"] == "pending" and s.get("label") and s["id"] not in args.skip
                ]
                done = store.review(ids, args.by)
                print(f"Marked {len(done)} segments reviewed in {path}: {[s['id'] for s in done]}")
            elif args.command == "export-segments":
                from .segment_export import export_segments

                summary = export_segments(
                    args.labels_dir,
                    Path("data/processed/recordings"),
                    args.output,
                    args.test,
                    reviewer=args.reviewer,
                    include_pending=args.include_pending,
                    fps=args.fps,
                    clips=not args.no_clips,
                )
                print(f"{summary['clips']} clips in {args.output}")
                for split, counts in summary["counts"].items():
                    print(f"  {split}: {counts}")
                print(f"  skipped: {summary['skipped']}")
            elif args.command == "train-tactics":
                from .tactic_tcn import TacticConfig, run_tactic_experiment

                cfg = TacticConfig(
                    epochs=args.epochs,
                    seeds=tuple(range(args.seeds)),
                    grid=args.grid,
                    models=tuple(args.models),
                    input="clip-data" if args.clip_data else "detections",
                    resnet_pcs=args.resnet_pcs,
                )
                run_tactic_experiment(
                    args.dataset,
                    args.output,
                    cfg,
                    evaluate_test=args.evaluate_test,
                    exclude_uncertain=args.exclude_uncertain,
                    cache_dir=args.cache,
                    clip_data=args.clip_data,
                    device=args.device,
                )
                print(f"Report: {args.output / 'report.md'}")
            elif args.command == "build-clip-data":
                from .clip_data import build_clip_data

                build_clip_data(
                    args.dataset,
                    args.output,
                    cache_dir=args.cache,
                    with_resnet=not args.no_resnet,
                    device=args.device,
                )
                print(f"Report: {args.output / 'report.md'}")
            elif args.command == "predict-tcn":
                from .tcn_predict import run_predict

                run_predict(
                    args.clips,
                    args.models,
                    kind=args.kind,
                    output=args.output,
                    background_from=args.background_from,
                    min_confidence=args.min_confidence,
                    device=args.device,
                    regions_path=args.regions,
                )
            elif args.command == "process-recordings":
                from .recording_batch import process_recordings

                process_recordings(
                    args.input,
                    args.profile,
                    args.output,
                    at=args.at,
                    playback_speed=args.playback_speed,
                    interactive=not args.non_interactive,
                )
            elif args.command == "video-profile":
                from .video_profiles import create_profile

                create_profile(args.video, args.output, crop=args.crop, at=args.at)
                print(f"Crop profile saved: {args.output}")
            elif args.command == "prepare-video":
                from .video_preparation import prepare_video

                result = prepare_video(
                    args.video,
                    args.profile,
                    args.output,
                    rates=args.fps,
                    start=args.start,
                    duration=args.duration,
                    match_id=args.match_id,
                    playback_speed=args.playback_speed,
                )
                print(
                    f"Validated {len(result['variants'])} variants: {args.output / 'manifest.json'}"
                )
            elif args.command == "compare-fps":
                from .fps_comparison import compare_fps

                compare_fps(args.input, args.windows, output=args.output)
                print(f"Comparison: {(args.output or args.input / 'comparison') / 'report.html'}")
        except (KeyboardInterrupt, EOFError):
            parser.exit(130, "StratScout: cancelled or terminal input closed; source preserved.\n")
        except (OSError, ValueError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
            parser.exit(1, f"StratScout: {exc}\n")
        return
    import matplotlib

    if not args.plots:
        matplotlib.use("Agg")
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    try:
        if args.command == "prototype":
            from .config import Config
            from .context import RunContext
            from .workflow import run_prototype

            settings["out_dir"] = str(args.output)
            cfg = Config(**settings)
            if args.smoke:
                import torch

                torch.set_num_threads(1)
                cfg.minimap_px = 32
                cfg.det_epochs = 1
                cfg.det_batch = 8
                cfg.test_matches = cfg.detector_matches = 1
                cfg.detector_train_frames, cfg.detector_val_frames = 32, 16
                # A smoke run must not discover or rewrite real inputs.
                cfg.data_dir = str(args.output / "empty-inputs")
                cfg.video_path = None
                for name in (
                    "frames_npy",
                    "icon_labels_json",
                    "frames_src_json",
                    "crops_json",
                    "map_image",
                ):
                    setattr(cfg, name, str(args.output / Path(getattr(cfg, name)).name))
            args.output.mkdir(parents=True, exist_ok=False)
            context = RunContext(cfg, device="cpu" if args.smoke else args.device)
            run_prototype(context, plots=args.plots)
        else:
            from .ingest import workflow
            from .ingest.config import DemCfg
            from .ingest.context import IngestContext

            context = IngestContext(DemCfg(**settings))
            if args.stage == "radar":
                workflow.prepare_radar_videos(context)
            else:
                workflow.parse_demos(context)
                if args.stage in ("map", "export"):
                    workflow.prepare_background(context)
                    workflow.prepare_mapping(context)
                if args.stage == "export":
                    workflow.export_dataset(context)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(1, f"StratScout: {exc}\n")


if __name__ == "__main__":
    main()
