"""Command-line entry points. Argument parsing never starts a workflow on import."""

import argparse
import json
import subprocess
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="StratScout modular prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    prototype = sub.add_parser("prototype", help="Run the existing experimental pipeline")
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
    counts = sub.add_parser(
        "icon-count-eval", help="Fit icon splitting on some matches; score team counts on others"
    )
    counts.add_argument("--train", nargs="+", required=True, help="Match name fragments to fit on")
    counts.add_argument("--test", nargs="+", required=True, help="Held-out match name fragments")
    counts.add_argument("--output", type=Path, required=True, help="New folder; must not exist")
    counts.add_argument("--fps", type=int, default=5, help="Prepared variant to evaluate")
    args = parser.parse_args(argv)
    if args.command not in ("prototype", "ingest"):
        try:
            if args.command == "bomb-scan":
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
                cfg.det_epochs = cfg.cls_epochs = 1
                cfg.cls_hidden = 16
                cfg.det_batch = 8
                cfg.train_matches = cfg.val_matches = cfg.test_matches = cfg.detector_matches = 1
                cfg.detector_train_frames, cfg.detector_val_frames = 32, 16
                # A smoke run must not discover or rewrite real inputs.
                cfg.data_dir = str(args.output / "empty-inputs")
                cfg.video_path = None
                for name in (
                    "frames_npy",
                    "icon_labels_json",
                    "frames_src_json",
                    "crops_json",
                    "rounds_npz",
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
