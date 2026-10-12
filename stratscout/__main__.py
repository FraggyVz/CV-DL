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
    prepare.add_argument("--start", type=float, default=0)
    prepare.add_argument("--duration", type=float)
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
    args = parser.parse_args(argv)
    if args.command not in ("prototype", "ingest"):
        try:
            if args.command == "process-recordings":
                from .recording_batch import process_recordings

                process_recordings(args.input, args.profile, args.output, at=args.at)
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
                )
                print(
                    f"Validated {len(result['variants'])} variants: {args.output / 'manifest.json'}"
                )
            elif args.command == "compare-fps":
                from .fps_comparison import compare_fps

                compare_fps(args.input, args.windows, output=args.output)
                print(f"Comparison: {(args.output or args.input / 'comparison') / 'report.html'}")
        except KeyboardInterrupt:
            parser.exit(
                130, "StratScout: cancelled; source preserved. Check incomplete run metadata.\n"
            )
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
