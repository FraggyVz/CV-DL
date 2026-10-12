"""Command-line entry points. Argument parsing never starts a workflow on import."""

import argparse
import json
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
    args = parser.parse_args(argv)
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
