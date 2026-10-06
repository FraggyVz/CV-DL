"""Run from the repository root with python -m stratscout."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Two-team colour detection from recorded minimap video")
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Select crop, sample T/CT colours and save a profile")
    calibration.add_argument("--video", type=Path, required=True)
    calibration.add_argument("--profile", type=Path, required=True, help="New profile filename; never overwritten")
    calibration.add_argument("--frame", type=int, default=0, help="Initial frame index for crop selection")
    detection = commands.add_parser("detect", help="Export annotated MP4, detections and run metadata")
    detection.add_argument("--video", type=Path, required=True)
    detection.add_argument("--profile", type=Path, required=True)
    detection.add_argument("--output", type=Path, required=True, help="New run directory; must not exist")
    evaluation = commands.add_parser("evaluate", help="Score reviewed held-out point annotations")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--annotations", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True, help="New metrics JSON filename")
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError:
        print("Missing OpenCV/NumPy. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    try:
        if args.command == "calibrate":
            from .calibration import calibrate
            calibrate(args.video, args.profile, args.frame)
        elif args.command == "detect":
            from .config import load_profile
            from .video import detect_video
            result = detect_video(args.video, load_profile(args.profile), args.output)
            print(f"Exported {result['frames_processed']} frames to {args.output}")
        else:
            from .evaluation import evaluate_run
            result = evaluate_run(args.run, args.annotations, args.output)
            print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled. Any partial detection run is marked failed.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
