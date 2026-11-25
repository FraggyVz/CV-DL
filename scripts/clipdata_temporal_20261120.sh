#!/usr/bin/env bash
# Mean pooling, temporal average pooling and the Compact pooled GRU on the shared clip
# data: nested epoch selection, then other-folds epoch selection. With the default tag this
# made runs/tactic-clipdata-nested-20261120 and runs/tactic-clipdata-other-folds-20261120.
#
# Usage, from anywhere (Git Bash on Windows, or any bash):
#   bash scripts/clipdata_temporal_20261120.sh [TAG]
# TAG names the outputs: runs/tactic-clipdata-{nested,other-folds}-TAG and the log
# runs/tactic-clipdata-TAG.log (default 20261120, the reported runs; pick a new tag to rerun).
#
# Needs the train-only copies (no test clip is ever read):
#   data/processed/datasets/2026-11-16-train-only   (export with an empty test manifest)
#   data/processed/clip-data-2026-11-16-train-only  (train .npz files, channels.json and
#                                                    an index.csv with only train rows)
# The output folders must not exist yet. Every log line is timestamped. The script stops at
# the first failure, so other-folds never runs after a failed nested run. No shuffles.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-20261120}
PYTHON=${PYTHON:-.venv/Scripts/python.exe}
[ -x "$PYTHON" ] || PYTHON=python
LOG=runs/tactic-clipdata-$TAG.log
stamp() { while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done >> "$LOG"; }
for mode in nested other-folds; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $mode: start" >> "$LOG"
    status=0
    "$PYTHON" -u -m stratscout tactic-temporal \
        --manifest data/processed/datasets/2026-11-16-train-only/train/manifest.csv \
        --output "runs/tactic-clipdata-$mode-$TAG" \
        --exclude-uncertain --seeds 0 1 2 --models mean thirds gru \
        --inputs clip-data --clip-data data/processed/clip-data-2026-11-16-train-only \
        --epoch-mode "$mode" 2>&1 | stamp || status=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $mode: end (exit $status)" >> "$LOG"
    if [ "$status" -ne 0 ]; then
        echo "$mode run failed (exit $status); see $LOG" >&2
        exit "$status"
    fi
done
