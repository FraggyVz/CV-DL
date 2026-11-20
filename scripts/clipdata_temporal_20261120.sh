#!/usr/bin/env bash
# Mean pooling, temporal average pooling and the final GRU on the shared clip data,
# nested epoch selection, then the TCN's other-folds rule. This made
# runs/tactic-clipdata-nested-20261120 and runs/tactic-clipdata-other-folds-20261120.
#
# Usage, from anywhere (Git Bash on Windows, or any bash):
#   bash scripts/clipdata_temporal_20261120.sh
#
# Needs the train-only copies (no test clip is ever read):
#   data/processed/datasets/2026-11-16-train-only   (export with an empty test manifest)
#   data/processed/clip-data-2026-11-16-train-only  (train .npz files, channels.json and
#                                                    an index.csv with only train rows)
# The output folders must not exist yet. Every log line is timestamped in
# runs/tactic-clipdata-20261120.log. No shuffles.
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-.venv/Scripts/python.exe}
[ -x "$PYTHON" ] || PYTHON=python
LOG=runs/tactic-clipdata-20261120.log
stamp() { while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done >> "$LOG"; }
for mode in nested other-folds; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $mode: start" >> "$LOG"
    "$PYTHON" -u -m stratscout tactic-temporal \
        --manifest data/processed/datasets/2026-11-16-train-only/train/manifest.csv \
        --output "runs/tactic-clipdata-$mode-20261120" \
        --exclude-uncertain --seeds 0 1 2 --models mean thirds gru \
        --inputs clip-data --clip-data data/processed/clip-data-2026-11-16-train-only \
        --epoch-mode "$mode" 2>&1 | stamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $mode: end (exit ${PIPESTATUS[0]})" >> "$LOG"
done
