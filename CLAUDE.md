# StratScout — Project Handoff

## Current direction

Use the shared modular prototype as the base. Implementation is in `stratscout/`;
the two notebooks are thin experiment interfaces. Read [AGENTS.md](AGENTS.md),
[README](README.md), [architecture](docs/architecture.md) and the
[baseline review](docs/base-review-2026-10-12.md) before changing behaviour.

On 14 October 2026 the user requested a recording-data reset. Old recordings,
exports, capture diagnostics/scripts, crop profiles and unlabelled pilot caches
were removed. Do not restore them or reuse their crop coordinates by default.
Reusable model checkpoints, map assets and regression fixtures remain.

## Recording workflow

The current setup is manual OBS capture at **2560×1440, 60 fps, constant 2× replay**.
Do not touch a file while OBS is recording it. After recording stops, place the file
in `data/raw/inbox/` and run `process-recordings.ps1`.

The batch prompts for speed (default 2), start and end in raw-video time. First use
selects the minimap crop and creates the local `configs/recording-video.json`.
There are no per-video settings files. Processing trims, restores 1× playback,
crops once and directly encodes 5/10/30 fps variants. It also generates visual
comparisons and run metadata. Completed bundles are skipped without prompting.
Output folders use cleaned match filenames; full source hashes in the manifests
still identify duplicates and distinguish collisions. Share complete match folders
as a ZIP so report links, variants and provenance stay together.

Keep encoding in the shared preparation API. The standalone `prepare-video`
command defaults to speed 1; explicitly pass `--playback-speed 2` for these captures.
No OBS WebSocket integration or replay automation is needed.
See [recording setup](docs/recording-setup.md) and
[validation status](docs/processing-validation.md). The new capture has not yet
passed real-footage acceptance. Nominal FPS does not prove unique rendered frames.

## Research scope and contracts

Vision inference takes minimap pixels. Demo-derived positions may support
development/evaluation, but must not leak into video-only inference.

- Before plant: `A_RUSH`, `B_RUSH`, `A_SPLIT`, `B_SPLIT`, `DEFAULT`.
- After plant: `RETAKE`, `TRICKLE`, `SAVE`.
- The CNN heatmaps use channel 0 for T and channel 1 for CT.
- OpenCV decoding returns BGR; vision tensors use RGB. Keep input frames uint8
  where preprocessing performs its own division by 255.
- Centre heatmaps must contain positive peaks; preserve the integer-peak handling.
- Retain the colour detector as a baseline. Numbers, objective graphics, death
  markers and overlapping icons need explicit labelled evaluation.
- Do not invent invisible players when annotating overlaps or bomb occlusion.
- Keep splits grouped by match, including all FPS variants and demo-derived samples.
- Round winner is an annotation/reporting field, never a vision-model input.

## Unresolved problems

The baseline review documents annotation overwrite/empty-frame loss, missing-player
tracking, non-finite motion features, incompatible ingest labels and evaluation
leakage. The B-prime experiment remains disabled. Real classifier training is not
connected; synthetic scores are not real-match accuracy.

The retained fixed-slot tracker can invent or retain stale positions. It is not
a validated solution to occlusion. Repair observed/missing state and feature
handling together. Avoid blanket masks over playable map areas.

The companion demo-ingest utilities remain reusable research code, separate from
the recording processor. Their legacy 8 fps exporter is not the current preparation
route. World-to-radar mapping and demo/video synchronisation require calibration
for the new crop; previous fitted offsets must not be assumed valid.

## Next work and checks

Finish and validate the new recording, then protect labels, establish shared match
identities and normal-time samples, and measure detector/tracker behaviour before
choosing a training frame rate. Follow [development direction](docs/development-direction.md).

Use Python 3.13, run unittest discovery, Ruff lint/format and a fresh synthetic
prototype smoke run as described in AGENTS.md. Tests must generate temporary inputs
rather than depend on a personal recording or committed capture profile.
