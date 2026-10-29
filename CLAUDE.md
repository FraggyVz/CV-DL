# StratScout — Project Handoff

## Current direction

Use the shared modular prototype as the base. Implementation is in `stratscout/`;
the two notebooks are thin experiment interfaces. Read [AGENTS.md](AGENTS.md),
[README](README.md), [architecture](docs/architecture.md) and the
[baseline review](docs/base-review-2026-10-12.md) before changing behaviour.

**Vision-only scope (`vision-only` branch, 29 October 2026).** The old tactic
classifiers, trajectory features, inference pipeline and occupancy/win-rate reports
were removed. They stretched each round to 32 steps between start and plant, losing
timing in seconds, so they could not separate a rush from a default. The package now
covers capture processing, the radar crop, the colour baseline, icon splitting, bomb
detection, the CNN icon detector, the tracker, annotation tools, the demo ingest and
the live preview. A new time-aware tactic model will be built separately; the removed
code is available at commit `1a63107`.

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

- Round-label classes (used by `RoundLabeller`; no model consumes them yet):
  before plant `A_RUSH`, `B_RUSH`, `A_SPLIT`, `B_SPLIT`, `DEFAULT`; after plant
  `RETAKE`, `TRICKLE`, `SAVE`.
- A future tactic model must sample positions on a fixed clock in seconds (not a fixed
  step count per round) and express tracker gates per second.
- The CNN heatmaps use channel 0 for T and channel 1 for CT.
- OpenCV decoding returns BGR; vision tensors use RGB. Keep input frames uint8
  where preprocessing performs its own division by 255.
- Centre heatmaps must contain positive peaks; preserve the integer-peak handling.
- Retain the colour detector (`segmentation.find_icons_teams`) as a baseline. Numbers,
  objective graphics, death markers and overlapping icons need explicit labelled evaluation.
- Match split for spectator-radar detection work: **fit on Furia–Falcons and
  Spirit–Falcons, test on NaVi–Falcons.** Never tune thresholds on the held-out match.
- Do not invent invisible players when annotating overlaps or bomb occlusion.
- Keep splits grouped by match, including all FPS variants and demo-derived samples.
- Round winner is an annotation/reporting field, never a vision-model input.

## Spectator-radar detection (`bomb-state` branch)

Three processed matches (Gin's ZIPs, 494×500 native crops, same crop) are used:
`furia-falcons-mirage-cologne-major-2026`, `spirit-falcons-mirage-cologne-major` and
`navi-falcons-mirage-cologne-major` under `data/processed/recordings/` (git-ignored).

What the spectator radar shows (measured, OpenCV hue 0–180):

- T icons are yellow numbered discs (hue ≈ 27), CT icons cyan (≈ 94); every living icon
  has a **white direction arrow** attached. Icon blobs are ≈ 246 px².
- The C4 badge is an orange rectangle (≈ 16) drawn **on top of the carrier's icon**; only a
  thin yellow rim and the carrier's arrow show at its edge. Yellow icons beside it are
  teammates. Carried and dropped badges look identical.
- ✕ death markers appear in the team colour for ≈ 2 s and have no arrow. The yellow A/B
  site letters are static map art (hue 23–24). A planted badge on site appears to gain a
  blinking red outline (unconfirmed). A white ⊤ mark sometimes sits above a lone badge
  (meaning unconfirmed).

Implemented (see [bomb detection](docs/bomb-detection.md), [overlapping icons](docs/icon-split.md)):

- `bomb.find_bomb`: badge-orange mask, 3×3 close + open, then the shape rule
  (`segmentation.blob_shape`/`looks_like_bomb`: aspect, extent, corner fill). Found in
  94–96% of frames versus 26–32% for shape alone (`bomb-scan --shape-only`).
- `bomb.carrier_rim`: carried when ≥ 4 T-yellow or white pixels touch the badge, excluding
  background-matching yellow (site letters) and arrows attached to cyan icons.
  `BombStateTracker`/`dropped_candidates`: still ≥ 2 s with no carrier = dropped candidate.
  Planted bombs are currently listed as dropped.
- `rounds`: round starts from freeze-time spawn occupancy, cached in `runs/round-cache/`.
  All three matches give 20–21 rounds with halftime before round 13.
- `icon_split.find_icons_split` (used by the preview): per-team masks, arrowless ✕ removal,
  disc-fit icon counts per blob, five per team (four with a hidden carrier).
- `count_eval` / `icon-count-eval`: label-free checks (all-alive frames showing 5; counts
  above five). These are proxies, not labelled accuracy.
- `live_preview` / `bomb-preview`: interactive window with match picker, estimated
  rounds and the hidden carrier drawn as an inferred T. GUI code stays in this module.

Results on held-out NaVi: the old detector shows 5 per team in 8.9% of all-alive frames;
area splitting + cap 96.6% but over-counts small groups and ✕ markers in live play; the
disc fit 53.4%, correct on all reviewed live-play frames but under-counts stacks of five
in spawn as four. Neither proxy measures late-round over-counting; labelled frames do.

## Unresolved problems

The baseline review documents annotation overwrite/empty-frame loss, missing-player
tracking, incompatible ingest labels and evaluation leakage (its classifier items
refer to removed code). Synthetic detector scores are not real-match accuracy.

The retained fixed-slot tracker can invent or retain stale positions. It is not
a validated solution to occlusion. Its `track_gate` (0.09 per step) and
`track_max_miss` (8 steps) are counted in steps, so they only suit short step
intervals; repair observed/missing state and express these limits per second.
Avoid blanket masks over playable map areas.

The companion demo-ingest utilities remain reusable research code, separate from
the recording processor. Their legacy 8 fps exporter is not the current preparation
route. World-to-radar mapping and demo/video synchronisation require calibration
for the new crop; previous fitted offsets must not be assumed valid.

## Next work and checks

For detection: hand-label a small set of live-play frames (team counts, then centres) to
get real precision/recall and settle the disc-fit versus area trade-off; recover spawn
stacks and fully hidden players with tracking; add a planted state (red outline); obtain
the matches' `.dem` files to confirm bomb events, rounds and positions.

Finish and validate the new recording, then protect labels, establish shared match
identities and normal-time samples, and measure detector/tracker behaviour before
choosing a training frame rate. Follow [development direction](docs/development-direction.md).

Use Python 3.13, run unittest discovery, Ruff lint/format and a fresh synthetic
prototype smoke run as described in AGENTS.md. Tests must generate temporary inputs
rather than depend on a personal recording or committed capture profile.
