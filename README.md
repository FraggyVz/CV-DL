# StratScout

StratScout is a CS2 Mirage research prototype: minimap pixels → player detection →
tracking. A time-aware tactic model (attacker tactics before the plant, defender
responses afterwards) will be built on the tracked positions later.
Demo files are development evidence; the vision pipeline takes video pixels.

## Current status

Ryan's prototype has been moved into the `stratscout/` Python package. Both notebooks
are thin, optional experiment interfaces importing that implementation. Importing
modules does not extract data, train models, open windows or save results.

The colour baseline, CNN detector, tracker, annotation tools and companion demo-ingest
utilities are retained. **The old tactic classifiers, trajectory features, inference
pipeline and occupancy/win-rate reports were removed on the `vision-only` branch**:
they stretched each round to 32 steps, so timing in seconds was lost. A new
time-aware model will replace them (old code: commit `1a63107`). **This is an
experimental vision pipeline, not validated on real matches.** Existing tracking,
label-safety and ingest-schema problems are recorded in the
[baseline review](docs/base-review-2026-10-12.md).

On the `bomb-state` branch, spectator-radar detection adds C4 badge detection and
carried/dropped candidates, estimated rounds, overlapping-icon splitting with a five-per-team
cap, label-free count checks and an interactive preview. Results are proxies and visual
reviews, not labelled accuracy; see [bomb detection](docs/bomb-detection.md) and
[overlapping icons](docs/icon-split.md).

Independent recording-preparation commands now export aligned **5, 10 and 30 fps**
videos with validated manifests and visual comparison reports. Recording is manual
in OBS; the Python tools process saved videos. See the
[validation status and acceptance checks](docs/processing-validation.md).

## Windows setup

Use Python 3.13. The refactor was checked on Python 3.13.14 with CPU Torch.
Create a new environment if an existing one points to a moved interpreter:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The Python 3.13 launcher entry must exist; otherwise run its executable directly.
During this refactor, the working environment was created at `.venv-modular/`;
activate it with `.\.venv-modular\Scripts\Activate.ps1` to use it locally.
The requirements include the companion's pandas/demoparser2 dependencies.
For notebooks, select this environment in your editor; install `ipykernel` there
if your editor requests it. For linting, install `requirements-dev.txt`.

Video preparation also requires **FFmpeg and ffprobe** with `libx264` on PATH.
Check `ffmpeg -version` and `ffprobe -version` in the same terminal.
FFmpeg is a separate executable dependency. The new
commands do not import Torch or the legacy pipeline.

## Record once, prepare three frame rates

**Normal workflow:** put finished recordings directly in `data/raw/inbox/`, then run:

```powershell
.\process-recordings.ps1
```

For each new recording, answer three terminal prompts: **demo playback speed,
start time and end time**. For example, enter `2`, `0:00` and `23:51` for a 2×
recording whose useful footage ends at raw-video time 23:51. Press Enter to accept
2× speed, start at zero or keep the remainder. Times refer to the raw video
player, not the demo clock. Type `s` to skip a file or Ctrl+C to stop.

There are no per-video settings files to create or edit. The processor restores
normal game time before creating 5/10/30 fps versions; 60 fps alone does not imply
2× playback. The batch defaults to 2× because the current capture setup is
**2560×1440, 60 fps, constant 2× demo playback**. See the
[speed and trimming guide](docs/recording-setup.md#prompt-based-processing).
Old recordings, exports, capture-specific scripts/profiles and unlabelled pilot
caches were removed on 14 October. The new recording needs its own crop and validation.

The launcher uses `.venv-modular` or `.venv`. Alternatively, with your environment
active, run `python -m stratscout process-recordings`. The first run opens a crop
selector and saves `configs/recording-video.json`. Select the entire minimap;
odd dimensions are adjusted by one pixel without resizing. Confirm both selection
and preview with Enter. Later runs
reuse that crop for recordings with the same resolution **and radar layout**.

The batch scans MKV/MP4/MOV/AVI files directly in the inbox, encodes and validates
5/10/30 fps minimap videos, and generates comparison previews, stills and an HTML
report. Outputs go under `data/processed/recordings/<match-name>/`, using a cleaned
version of the source filename. Full hashes in the manifests still identify duplicates;
a short hash suffix distinguishes different sources with the same name. The command
prints each report path. It leaves masters untouched, skips completed bundles and
continues past failed files. Run it again after adding recordings; it is not a
background watcher. Wait for OBS to stop and finish saving before running it.

Reports sample three spread-out windows of up to ten seconds (one for short clips).
They do not automatically find rounds, deaths or bomb events. Crop selection and
visual review are still human tasks; encoding/export/reporting are automatic.

Use `--input FOLDER`, `--profile FILE`, `--output FOLDER` or `--at SECONDS` to override
defaults. Changing resolution/HUD layout requires a new profile; use a new output
root to retry incomplete bundles or change their crop, speed or trim. Completed
bundles retain their saved settings and are skipped before any prompts. For unattended
whole-file processing, use `--non-interactive --playback-speed 2` with an existing
crop profile. See
[recovery and identity rules](docs/recording-setup.md#outputs-and-recovery).

For explicit control of individual steps, the original commands remain available:

```powershell
python -m stratscout video-profile --video data/raw/inbox/match.mkv --output configs/recording-video.json --at 60
python -m stratscout prepare-video --video data/raw/inbox/match.mkv --profile configs/recording-video.json --playback-speed 2 --fps 5 10 30 --match-id mirage-match-001 --output data/processed/mirage-001
python -m stratscout compare-fps --input data/processed/mirage-001 --windows 55:5 115:5 190:5
```

Select a rectangle and confirm with Enter, then inspect the adjusted native crop
and press Enter to save (Escape cancels). `--at` selects the preview time; `--crop
x0 y0 x1 y1` bypasses the GUI for known coordinates and still requires even dimensions.
Profiles store source dimensions
and half-open bounds; a mismatched resolution is refused. No recording-specific
profile is bundled. The first selection creates the local profile automatically.

`prepare-video --start 115 --duration 10` makes a short pilot with its source offset
retained. Both values always use **raw-video seconds**. Add `--playback-speed 2`
to turn those ten recorded seconds into twenty seconds at normal game speed.
Choose a new destination each time. Outputs include three H.264 MP4s,
`manifest.json`, encoding/decode logs and, after comparison, `comparison/report.html`,
JSON, labelled videos and PNG stills. Comparison windows use **source-video seconds**.
File sizes and decoding timings do not measure detection accuracy.

For future recordings, follow [manual OBS and radar setup](docs/recording-setup.md).
The current captures use **2560×1440 output, 60 fps, constant 2× demo playback**.
Use NVENC H.264, MKV and High Quality/Medium File Size. Keep the resolution and radar
layout fixed across new recordings.
Capture FPS divided by replay speed must meet the highest output rate. This checks
the file's nominal rate, not unique rendered frames: if CS2 renders below 60 fps at
2×, repeated OBS frames reduce the actual observations. No WebSocket setup is needed.
Check a short recording and its
crop before capturing a full match; OBS/game settings are not validated by the CLI.

## Run and check

From the repository root:

```powershell
python -m stratscout --help
python -m stratscout prototype --smoke --output runs/smoke-001
python -m unittest discover -s tests
```

The smoke command runs a small synthetic detector/tracker experiment on CPU and
writes `detector.pt` and `summary.json`. It does not read your real
recordings or annotations. **Its scores are wiring checks, not accuracy evidence.**
Choose a new output directory for every run; existing destinations are refused.

For the full prototype, use a JSON object containing `Config` fields:

```json
{
  "data_dir": "data/raw/inbox",
  "video_exts": [".mkv", ".mp4"],
  "map_image": "mirage_radar.png",
  "det_epochs": 25
}
```

Save it as `configs/local.json`, then run:

```powershell
python -m stratscout prototype --config configs/local.json --output runs/experiment-001
```

Add `--plots` for interactive figures or `--device cuda` for a suitable GPU Torch
installation. The full workflow can extract/cache pilot frames and crop metadata;
read the annotation hazards below before using reviewed data. Video discovery scans
the configured directory directly, not recursively. One file represents one match.
The prototype does not consume the preparation manifest or correct 2× raw timing
automatically; this example selects inputs only. Establish normal-time samples and
reviewed labels before using the new capture for model training.

## Bomb badge scan and live preview

The C4 badge is found by colour then shape (badge-orange pixels forming a filled
rectangle). It is drawn over its carrier, so a badge is carried when the carrier's yellow
rim or white arrow shows at its edge. Carried and dropped badges look identical, so a still
badge with no carrier showing for 2 s is listed as a dropped candidate for review:

```powershell
python -m stratscout bomb-scan --video data/processed/recordings/<match>/minimap-5fps.mp4 --output runs/bomb-scan/<match>
```

`bomb-preview` (pick a match, or `--match navi --round 7`) plays a video with live player
and bomb detections and estimated rounds: Space play/pause, Right/Left arrows
fast-forward/rewind, K/L ∓5 s, N/B one frame, `[`/`]` previous/next round, Q quits.
Candidates are unreviewed, not labels. See [bomb detection](docs/bomb-detection.md).

## Overlapping players

Overlapping icons of one team are split by disc fitting, ✕ death markers (no white arrow)
are ignored and each team is capped at five. Compare versions without labels, fitting on
two matches and testing on the third:

```powershell
python -m stratscout icon-count-eval --train furia spirit --test navi --output runs/icon-count-eval
```

See [overlapping icons](docs/icon-split.md) for the method, results and trade-offs.

## Notebooks and demo ingest

- `stratscout_prototype.ipynb`: inspect each vision/training/report step.
- `stratscout_dem_ingest.ipynb`: radar-video exports and optional demo parsing/mapping.
- `python -m stratscout ingest --stage radar --config configs/ingest.json`: explicit
  radar export using a `DemCfg` JSON object and existing `crops.json`.
- Other ingest stages are `parse`, `map` and `export`; export requires reviewed demo
  labels. **The companion's legacy label schema is incompatible with the main
  two-head schema.** Its outputs are not automatically wired into training.

The companion's legacy exporter still defaults to 8 fps and retains its old API and
overwrite semantics. Use **`prepare-video` for new recording preparation**. Its
timestamped variants are independent of the prototype's 32-frames-per-round sampler;
they are not automatically a training dataset.

## Development and limitations

See [module responsibilities and API examples](docs/architecture.md),
[the recording direction](docs/development-direction.md) and
[contributor rules](AGENTS.md).

- Tracker placeholders/coasting are unchanged, and its gate/miss limits are counted
  per step rather than per second.
- Legacy annotation proposal/save behaviour still needs repair. Back up reviewed
  labels; never direct `auto_label_icons` at a reviewed annotation file.
- Icon splitting is evaluated only with label-free proxies and visual review; it
  under-counts spawn stacks of five. Labelled live-play frames are still needed.
- Recording and CS2 playback are manual; no OBS control command is included.
- Lower frame rates do not fix bomb-carrier occlusion; labelled accuracy comes later.
- Old capture assets were explicitly cleared; reusable models, map assets and
  regression fixtures remain. New capture validation and labelled evaluation are pending.
