# StratScout

StratScout is a CS2 Mirage research prototype: minimap pixels → player detection →
tracking → attacker tactics before the plant and defender responses afterwards.
Demo files are development evidence; the vision pipeline takes video pixels.

## Current status

Ryan's prototype has been moved into the `stratscout/` Python package. Both notebooks
are thin, optional experiment interfaces importing that implementation. Importing
modules does not extract data, train models, open windows or save results.

The colour baseline, CNN detector, tracker, four classifier architectures, annotation
tools, reports and companion demo-ingest utilities are retained. **This is an
experimental pipeline, not a validated real-match classifier.** Classifier training
still uses generated rounds; real icon labels can supply the detector separately.
Existing tracking, label-safety and ingest-schema problems are recorded in the
[baseline review](docs/base-review-2026-10-12.md).

Independent recording-preparation commands now export aligned **5, 10 and 30 fps**
videos with validated manifests and visual comparison reports. Recording is manual
in OBS; the Python tools process saved videos. See the
[real-footage processing results](docs/processing-validation.md).

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

The launcher uses `.venv-modular` or `.venv`. Alternatively, with your environment
active, run `python -m stratscout process-recordings`. The first run opens a crop
selector and saves `configs/recording-video.json`. Select the entire minimap with
even width/height and confirm both selection and preview with Enter. Later runs
reuse that crop for recordings with the same resolution **and radar layout**.

The batch scans MKV/MP4/MOV/AVI files directly in the inbox, encodes and validates
5/10/30 fps minimap videos, and generates comparison previews, stills and an HTML
report. Outputs go under `data/processed/recordings/<source-hash>/`; the command
prints each report path. It leaves masters untouched, skips completed bundles and
continues past failed files. Run it again after adding recordings; it is not a
background watcher. Wait for OBS to stop and finish saving before running it.

Reports sample three spread-out windows of up to ten seconds (one for short clips).
They do not automatically find rounds, deaths or bomb events. Crop selection and
visual review are still human tasks; encoding/export/reporting are automatic.

Use `--input FOLDER`, `--profile FILE`, `--output FOLDER` or `--at SECONDS` to override
defaults. Changing resolution/HUD layout requires a new profile; use a new output
root to retry incomplete bundles or change their crop. See
[recovery and identity rules](docs/recording-setup.md#batch-processing-and-recovery).

For explicit control of individual steps, the original commands remain available:

```powershell
python -m stratscout video-profile --video data/raw/mirage.mkv --output configs/mirage-video.json --at 60
python -m stratscout prepare-video --video data/raw/mirage.mkv --profile configs/mirage-video.json --fps 5 10 30 --match-id mirage-match-001 --output data/processed/mirage-001
python -m stratscout compare-fps --input data/processed/mirage-001 --windows 55:10 115:10 190:25
```

Select an even-sized rectangle and confirm with Enter, then inspect the native crop
and press Enter to save (Escape cancels). `--at` selects the preview time; `--crop
x0 y0 x1 y1` bypasses the GUI for known coordinates. Profiles store source dimensions
and half-open bounds; a mismatched resolution is refused. The supplied Falcons–FURIA
profile is `configs/falcons-furia-video.json`, producing a native 492×498 crop.

`prepare-video --start 115 --duration 10` makes a short pilot with its source offset
retained. Choose a new destination each time. Outputs include three H.264 MP4s,
`manifest.json`, encoding/decode logs and, after comparison, `comparison/report.html`,
JSON, labelled videos and PNG stills. Comparison windows use **source-video seconds**.
File sizes and decoding timings do not measure detection accuracy.

For future recordings, follow [manual OBS and radar setup](docs/recording-setup.md).
Record a full-screen **1920×1080 output at 30 fps** using NVENC H.264, MKV and High
Quality/Medium File Size. CS2 can stay at 1440p with OBS downscaling to 1080p; use a
crop calibrated on the saved output. 1440p masters remain supported. Detection
accuracy at 1080p still needs evaluation. No WebSocket setup or password is required.
Check a short recording and its
crop before capturing a full match; OBS/game settings are not validated by the CLI.

## Run and check

From the repository root:

```powershell
python -m stratscout --help
python -m stratscout prototype --smoke --output runs/smoke-001
python -m unittest discover -s tests
```

The smoke command runs a small synthetic detector/classifier experiment on CPU and
writes `detector.pt`, `stratscout.pt` and `summary.json`. It does not read your real
recordings or annotations. **Its scores are wiring checks, not accuracy evidence.**
Choose a new output directory for every run; existing destinations are refused.

For the full prototype, use a JSON object containing `Config` fields:

```json
{
  "data_dir": "data/raw/falcons-vs-furia",
  "map_image": "mirage_radar.png",
  "det_epochs": 25,
  "cls_epochs": 40
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

- Tracker placeholders/coasting and feature handling of missing players are unchanged.
- Legacy annotation proposal/save behaviour still needs repair. Back up reviewed
  labels; never direct `auto_label_icons` at a reviewed annotation file.
- Real icon splitting still needs match-separated manifests.
- The leaking B-prime experiment is retained but disabled by default.
- Recording and CS2 playback are manual; no OBS control command is included.
- Lower frame rates do not fix bomb-carrier occlusion; labelled accuracy comes later.
- Real recordings, pilot data, existing annotations and checkpoints were preserved.
