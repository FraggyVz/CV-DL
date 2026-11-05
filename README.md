# StratScout

StratScout analyses CS2 Mirage minimap videos: player detection → tracking → tactical
classification. The current package implements the vision and recording-preparation
work; the five-class tactical model and its updated labelling workflow are next.
Inference uses video pixels. Demo files may support annotation and evaluation, but
are not required runtime input.

## Start here

- **This README:** setup, commands, structure, current limitations and next steps.
- **[Recording guide](docs/recording-setup.md):** OBS/radar settings, trimming,
  2× correction, cropping, 5/10/30 fps exports and sharing.
- **[Tactic guide](docs/tactic-definitions.md):** class definitions, map regions,
  labelling workflow, segment boundaries and annotation review.
- **[Labelling guide](docs/labelling-guide.md):** getting the videos, the labelling
  window's keys, marking segments reviewed and exporting clips.
- **[Contributor rules](AGENTS.md):** coding conventions, checks and data protection.

## Current status

The `vision-only` base contains the colour baseline, CNN icon detector, icon splitting,
tracker, bomb badge detection, estimated rounds, interactive preview and companion demo
utilities. Processing exports aligned 5/10/30 fps minimap videos with manifests and
comparison reports. Recording is manual; there is no OBS control or replay automation.

**This is not yet a validated tactical classifier.** The old classifiers were removed
because their fixed 32-step round sampling lost elapsed timing. That implementation is
available at commit `1a63107`. Detection results include visual reviews and label-free
count proxies, not a measured precision/recall result on a reviewed dataset.

The agreed tactic classes are `A_DIRECT`, `B_DIRECT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`.
Direct A includes ramp and/or palace; an A split adds coordinated mid → connector
participation. Direct B uses apartments; a B split adds mid → short. Label attack and
retake segments separately. Rush speed and actual default openings are not extra classes.
The [tactic guide](docs/tactic-definitions.md) owns the detailed definitions.

**Labelling pilot.** `label-segments` labels five-class attack/retake segments with the
keyboard, one file per match and reviewer, saved on every change. `review-segments`
marks agreed segments `reviewed` and `export-segments` cuts them into train/test clip
folders split by whole match (see the [labelling guide](docs/labelling-guide.md)).
Three matches are labelled by one reviewer with a second reviewer's comments in
[labels/review-notes.md](labels/review-notes.md): 44 reviewed training clips
(Furia, Spirit) and 24 test clips (NaVi). No tactic model uses them yet.
Approach/entry-time fields, reviewer comparison and region polygons remain planned.
`contracts.Round` and `RoundLabeller` keep
the old pre/post-plant labels, as did the earlier `label-rounds` command. Do not rename
old annotations automatically or use the legacy keys as the new class mapping.

## Windows setup

Use Python 3.13 and FFmpeg/ffprobe with `libx264` on PATH. The refactor was checked with
Python 3.13.14 and CPU Torch. From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
ffmpeg -version
ffprobe -version
```

A working environment may already exist at `.venv-modular`; use its Python instead.
Commands below use `python` from the chosen environment. If PowerShell blocks activation
or `.ps1` launchers, invoke `.\.venv\Scripts\python.exe` (or `.venv-modular`) directly;
changing the machine execution policy is unnecessary. Select the same interpreter in
an editor. Development dependencies provide Ruff; the main requirements include the
companion's pandas/demoparser2 dependencies.

## Where files go

Large files are shared outside GitHub. Put them in these folders, which Git already
ignores, so every machine has the same layout and the commands find them by default:

```text
data/
    raw/inbox/                     new OBS recordings (2× masters) waiting to be processed
    processed/
        recordings/<match>/        prepared match videos: minimap-5/10/30fps.mp4, manifest.json
        datasets/<export date>/    labelled clip datasets
            export.json            counts, splits and label-file hashes for this export
            train/  manifest.csv, clips/<class>/*.mp4
            test/   manifest.csv, clips/<class>/*.mp4
labels/                            segment label files (*.segments.json): committed to Git
configs/                           local settings, e.g. recording-video.json (not committed)
runs/                              experiment, scan and smoke-test outputs (not committed)
```

- **Your own recording:** see [Process a recording](#process-a-recording). The batch
  reads `data/raw/inbox/` and writes `data/processed/recordings/<match>/`.
- **Shared match videos:** extract each match ZIP into `data/processed/recordings/` so
  each match is one folder with the videos directly inside it; see the
  [labelling guide](docs/labelling-guide.md#get-the-videos-do-this-once). Keep
  `manifest.json`: labels and exports check the match ID it contains.
- **Shared clip dataset:** extract it into a dated folder such as
  `data/processed/datasets/2026-11-01/`, with `export.json`, `train/` and `test/`
  directly inside. If train and test arrive as separate ZIPs carrying the same
  `export.json`, keep one copy at the top. Make a new dated folder for each export;
  never mix clips from two exports.
- Do not commit videos, datasets or checkpoints. Only `labels/` and code are shared
  through Git.

## Process a recording

Record at **2560×1440, 60 fps, constant 2× demo playback** with the fixed radar layout
in the [recording guide](docs/recording-setup.md). Wait for OBS to finish, put the file
in `data/raw/inbox/`, then run:

```powershell
python -m stratscout process-recordings
```

The optional `process-recordings.ps1` launcher selects a local project environment.
The batch prompts for playback speed, start and end in **raw-video time**. First use
selects a crop and saves `configs/recording-video.json`; matching layouts reuse it.
Outputs appear in `data/processed/recordings/<match-name>/`: three normal-speed MP4s,
`manifest.json` and `comparison/report.html`. Masters are preserved and completed
bundles are skipped. Keep the manifest with the videos; use a new output directory to
change settings or retry an incomplete bundle. Advanced commands, recovery, acceptance
checks and sharing instructions belong in the recording guide. Prepared videos are
not labelled training data.

## Preview and experiments

```powershell
python -m stratscout --help
python -m stratscout bomb-preview
python -m stratscout bomb-preview --match navi --round 7
python -m stratscout label-segments --match furia --reviewer <name>
python -m stratscout bomb-scan --video data/processed/recordings/<match>/minimap-5fps.mp4 --output runs/bomb-scan-001
python -m stratscout icon-count-eval --train furia spirit --test navi --output runs/icon-count-eval-001
python -m stratscout prototype --smoke --output runs/smoke-001
```

Replace `<match>` with a processed folder name. New experiment/scan destinations must
not already exist. The preview uses Space to play/pause, arrows to seek, K/L for ±5 s,
N/B for one frame, `[`/`]` for estimated rounds and Q to quit. It shows player/bomb
estimates; the hidden carrier is inferred, not directly observed. A stationary bomb
without carrier evidence is a dropped candidate; planted bombs are not separated.

`icon-count-eval` checks counts without reviewed labels; it does not establish detection
accuracy. Fit detector settings on Furia–Falcons and Spirit–Falcons and keep NaVi–Falcons
held out. Do not tune on the test match. Keep every FPS variant and tactical segment of
a match in one dataset split, and reserve validation data separately for model selection.

The smoke command trains a tiny synthetic detector and checks tracking, saving
`detector.pt` and `summary.json`. It does not use real recordings or establish accuracy.
For an explicit vision experiment, pass `prototype --config configs/local.json --output
runs/experiment-001`; the JSON contains `Config` fields such as `data_dir`, `video_exts`,
`map_image` and `det_epochs`. Add `--device cuda` for a compatible GPU Torch installation,
or `--plots` for interactive figures. The prototype scans one directory directly and
can write pilot caches; it does not automatically use preparation manifests or correct
raw 2× timing. Establish prepared inputs and safe labels before training on real data.

Companion demo stages remain available via `ingest --stage radar|parse|map|export
--config configs/ingest.json` (choose one stage). Their legacy 8 fps exporter and label
schema are not the current video-preparation or five-class dataset route.

## Project structure and contracts

| Location/modules | Responsibility |
|---|---|
| `stratscout/media.py`, `video_profiles`, `video_preparation`, `recording_batch`, `fps_comparison` | Shared preparation and comparison APIs |
| `segmentation`, `icon_split`, `bomb`, `rounds`, `live_preview`, `count_eval` | Colour/shape detection, overlap handling, estimates and review |
| `segments`, `segment_labeller` | Five-class segment records, validation, safe saving and the keyboard labeller |
| `preprocessing`, `detector`, `datasets`, `training`, `evaluation`, `synthetic` | CNN icon detector and experiments |
| `tracking`, `annotations`, `config`, `contracts`, `context`, `workflow`, `storage`, `video`, `pilot`, `visualisation` | Tracking, legacy labels, explicit run state and orchestration |
| `stratscout/ingest/` | Optional demo parsing, coordinate mapping and legacy exports |
| `tests/` | Regression, import-safety, media and synthetic workflow checks |
| `configs/`, `data/`, `runs/` | Local settings, recordings, clip datasets and generated experiments; see [Where files go](#where-files-go) |
| `mirage_radar.png` | Map asset; recordings, generated results and environments stay outside Git |

Implementation lives in Python modules, not notebooks. Pass `RunContext` or
`IngestContext` explicitly; imports must not start processing, training, GUIs or writes.
OpenCV returns BGR; detector tensors use RGB. Preserve uint8 input where preprocessing
divides by 255, T/CT heatmap channel order (0/1) and positive centre peaks. Keep encoding
in the shared preparation API. Record observed/missing/inferred positions distinctly;
never turn a team-size cap into invented detections. Demo truth and round winners are
annotation/evaluation evidence, not vision-model inputs.

## Known limitations

- **Label safety:** legacy proposal/review writes can overwrite labels; empty icon
  frames are dropped. Explicit review status, uncertain centres and safe migration
  are not implemented. Never target reviewed labels with automatic proposals.
- **Tracking:** fixed slots can invent or retain stale positions; gates/miss limits are
  per step rather than per second. Missing-player handling needs repair before relying
  on trajectories for classification.
- **Detection:** spawn stacks can be under-counted; overlaps, carriers and death markers
  need labelled evaluation. Five-per-team counts and visual reviews are not precision/recall.
  Planted-bomb separation and round boundaries remain unverified estimates.
- **Data integration:** legacy ingest/round schemas differ from the target segment format.
  World-to-radar mapping and demo/video alignment require calibration. Real icon data
  splitting must enforce match separation. Removed classifier results and historical
  synthetic scores are not current performance evidence.
- **Capture:** nominal FPS cannot prove unique rendered frames; pauses, stalls or replay
  speed changes break the constant-speed assumption. Check each new layout and recording.

## Next steps

1. **Agree labels:** the pilot labels were reviewed by comment rather than by two
   independent label files. Resolve the open items in
   [labels/review-notes.md](labels/review-notes.md), then freeze a definition version.
2. **Finish the labelling tool:** add reviewer comparison and approach/entry-time
   fields; refuse approval of segments with uncertain boundaries. Preserve original
   records. Define map regions once when adding route features.
3. **Grow the dataset:** label more matches (class coverage is thin, e.g. one
   `A_SPLIT` in the test match), keep match IDs and provenance, and choose validation
   by whole match from the training matches before tuning.
4. **Validate vision/tracking:** measure per-team and carrier performance and repair
   missing-player handling. Compare 5/10/30 fps on labelled evidence.
5. **Train tactics:** compare a region/route baseline with a learned temporal model using
   elapsed seconds. Report per-class precision/recall, macro-F1, confusion matrices and
   failures on unseen matches. Manual clip boundaries do not prove whole-match detection.
6. **Deliver a usable application:** show detections and tactical segments, make local
   deployment reproducible, evaluate optimisation/cloud deployment where feasible and
   document limits. Other maps and default-opening/tempo attributes are later extensions.

## Development checks and historical references

Follow [AGENTS.md](AGENTS.md). From the root with the chosen environment:

```powershell
python -m unittest discover -s tests
ruff check stratscout tests
ruff format --check stratscout tests
python -m stratscout prototype --smoke --output runs/new-smoke-run
```

Choose a fresh smoke directory. Synthetic checks do not establish real-match accuracy.
Historical material is retained in `docs/archive/` for investigations and report evidence,
not routine onboarding. Statements there apply to the recorded revision:

- [Original baseline audit](docs/archive/base-review-2026-10-12.md)
- [Architecture and API notes](docs/archive/architecture.md)
- [Previous development plan](docs/archive/development-direction.md)
- [Bomb-detection experiments](docs/archive/bomb-detection.md)
- [Icon-splitting experiments](docs/archive/icon-split.md)
- [Processing validation history](docs/archive/processing-validation.md)
- [Module migration inventory](docs/archive/module-inventory.json), retained for regression checks
- [8 October progress deck](docs/archive/StratScout%20progress.pptx), which predates the
  five-class definitions and reports synthetic-only results
