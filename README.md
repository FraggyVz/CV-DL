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

### The pipeline

```
recording ─► prepare (crop radar, fix 2× speed, 5/10/30 fps) ─► label + review ─► export clips by whole match
          ─► detector (players, bomb) + ResNet-18 features ─► shared clip data ─► strategy models ─► tactic
```

| Part | State | Who | Where |
|---|---|---|---|
| Recording preparation | Working; recording itself is manual | Gin, jx | `process-recordings`, [recording guide](docs/recording-setup.md) |
| Labelling and export | Working | Gin, jx | `label-segments`, `review-segments`, `export-segments` |
| Player and bomb detector | Working: colour and shape rules, no training. Splits overlapping icons, drops death markers, finds the C4 and its carrier. | zhengyongheng1, on Gin's colour baseline | `live_preview.detect`, `icon_split`, `bomb` |
| Detector check | Working: measured against demo positions on the training matches | jx | `detector-check`, [doc](docs/detector-check.md) |
| ResNet-18 frame features | Working: frozen ImageNet weights | jx | `tactic_model`, [tactic baseline](docs/tactic-baseline.md) |
| Shared clip data | Working: detector channels per named map region, bomb state and plant estimate, ResNet features on one timeline. Draft regions (`mirage-spectator-v1`). | ri3n | `build-clip-data`, [doc](docs/clip-data.md) |
| TCN strategy model | Working, with cross-validation and a scored test set | ri3n | `train-tactics`, `predict-tcn`, [Tactic model](#tactic-model) |
| Tactic baseline (ResNet mean + linear) | Working, development scores only | jx | `tactic-baseline` |
| Temporal models (temporal average pooling, GRU) | Working, development scores only | jx | `tactic-temporal`, [doc](docs/tactic-temporal.md) |
| CNN icon detector | Exists; not used by the strategy models | Gin | `detector`, `training` |
| Web app | MP4 round upload, whole-clip review, match-wide analysis, TCN/GRU predictions, removal and reset | violetdisp2408 | [Run the web application](#run-the-web-application) |

### Data

The current dataset is **`data/processed/datasets/2026-11-16`**: six matches, all
segments reviewed. Training has 82 clips from Furia–Falcons, Spirit–Falcons,
Aurora–Vitality and MOUZ–BetBoom; test has 45 clips from NaVi–Falcons and
Vitality–Falcons. One clip, Spirit #5, has an uncertain boundary and is excluded
with `--exclude-uncertain`. It contains every clip of the older `2026-11-01` export,
so don't combine the two. Open label questions are in
[labels/review-notes.md](labels/review-notes.md).

### Results so far (2026-11-16, macro-F1)

Cross-validation on the 82 training clips (Spirit #5 excluded): each of the four training
matches is held out in turn, seeds 0–2. The main column is **nested epoch selection**: the
epoch is chosen from the three training matches only, so the scored match never helps
choose it. Values are the mean over the four held-out matches and three seeds, ± the seed
SD. The TCN's **other-folds rule** (each match scored at the epoch best on the other
matches, whose models were trained on it) is shown for reference only; it scored most
models higher.

| Model | Input | Nested macro-F1 (main) | Other-folds | Test (scored once) | Details |
|---|---|---|---|---|---|
| Mean pooling | ResNet images | 0.518 ± 0.034 | 0.584 | not run | [tactic baseline](docs/tactic-baseline.md) |
| Temporal average pooling (3 segments) | ResNet images | 0.620 ± 0.015 | 0.650 | not run | [temporal models](docs/tactic-temporal.md) |
| GRU | ResNet images | 0.600 ± 0.028 | 0.681 | not run | [temporal models](docs/tactic-temporal.md#the-gru) |
| TCN | Detector features | not yet run (slow) | 0.679 | 0.919 | [Tactic model](#tactic-model) |
| Pooled classifier | Detector features | not yet run (slow) | 0.645 | 0.862 | [Tactic model](#tactic-model) |
| Mean pooling | Clip data | 0.796 ± 0.016 | 0.790 | not run | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| Temporal average pooling (3 segments) | Clip data | 0.763 ± 0.048 | 0.775 | not run | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| GRU | Clip data | 0.792 ± 0.033 | 0.809 | [0.890 (ensemble 0.963)](notebooks/tactic-final-test.ipynb) | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| **TCN** | Clip data | **0.849 ± 0.021** | 0.834 | [**0.897** (ensemble 0.927)](notebooks/tactic-final-test.ipynb) | [shared clip data](docs/clip-data.md) |
| Pooled classifier | Clip data | 0.809 ± 0.035 | 0.823 | not reported | [shared clip data](docs/clip-data.md) |

Test results for the clip-data TCN and GRU are in `notebooks/tactic-final-test.ipynb`
(macro-F1 pooled over the 45 test clips, mean over three seeds, with the seed ensemble in
brackets); the TCN was selected by nested cross-validation before the GRU was scored.

The pooled classifier is the TCN's control: the same rows pooled over the clip before a
small network, so step order cannot matter. The detector-features test scores come from
the earlier run (cross-validation 0.667 for the TCN and 0.625 for the pooled classifier);
the other-folds column is the 2026-11-19 rerun (`runs/tactic-tcn-repro-20261119`). Clip data (ri3n's `build-clip-data`) is detector
channels per map region plus 32 ResNet PCA numbers fitted per fold; image runs use 16
sampled frames per clip. The nested TCN and pooled-classifier rows come from
`scripts/nested_tcn.py`; the other-folds clip-data rows from a train-only rerun of
`train-tactics`. Every value and the per-match, per-class and error breakdowns are computed
in `notebooks/tactic-model-comparison.ipynb`. The notebook is saved with its outputs. The
result files it reads are tracked in Git, so a clone can rerun it with the same scores.
Figures 2.2 and 2.3 and a few review and input checks also need the train-only dataset
and clip-data copies; without them they show "—" or "unverified".

- Training choices on ResNet images (nested): the Last-state GRU scores 0.345, the Compact
  last-state GRU 0.327 and the Compact pooled GRU 0.600; temporal average pooling without class weights 0.564,
  against 0.620 with them ([details](docs/tactic-temporal.md#the-gru)).
- Majority class: 0.065 (test 0.047).
- Random-segment average pooling + linear classifier (control): the same three segments with the frames assigned at random, 0.526 (nested);
  evidence that frame order helps on these development matches
  ([details](docs/tactic-temporal.md#the-random-segment-control)).
- Position features: the ResNet-row models on detector positions (8×8 grid) instead of
  image features: mean 0.595, temporal average pooling 0.557, first GRU 0.384 (nested)
  ([details](docs/tactic-temporal.md#position-features)).
- The clip data's map regions were adjusted after looking at training labels, so the
  clip-data scores may be optimistic.

Report cross-validation as the estimate for new matches. The test score is optimistic:
the test teams also appear in training, and some classes have only 5–6 test clips. Most
remaining errors are direct vs split on the same site. Compare models only on the same
dataset, exclusions, folds and seeds.

### Known gaps

- Clips are cut by hand. Nothing finds tactic segments in a full match yet.
- The web app still requires manually marked attack/retake boundaries; it does not find
  tactics automatically in a full match.
- The detector is rule-based and imperfect: stacked or hidden icons are missed. There is
  no reliable player identity (tracker).
- Map regions are a draft and need the team's review.
- Mirage and the spectator radar layout only.

The detailed class rules live in the [tactic guide](docs/tactic-definitions.md).
`contracts.Round`, `RoundLabeller` and the old `label-rounds` command keep legacy
pre/post-plant labels: don't use them as the five-class mapping.

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

## Run the web application

Install Docker Desktop, then from the repository root run:

```powershell
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The service binds to localhost;
it is not exposed to the network. The first build downloads the pinned frontend and
Python web dependencies. Stop it with Ctrl+C, or run `docker compose down`.

Create a match, choose whether the uploaded tactic clips are attacks or retakes, then
upload one MP4 per round (or select several to upload them sequentially). Each clip is
assigned the next round number and treated as one complete tactic interval. Select a
round and choose **Analyse Round** to prepare the entire clip (first frame to last), run
the player/bomb detector, and compare the saved TCN and GRU predictions. **Analyse all
rounds** starts the pending analyses together; the classification panel shows results for
every round in the selected match. Playback speed is shared, while each round's
prepared-clip setting is saved with that recording and used independently. Previous round
results stay in that match's round list when later clips are uploaded. Use **Remove round**
to delete a mistaken upload and its analysis;
later rounds in that match are renumbered. Use **New match** to start a separate sequence;
existing matches and recordings remain saved. Empty match workspaces and the current
selection are also saved in that browser, so a match remains selectable before its first
round is uploaded. Predictions below 0.6 confidence are flagged for review; they are not
ground truth.

Older recordings uploaded before match rounds were introduced remain unassigned and
available under **Previous recordings**. Existing saved segments are preserved; an older
recording with no segments defaults to its full duration. **Reset workspace** deletes
all app uploads and analyses across every match, including these previous recordings,
after confirmation. It does not delete source footage, labels, or checkpoints. Reset is
blocked while an analysis is running. The app does not discover round boundaries in a
full-match recording; upload one clip that already represents one round/tactic.

For an already-prepared minimap clip (494×500 at 10 fps), enable the prepared-clip option
in the round panel. For a full-screen recording, create the per-machine crop profile at
`configs/recording-video.json` using the [recording guide](docs/recording-setup.md)
before running analysis. The app defaults to the guide's 2× replay speed for full-screen
recordings; choose 1× if that recording was captured at normal speed. Prepared clips use
their source speed. New-round clips are analysed from start to finish.

Videos that the browser cannot decode get a separate H.264/yuv420p playback copy; the
uploaded original is preserved. Prepared analysis clips, manifests and result status are
also kept under `data/web/uploads/analysis/` for provenance. The Docker image includes
the CPU PyTorch/OpenCV inference stack, the tracked TCN/GRU checkpoints, FFmpeg and the
cached ImageNet ResNet weights. The first build is larger and takes longer because it
installs these packages and downloads the ResNet weights. Docker is not required for
the Python research commands elsewhere in this README.

Uploads are limited to 20 GiB by default. Set `STRATSCOUT_MAX_UPLOAD_BYTES` before
starting Compose to change the byte limit. Keep sufficient free disk space: videos are
staged during upload, then stored in the ignored `data/web/uploads/` folder. A
browser-compatible copy and analysis outputs need additional space. Uploads and generated
files are not deleted when the container stops. Files manually extracted into nested
dataset folders are not automatically registered as recordings; use the upload control
for a match video.

For frontend development outside Docker, use Python 3.13 and Node.js 22. In one
PowerShell window install the web dependencies and start the API:

```powershell
python -m pip install -r requirements-web.txt
python -m uvicorn stratscout.webapp:app --host 127.0.0.1 --port 8000
```

In another window, install the pinned frontend packages and start Vite:

```powershell
Set-Location frontend
npm ci
node .\node_modules\vite\bin\vite.js --host 127.0.0.1
```

Vite runs at `http://127.0.0.1:5173` and proxies API requests to the local Python
service. To build assets from PowerShell, run:

```powershell
node .\node_modules\typescript\bin\tsc --noEmit
if ($?) { node .\node_modules\vite\bin\vite.js build }
```

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
| `frontend/`, `stratscout/webapp.py`, `Dockerfile`, `compose.yaml` | Local-only web scaffold and container launch; analysis workflows are not connected yet |
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
