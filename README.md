# StratScout

StratScout classifies CS2 Mirage tactics from spectator-radar videos. A clip of one
tactic is turned into player and bomb positions, then a model names the tactic: one of
`A_DIRECT`, `B_DIRECT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`. Inference uses video pixels only.
Demo files are used for evaluation, never as model input.

## Start here

**[Open the StratScout live demo](https://stratscout.streamlit.app/)**. Select
**Use sample clips**, add the Vitality–Falcons clip ending in `s006`, then select
**Analyse Round 1**. Open **TCN only** or **GRU only** to follow the clip from
player detections to its predicted tactic. The reviewed label for this example is
`B_SPLIT`. The three bundled samples come from the held-out test matches.

The cloud app accepts prepared radar clips. To analyse a full-screen recording of
one tactic locally, use the [Docker setup](#run-the-web-application). For the
dataset, checkpoints and scores, see [Reproduce the submitted results](#reproduce-the-submitted-results).

| I want to…                                   | Read                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Walk through StratScout in one notebook      | [Project showcase](notebooks/stratscout-showcase.ipynb) — a real clip, detector overlays, both model predictions and held-out results |
| Know what exists and what's next             | [Current status](#current-status) below                                                                            |
| Record and prepare a match                   | [Recording guide](docs/recording-setup.md)                                                                         |
| Label a match                                | [Labelling guide](docs/labelling-guide.md), with the class rules in the [tactic guide](docs/tactic-definitions.md) |
| Turn new labels into model-ready data        | [Data processing](docs/data-processing.md)                                                                         |
| Understand the shared model data             | [Shared clip data](docs/clip-data.md) and [its walkthrough notebook](notebooks/clip-data-walkthrough.ipynb)        |
| Understand how the TCN is trained and judged | [TCN walkthrough notebook](notebooks/tactic-tcn-walkthrough.ipynb) and [Tactic model](#tactic-model)               |
| See how accurate the detector is             | [Detector check](docs/detector-check.md)                                                                           |
| Work on the ResNet models                    | [Tactic baseline](docs/tactic-baseline.md), [clip loader](docs/clip-loader.md)                                     |
| Contribute code                              | [Contributor rules](AGENTS.md) and [Development checks](#development-checks-and-historical-references)             |

## Current status

### The pipeline

```
recording ─► prepare (crop radar, fix 2× speed, 5/10/30 fps) ─► label + review ─► export clips by whole match
          ─► detector (players, bomb) + ResNet-18 features ─► shared clip data ─► strategy models ─► tactic
```

| Part                                            | State                                                                                                                                                   | Who                                      | Where                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| Recording preparation                           | Working; recording itself is manual                                                                                                                     | Gin, jx                                  | `process-recordings`, [recording guide](docs/recording-setup.md) |
| Labelling and export                            | Working                                                                                                                                                 | Gin, jx                                  | `label-segments`, `review-segments`, `export-segments`           |
| Player and bomb detector                        | Working: colour and shape rules, no training. Splits overlapping icons, drops death markers, finds the C4 and its carrier.                              | zhengyongheng1, on Gin's colour baseline | `live_preview.detect`, `icon_split`, `bomb`                      |
| Detector check                                  | Working: measured against demo positions on the training matches                                                                                        | jx                                       | `detector-check`, [doc](docs/detector-check.md)                  |
| ResNet-18 frame features                        | Working: frozen ImageNet weights                                                                                                                        | jx                                       | `tactic_model`, [tactic baseline](docs/tactic-baseline.md)       |
| Shared clip data                                | Working: detector channels per named map region, bomb state and plant estimate, ResNet features on one timeline. Draft regions (`mirage-spectator-v1`). | ri3n                                     | `build-clip-data`, [doc](docs/clip-data.md)                      |
| TCN strategy model                              | Working, with cross-validation and a scored test set                                                                                                    | ri3n                                     | `train-tactics`, `predict-tcn`, [Tactic model](#tactic-model)    |
| Tactic baseline (ResNet mean + linear)          | Working, development scores only                                                                                                                        | jx                                       | `tactic-baseline`                                                |
| Temporal models (temporal average pooling, GRU) | Temporal pooling evaluated in development. The Compact pooled GRU was also scored on the held-out test set and is available in both apps. | jx | [Temporal models](docs/tactic-temporal.md), [GRU inference](docs/gru-predict.md) |
| CNN icon detector                               | Exists; not used by the strategy models                                                                                                                 | Gin                                      | `detector`, `training`                                           |
| Web app                                         | MP4 round upload, whole-clip review, match-wide analysis, TCN/GRU predictions, removal and reset                                                        | violetdisp2408                           | [Run the web application](#run-the-web-application)              |

### Data

The **2026-11-16 dataset** contains **128 reviewed clips from six matches**.
The export has 83 training clips from Furia–Falcons, Spirit–Falcons,
Aurora–Vitality and MOUZ–BetBoom, and 45 test clips from NaVi–Falcons and
Vitality–Falcons. Spirit segment 5 has an uncertain boundary. Excluding it left
**82 training + 45 test = 127 clips** for modelling. The label files retain 14
pending and 20 excluded segments outside the export. The November export includes
the clips from 1 November. Use one export at a time. [Review notes](labels/review-notes.md)

## Reproduce the submitted results

**Inspect the scores from a clone.** Open the [model comparison notebook](notebooks/tactic-model-comparison.ipynb)
for match-separated validation and the [final test notebook](notebooks/tactic-final-test.ipynb)
for the 45 held-out clips. Both contain saved outputs. The [TCN test report](runs/tactic-tcn-clipdata-final/report.md)
and [GRU test report](runs/tactic-gru-clipdata-test-20261121/report.md) include per-class results.
For the upstream player-position check, read the [four-match detector report](docs/evidence/detector-check-2026-11-27/report.md).

**Download the dataset to rerun training.** The [dataset ZIP (75 MiB)](https://github.com/FraggyVz/CV-DL/releases/download/dataset-2026-11-16/stratscout-clips-2026-11-16.zip)
contains 128 cropped 10 fps MP4s, both split manifests and `export.json`.
All 128 clips match the SHA-256 prefixes recorded in the original detection cache.
The [dataset release](https://github.com/FraggyVz/CV-DL/releases/tag/dataset-2026-11-16)
includes the archive digest and per-clip checksums.

From the repository root in PowerShell, extract the downloaded ZIP into a fresh folder.
Adjust the ZIP path if your browser saves downloads elsewhere.

```powershell
Expand-Archive "$env:USERPROFILE\Downloads\stratscout-clips-2026-11-16.zip" -DestinationPath data/processed/datasets/2026-11-16
```

You should see `export.json`, `train/manifest.csv` and `test/manifest.csv` directly
under `data/processed/datasets/2026-11-16/`. Complete [Windows setup](#windows-setup),
then train with the [shared features](runs/clip-data-2026-11-16/) and a fresh output directory.

```powershell
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --clip-data runs/clip-data-2026-11-16 --exclude-uncertain --seeds 3 --output runs/submission-tcn-001
```

This runs the TCN's development comparison. The headline nested validation scores
use a separate epoch-selection procedure documented in the comparison notebook.
The [processing guide](docs/data-processing.md) covers feature extraction and training on new footage.

### Submitted model results (macro-F1)

Cross-validation on the 82 training clips (Spirit #5 excluded): each of the four training
matches is held out in turn, seeds 0–2. The main column is **nested epoch selection**: the
epoch is chosen from the three training matches only, so the scored match never helps
choose it. Values are the mean over the four held-out matches and three seeds, ± the seed
SD. The TCN's **other-folds rule** (each match scored at the epoch best on the other
matches, whose models were trained on it) is shown for reference only; it scored most
models higher.

| Model                                 | Input             | Nested macro-F1 (main) | Other-folds | Test (scored once)                                              | Details                                                      |
| ------------------------------------- | ----------------- | ---------------------- | ----------- | --------------------------------------------------------------- | ------------------------------------------------------------ |
| Mean pooling                          | ResNet images     | 0.518 ± 0.034          | 0.584       | not run                                                         | [tactic baseline](docs/tactic-baseline.md)                   |
| Temporal average pooling (3 segments) | ResNet images     | 0.620 ± 0.015          | 0.650       | not run                                                         | [temporal models](docs/tactic-temporal.md)                   |
| GRU                                   | ResNet images     | 0.600 ± 0.028          | 0.681       | not run                                                         | [temporal models](docs/tactic-temporal.md#the-gru)           |
| TCN                                   | Detector features | not yet run (slow)     | 0.679       | 0.919                                                           | [Tactic model](#tactic-model)                                |
| Pooled classifier                     | Detector features | not yet run (slow)     | 0.645       | 0.862                                                           | [Tactic model](#tactic-model)                                |
| Mean pooling                          | Clip data         | 0.796 ± 0.016          | 0.790       | not run                                                         | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| Temporal average pooling (3 segments) | Clip data         | 0.763 ± 0.048          | 0.775       | not run                                                         | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| GRU                                   | Clip data         | 0.792 ± 0.033          | 0.809       | [0.890 (ensemble 0.963)](notebooks/tactic-final-test.ipynb)     | [clip-data inputs](docs/tactic-temporal.md#clip-data-inputs) |
| **TCN**                               | Clip data         | **0.849 ± 0.021**      | 0.834       | [**0.897** (ensemble 0.927)](notebooks/tactic-final-test.ipynb) | [shared clip data](docs/clip-data.md)                        |
| Pooled classifier                     | Clip data         | 0.809 ± 0.035          | 0.823       | not reported                                                    | [shared clip data](docs/clip-data.md)                        |

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

- A person picks each tactic's start and end when preparing clips. The app analyses
  the entire uploaded clip as one interval. Full-match tactic segmentation is still open.
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
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
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

In the **TCN only** and **GRU only** tabs, each prediction is paired with a walkthrough
of clip preparation, detector counts, feature extraction and model scoring. Below it, the
round's own values are drawn stage by stage:
- detected icons on the radar with the map regions, and icon counts over time
- region occupancy heatmaps
- the ResNet-18 PCA features and the scaled model input
- the TCN's five dilated residual blocks, or the GRU's projection and hidden state
- mean and max pooling
- each seed's probabilities and their average

The analysis saves these values in `stratscout/web_pipeline.py`'s format, and the app reads
them from `GET /api/videos/{id}/analysis/pipeline/{segment_id}`. Rounds analysed before
this view existed need to be analysed again. It explains the recorded inference pipeline
and class ranking; it is not a feature-attribution explanation of why the model chose a
tactic.

Older recordings uploaded before match rounds were introduced remain unassigned and
available under **Previous recordings**. Existing saved segments are preserved; an older
recording with no segments defaults to its full duration. **Reset workspace** deletes
all app uploads and analyses across every match, including these previous recordings,
after confirmation. It does not delete source footage, labels, or checkpoints. Reset is
blocked while an analysis is running. The app does not discover round boundaries in a
full-match recording; upload one clip that already represents one round/tactic.

New uploads start with the prepared-clip option ticked, for already-prepared minimap clips
(494×500 at 10 fps). Untick it in the round panel for a full-screen recording, and
create the per-machine crop profile at
`configs/recording-video.json` using the [recording guide](docs/recording-setup.md)
before running analysis. The app defaults to the guide's 2× replay speed for full-screen
recordings; choose 1× if that recording was captured at normal speed. Prepared clips use
their source speed. New-round clips are analysed from start to finish.

Videos that the browser cannot decode get a separate H.264/yuv420p playback copy; the
uploaded original is preserved. Prepared analysis clips, manifests and result status are
also kept under `data/web/uploads/analysis/` for provenance. Docker Compose mounts
the tracked TCN and GRU checkpoints from the checkout. The image includes the CPU
PyTorch/OpenCV inference stack, FFmpeg and the ImageNet ResNet weights. The first build takes longer because it
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

Frontend unit tests use Node's built-in test runner and need no extra packages:
`node --experimental-strip-types --test tests/*.test.ts` from `frontend/` (or `npm test`).

## Where files go

Keep full match videos and local exports in the following folders so the commands
find them on each machine.

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
runs/                              experiment outputs: ignored by Git except the files listed below
```

- **Your own recording:** see [Process a recording](#process-a-recording). The batch
  reads `data/raw/inbox/` and writes `data/processed/recordings/<match>/`.
- **Shared match videos:** extract each match ZIP into `data/processed/recordings/` so
  each match is one folder with the videos directly inside it; see the
  [labelling guide](docs/labelling-guide.md#get-the-videos-do-this-once). Keep
  `manifest.json`: labels and exports check the match ID it contains.
- For the **submitted clip dataset**, follow [the download and extraction instructions](#reproduce-the-submitted-results).
  Use a fresh dated directory for each subsequent export.
- **Tracked in Git** under `runs/`:
  - the **final deployment models**, so a clone can predict straight away: the TCN in
    `runs/tactic-tcn-clipdata-final` and the Compact pooled GRU in
    `runs/tactic-gru-clipdata-final-20261121` (its `models/*.pt` and `report.json`);
  - the **comparison notebook's result files**: the `.json`, `.csv` and `.md` files of
    the run folders listed in the setup cell of
    `notebooks/tactic-model-comparison.ipynb`, so a clone can rerun it;
  - the **final test results**: the GRU's one test run in
    `runs/tactic-gru-clipdata-test-20261121` (made by `scripts/final_test_gru.py`), read
    by `notebooks/tactic-final-test.ipynb` with the TCN's saved test results.
  - the shared clip features in `runs/clip-data-2026-11-16/` and the per-clip player
    and bomb detections in `runs/tactic-cache/`.
- Keep raw match recordings, new exports and scratch runs outside Git. The three
  curated [demo clips](deploy/streamlit/samples/) ship with the app.

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
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --output runs/tactic-tcn-<new> --exclude-uncertain --seeds 3
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --clip-data runs/clip-data-2026-11-16 --output runs/tactic-tcn-clipdata-<new> --exclude-uncertain --seeds 3
python -m stratscout predict-tcn data/processed/datasets/2026-11-16/test/clips --output runs/predictions-001.csv
python -m stratscout predict-gru <clips> --with-tcn --background runs/backgrounds/<match>.png --output runs/predictions-gru-001.csv
python -m stratscout build-clip-data --dataset data/processed/datasets/2026-11-16 --output runs/clip-data-2026-11-16
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

### Tactic model

`train-tactics` trains a residual dilated TCN on an exported clip dataset, alongside a
time-blind control (the same features pooled over the clip) and a majority-class baseline.

- **Features.** Every clip frame goes through the spectator-radar detector, giving one
  feature vector per frame. Detections are cached in `runs/tactic-cache/`; the first run
  takes a few minutes. Each match's background is the median of frames from all of its
  clips. Each vector holds per-team soft occupancy on an 8×8 grid, team counts,
  centroids and spread, the C4 badge and elapsed seconds. No tracker is involved.
- **Model.** Steps are taken at 5 fps, so real timing is kept; clips are padded and
  masked, never stretched. Five residual blocks (dilations 1–16) see about 25 s, and
  pooling covers the whole clip.
- **Protocol.** Leave-one-match-out cross-validation on the training matches (5
  seeds by default; `--seeds 3` was used for the results above). The main score is
  cross-fold, as in the [tactic baseline](docs/tactic-baseline.md): each held-out match
  is scored at the epoch that was best on the other training matches. The
  "optimistic" score (one epoch chosen and scored on the same folds) is for comparison
  only. The learning rate is constant after a 5-epoch warm-up, so an epoch chosen
  inside a longer run equals a run of exactly that length. Clips with uncertain
  boundaries (currently Spirit #5) are refused unless `--exclude-uncertain` is passed,
  matching the clip loader; excluded clips are listed in the report. The test match is
  scored only with `--evaluate-test`: by then the models are trained on all training
  matches, and the score is reported per seed and as a seed ensemble. Run that once,
  after the settings are fixed; changing settings after seeing the test score turns
  the test matches into validation data.
- **Input.** By default the original features above, rebuilt with jitter and dropped
  detections in every training batch. With `--clip-data <build-clip-data folder>`, the
  model reads the [shared clip data](docs/clip-data.md) instead: 72 detector channels
  plus `--resnet-pcs` (default 32) ResNet numbers per row. The ResNet compression is
  fitted per fold on its training clips, with no input augmentation. The current model,
  `runs/tactic-tcn-clipdata-final`, uses this input.
- **Outputs.** `report.md`, `report.json`, `config.json` and `cv_predictions.csv`.
  The report gives macro-F1 per held-out match, then per-class precision, recall and
  F1 and a confusion matrix, pooled over the held-out matches (every training clip is
  predicted once, by a model that never saw its match). With `--evaluate-test` it also
  writes `predictions.csv`, `models/` and the same per-class tables for the test match.
- **Predicting new clips (`predict-tcn`).** TCN only; the GRU and temporal models load
  through `tactic_temporal.py`. `predict-tcn <clips or folders>` loads every TCN seed in
  `--models` (default `runs/tactic-tcn-clipdata-final/models`, the current model) and
  prints each clip's class and seed-ensemble probability. Predictions below
  `--min-confidence` (default 0.6) are flagged as uncertain, with the runner-up class.
  `--output new.csv` also saves every class probability. For a clip-data model, each
  clip goes through the same detector, map regions and ResNet-18 as `build-clip-data`,
  and each seed applies its own saved scaling and ResNet compression. It refuses a
  regions file whose version differs from the one the model was trained with
  (`--regions` points at another file).

  **What to give it:** prepared clips (.mp4), each covering one tactic. Prepare them like
  the training clips: the spectator-radar crop from `process-recordings` at 10 fps, cut
  from the start to the end of one tactic (`export-segments` clips are ready to use).
  The empty-map background is built per match from that match's clips, grouped by the
  file name before the first dot, as `export-segments` names them. Give at least about
  5 clips per match, or pass the prepared match video (`minimap-10fps.mp4`) with
  `--background-from`. It needs no labels.

- **Second opinion (`predict-gru`, `gru_predict.predict_clips`).** The final Compact pooled
  GRU (`train-gru-final`, epochs chosen by leave-one-match-out CV on the training matches)
  as an optional second opinion or backup; the TCN stays the selected model and the default.
  `predict-gru` mirrors `predict-tcn`'s options and output, with a model column, and
  `--with-tcn` adds the TCN's rows. For the web app, `predict_clips` computes the clip
  features once and returns JSON-ready results; `save-background` saves an empty-map
  image from a training match so a single clip is detected as in training. See the
  [prediction guide](docs/gru-predict.md).

The strategy models inherit colour-detector errors on stacked players and obscured
icons. The [detector check](docs/detector-check.md) compares estimated positions
with demo coordinates on the four training matches. The tactic test covers two
hand-segmented matches, both of which share teams with training.

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

| Location/modules                                                                                                      | Responsibility                                                                                             |
| --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `stratscout/media.py`, `video_profiles`, `video_preparation`, `recording_batch`, `fps_comparison`                     | Shared preparation and comparison APIs                                                                     |
| `segmentation`, `icon_split`, `bomb`, `rounds`, `live_preview`, `count_eval`                                          | Colour/shape detection, overlap handling, estimates and review                                             |
| `segments`, `segment_labeller`                                                                                        | Five-class segment records, validation, safe saving and the keyboard labeller                              |
| `segment_export`, `tactic_features`, `tactic_tcn`, `tcn_predict`                                                      | Clip export, per-frame clip features, the tactic TCN experiment and TCN prediction                         |
| `gru_predict`                                                                                                         | Final Compact pooled GRU, and TCN/GRU prediction for the web app ([guide](docs/gru-predict.md))            |
| `clip_data`, `mirage_regions.json`                                                                                    | Shared model-ready clip data: detector channels per map region, bomb state and ResNet features             |
| `preprocessing`, `detector`, `datasets`, `training`, `evaluation`, `synthetic`                                        | CNN icon detector and experiments                                                                          |
| `tracking`, `annotations`, `config`, `contracts`, `context`, `workflow`, `storage`, `video`, `pilot`, `visualisation` | Tracking, legacy labels, explicit run state and orchestration                                              |
| `stratscout/ingest/`                                                                                                  | Optional demo parsing, coordinate mapping and legacy exports                                               |
| `tests/`                                                                                                              | Regression, import-safety, media and synthetic workflow checks                                             |
| `configs/`, `data/`, `runs/`                                                                                          | Local settings, recordings, clip datasets and generated experiments; see [Where files go](#where-files-go) |
| `mirage_radar.png`                                                                                                    | Map asset; recordings, generated results and environments stay outside Git                                 |

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

1. **Evaluate on new teams and matches.** Keep the submitted splits frozen. Prioritise
   split attacks when collecting new data. Only nine `B_SPLIT` clips were available for training.
2. **Review the map regions** in `stratscout/mirage_regions.json`, using the
   [walkthrough notebook](notebooks/clip-data-walkthrough.ipynb). Connector also
   catches some Ts on direct A attacks.
3. **Resolve open labels** for a future dataset version and record the decisions in
   the [review notes](labels/review-notes.md). Keep the submitted annotations and scores reproducible.
4. **Extend beyond prepared Mirage clips.** Full-match tactic segmentation, reliable
   player identities and new maps each need separate validation.

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
