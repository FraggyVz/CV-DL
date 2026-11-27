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
| Understand how the TCN is trained and judged | [Tactic TCN](docs/tactic-tcn.md) and its [walkthrough notebook](notebooks/tactic-tcn-walkthrough.ipynb)               |
| See how accurate the detector is             | [Detector check](docs/detector-check.md)                                                                           |
| See every model's scores                     | [Submitted results](docs/results.md)                                                                               |
| Know the findings, challenges and limits     | [Findings, challenges, edge cases and limitations](docs/challenges-and-findings.md)                                |
| Run or change the web app                    | [Web app guide](docs/web-app.md) and the [Streamlit deployment](deploy/streamlit/README.md)                        |
| Work on the ResNet models                    | [Tactic baseline](docs/tactic-baseline.md), [clip loader](docs/clip-loader.md)                                     |
| Contribute code                              | [Contributor rules](AGENTS.md), [commands and code structure](docs/development.md) and [Development checks](#development-checks-and-historical-references) |

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
| TCN strategy model                              | Working, with cross-validation and a scored test set                                                                                                    | ri3n                                     | `train-tactics`, `predict-tcn`, [doc](docs/tactic-tcn.md)    |
| Tactic baseline (ResNet mean + linear)          | Working, development scores only                                                                                                                        | jx                                       | `tactic-baseline`                                                |
| Temporal models (temporal average pooling, GRU) | Temporal pooling evaluated in development. The Compact pooled GRU was also scored on the held-out test set and is available in both apps. | jx | [Temporal models](docs/tactic-temporal.md), [GRU inference](docs/gru-predict.md) |
| CNN icon detector                               | Exists; not used by the strategy models                                                                                                                 | Gin                                      | `detector`, `training`                                           |
| Web app                                         | MP4 round upload, whole-clip review, match-wide analysis, TCN/GRU predictions, removal and reset                                                        | violetdisp2408                           | [web app guide](docs/web-app.md)                                  |

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

Cross-validation holds out each of the four training matches in turn (82 clips, Spirit #5
excluded, seeds 0–2), with the epoch chosen from the other training matches only. The
test set is 45 clips from two held-out matches, never used for training or for choosing a
model.

| Model              | Input         | Nested CV (mean ± seed SD) | Test, mean over seeds | Test, seed ensemble |
| ------------------ | ------------- | -------------------------- | --------------------- | ------------------- |
| **TCN** (selected) | Clip data     | **0.849 ± 0.021**          | **0.897**             | **0.927**           |
| Compact pooled GRU | Clip data     | 0.792 ± 0.033              | 0.890                 | 0.963               |
| Pooled classifier  | Clip data     | 0.809 ± 0.035              | not reported          | not reported        |
| Best image model   | ResNet images | 0.620 ± 0.015              | not run               | not run             |
| Majority class     | –             | 0.065                      | 0.047                 | 0.047               |

Report cross-validation as the estimate for new matches. The test score is optimistic:
the test teams also appear in training, and some classes have only 5–6 test clips.
[Submitted results](docs/results.md) has every model, the other-folds scores and where
each number comes from; the [test notebook](notebooks/tactic-final-test.ipynb) has the
per-class breakdowns.

### Known gaps

- A person picks each tactic's start and end when preparing clips. The app analyses
  the entire uploaded clip as one interval. Full-match tactic segmentation is still open.
- The detector is rule-based and imperfect: stacked or hidden icons are missed. There is
  no reliable player identity (tracker).
- Map regions are a draft and need the team's review.
- Mirage and the spectator radar layout only.

The full list, with the evidence for each point, is in
[findings, challenges, edge cases and limitations](docs/challenges-and-findings.md).

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
it is not exposed to the network. Stop it with Ctrl+C, or run `docker compose down`.

Create a match, upload one MP4 per round, then choose **Analyse Round**. Each clip is
treated as one complete tactic. Predictions below 0.6 confidence are flagged for review.
New uploads assume a prepared radar clip (494×500 at 10 fps); a full-screen recording
needs the crop profile from the [recording guide](docs/recording-setup.md) first.

The [web app guide](docs/web-app.md) covers the pipeline view, removing and resetting
rounds, full-screen recordings, storage limits and frontend development. The cloud
version is described in the [Streamlit deployment](deploy/streamlit/README.md).

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

## Commands and code structure

`python -m stratscout --help` lists every command. The main ones, the preview's keys,
the legacy prototype and the module layout are in [development](docs/development.md).
The TCN's training protocol, inputs, outputs and `predict-tcn` are in
[Tactic TCN](docs/tactic-tcn.md).

## Known limitations

The limitations, edge cases and engineering challenges are collected in
[findings, challenges, edge cases and limitations](docs/challenges-and-findings.md#limitations).
The ones most likely to catch out a contributor:

- **Label safety:** legacy proposal and review writes can overwrite labels. Never target
  reviewed labels with automatic proposals.
- **Tracking:** fixed slots can invent or keep stale positions. Repair missing-player
  handling before relying on trajectories for classification.
- **Detection:** spawn stacks are under-counted, and hidden bomb carriers are inferred.
  Planted-bomb separation and round boundaries are unverified estimates.
- **Capture:** pauses, stalls or replay-speed changes break the constant-speed
  assumption. Check each new layout and recording.

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
