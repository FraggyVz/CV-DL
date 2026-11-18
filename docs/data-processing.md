# Data processing: from new labelled matches to model-ready data

Follow these steps whenever new matches are recorded and labelled. They turn a replay
recording into the shared data that every strategy model trains on.

Run every command from the repository root with the project environment:
`.\.venv\Scripts\python.exe -m stratscout <command>`.

```
recording ─► prepare ─► label ─► review ─► export dataset ─► build shared data ─► check ─► train models
  (1)          (1)       (2)       (3)          (4)                 (5)              (6)        (7)
```

## For each new match

### 1. Prepare the recording

Put the recording in `data/raw/inbox/`, then run:

```powershell
.\.venv\Scripts\python.exe -m stratscout process-recordings
```

This crops the spectator radar, corrects the 2× replay speed and writes the 5, 10 and
30 fps versions to `data/processed/recordings/<match>/`. The
[recording guide](recording-setup.md) has the capture settings.

### 2. Label it

```powershell
.\.venv\Scripts\python.exe -m stratscout label-segments --match <match> --reviewer <yourname>
```

Mark where each tactic starts and ends, and choose its class using the
[tactic guide](tactic-definitions.md). The labels are saved as
`labels/<match>.<yourname>.segments.json`. The [labelling guide](labelling-guide.md)
lists the keys.

### 3. Have a second person review it

```powershell
.\.venv\Scripts\python.exe -m stratscout review-segments --match <match> --reviewer <yourname> --by <checker> --all-labelled
```

Only reviewed segments are exported. Leave uncertain cases `pending`, or flag an
uncertain boundary, rather than forcing a class.

## Then, once for the whole dataset

### 4. Export a new dataset

```powershell
.\.venv\Scripts\python.exe -m stratscout export-segments --test navi-falcons-mirage-cologne-major vitality-falcons-mirage-esl-pro-league-season-24 --output data/processed/datasets/<date>
```

- Use a **new dated folder** each time and keep the old datasets, so earlier results stay
  reproducible.

**About `--test`.** It is a choice, not a fixed part of the command: it names the
matches held out for the final test. A match goes to test if its folder name (in
`data/processed/recordings/`) **contains** any of the given texts. Every other labelled
match goes to training.

- **Keep the same test matches** for every new dataset. The command above uses the
  2026-11-16 test set (NaVi–Falcons and Vitality–Falcons), so new results stay comparable
  with earlier ones. New matches you label go to training without any change to the
  command.
- **Use full folder names, not short ones.** `--test navi` also catches any future match
  with "navi" in its name (such as a later NaVi–Vitality match), and `falcons` catches
  four of the current matches. Short names move matches into test without any warning.
  A name that matches no folder *is* refused with an error.
- **Change it only to choose a new test set on purpose**, for example a match whose teams
  are not in training at all, which is a fairer test. List all the folders you want
  in test by their full names. Scores from a new test set cannot be compared with scores
  from the old one, so report which test set each result used (export writes it to
  `export.json` as `test_matches`).
- To check the split before training, open `data/processed/datasets/<date>/export.json`
  or the two `manifest.csv` files and confirm which matches ended up in test.
- Export uses every label file in `labels/` and every prepared recording on your
  machine. **Get all of them first.** A match whose labels or recording are missing is
  left out without an error.

### 5. Build the shared data

```powershell
.\.venv\Scripts\python.exe -m stratscout build-clip-data --dataset data/processed/datasets/<date> --output runs/clip-data-<date>
```

This runs the two upstream steps on every clip and combines them on one timeline:

| Step | Whose work | Time |
|---|---|---|
| Player and bomb detector | zhengyongheng1, on Gin's colour baseline | About 5 s per **new** clip. Clips seen before are read from `runs/tactic-cache/`. |
| Frozen ResNet-18 features | jx (tactic baseline) | About 5 s per clip, for **every** clip |

Expect about 12–15 minutes for 150 clips. The output folder must be new. Every field is
described in [clip-data.md](clip-data.md), and the
[walkthrough notebook](../notebooks/clip-data-walkthrough.ipynb) shows each step on a
real clip.

### 6. Check the report before training

Open `runs/clip-data-<date>/report.md`.

| Check | What it should look like (2026-11-16 for reference) |
|---|---|
| T detections placed | 100%. Lower means players were found outside every map region. |
| A_SPLIT vs A_DIRECT reaching connector | A_SPLIT higher (100% vs 60%) |
| Median A split gap | A_SPLIT short (1.0 s), A_DIRECT longer (8.6 s) |
| B_SPLIT vs B_DIRECT reaching short | B_SPLIT much higher (89% vs 8%) |
| RETAKE with an estimated plant | Most (75%), and far more than the attack classes (0–20%) |

If a class suddenly looks different, check the new labels and the map regions before
training. These checks use labels only to describe the data; no feature is built from
them.

### 7. Train the models

Train every model on the same dataset, the same exclusions, the same test matches and
the same leave-one-match-out folds. Record the `data_version` and `regions_version`
from `runs/clip-data-<date>/channels.json` with the results. **Score the test set once,
at the end**, after all settings are fixed.

Training on the shared data is not wired into the models yet. Until then,
`train-tactics` still trains the TCN directly on the exported dataset:

```powershell
.\.venv\Scripts\python.exe -m stratscout train-tactics --dataset data/processed/datasets/<date> --output runs/tactic-tcn-<new> --exclude-uncertain --seeds 3
```

## Things to watch

- **Same recording setup.** Mirage and the spectator radar layout only. A different
  radar size, layout or colour breaks both the detector and the map regions.
- **Map regions.** They live in `stratscout/mirage_regions.json`, currently version
  `mirage-spectator-v1`, a draft. If new matches show players outside every region
  (fewer than 100% placed), adjust the rectangles, raise `version` and rebuild. Use
  training matches only when deciding where a boundary goes.
- **Uncertain boundaries** are kept in the data and marked `uncertain` in `index.csv`.
  Models decide whether to exclude them, and should say so in their results.
- **Old datasets and runs stay.** Nothing overwrites an existing folder. Choose new
  names each time.
- **Which new matches help most:** teams not already in the data, and split rounds,
  especially B_SPLIT, which has only 9 training clips.

## Where files go

| What | Where | In Git? |
|---|---|---|
| Raw recordings | `data/raw/inbox/` | No (share outside GitHub) |
| Prepared recordings | `data/processed/recordings/<match>/` | No |
| Label files | `labels/` | **Yes:** commit them so everyone exports the same dataset |
| Exported datasets | `data/processed/datasets/<date>/` | No |
| Detection cache | `runs/tactic-cache/` | No |
| Shared clip data | `runs/clip-data-<date>/` | No |
| Model runs | `runs/<run name>/` | No |
