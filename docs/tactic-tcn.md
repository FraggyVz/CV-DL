# Tactic TCN

`train-tactics` trains a residual dilated TCN on an exported clip dataset, alongside a
time-blind control (the same features pooled over the clip) and a majority-class baseline.
The current model is `runs/tactic-tcn-clipdata-final`. Its scores are in
[submitted results](results.md), and the
[TCN walkthrough notebook](../notebooks/tactic-tcn-walkthrough.ipynb) steps through one
clip.

## Train it

```powershell
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --clip-data runs/clip-data-2026-11-16 --output runs/tactic-tcn-clipdata-<new> --exclude-uncertain --seeds 3
```

Without `--clip-data`, it trains on the original detector features below. The output
directory must not already exist.

## How it works

- **Features.** Every clip frame goes through the spectator-radar detector, giving one
  feature vector per frame. Detections are cached in `runs/tactic-cache/`; the first run
  takes a few minutes. Each match's background is the median of frames from all of its
  clips. Each vector holds per-team soft occupancy on an 8×8 grid, team counts,
  centroids and spread, the C4 badge and elapsed seconds. No tracker is involved.
- **Model.** Steps are taken at 5 fps, so real timing is kept; clips are padded and
  masked, never stretched. Five residual blocks (dilations 1–16) see about 25 s, and
  pooling covers the whole clip.
- **Protocol.** Leave-one-match-out cross-validation on the training matches (5
  seeds by default; `--seeds 3` was used for the submitted results). The main score is
  cross-fold, as in the [tactic baseline](tactic-baseline.md): each held-out match
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
  model reads the [shared clip data](clip-data.md) instead: 72 detector channels
  plus `--resnet-pcs` (default 32) ResNet numbers per row. The ResNet compression is
  fitted per fold on its training clips, with no input augmentation. The current model
  uses this input.
- **Outputs.** `report.md`, `report.json`, `config.json` and `cv_predictions.csv`.
  The report gives macro-F1 per held-out match, then per-class precision, recall and
  F1 and a confusion matrix, pooled over the held-out matches (every training clip is
  predicted once, by a model that never saw its match). With `--evaluate-test` it also
  writes `predictions.csv`, `models/` and the same per-class tables for the test match.

## Predict new clips (`predict-tcn`)

```powershell
python -m stratscout predict-tcn data/processed/datasets/2026-11-16/test/clips --output runs/predictions-001.csv
```

TCN only; the GRU and temporal models load through `tactic_temporal.py`.
`predict-tcn <clips or folders>` loads every TCN seed in `--models` (default
`runs/tactic-tcn-clipdata-final/models`, the current model) and prints each clip's class
and seed-ensemble probability. Predictions below `--min-confidence` (default 0.6) are
flagged as uncertain, with the runner-up class. `--output new.csv` also saves every class
probability. For a clip-data model, each clip goes through the same detector, map regions
and ResNet-18 as `build-clip-data`, and each seed applies its own saved scaling and ResNet
compression. It refuses a regions file whose version differs from the one the model was
trained with (`--regions` points at another file).

**What to give it:** prepared clips (.mp4), each covering one tactic. Prepare them like
the training clips: the spectator-radar crop from `process-recordings` at 10 fps, cut
from the start to the end of one tactic (`export-segments` clips are ready to use).
The empty-map background is built per match from that match's clips, grouped by the
file name before the first dot, as `export-segments` names them. Give at least about
5 clips per match, or pass the prepared match video (`minimap-10fps.mp4`) with
`--background-from`. It needs no labels.

## Second opinion: the GRU (`predict-gru`)

The final Compact pooled GRU (`train-gru-final`, epochs chosen by leave-one-match-out CV
on the training matches) is an optional second opinion or backup; the TCN stays the
selected model and the default. `predict-gru` mirrors `predict-tcn`'s options and output,
with a model column, and `--with-tcn` adds the TCN's rows. For the web app,
`gru_predict.predict_clips` computes the clip features once and returns JSON-ready
results; `save-background` saves an empty-map image from a training match so a single
clip is detected as in training. See the [prediction guide](gru-predict.md).

## Limits

The strategy models inherit colour-detector errors on stacked players and obscured
icons. The [detector check](detector-check.md) compares estimated positions
with demo coordinates on the four training matches. The tactic test covers two
hand-segmented matches, both of which share teams with training. More in
[findings and limitations](challenges-and-findings.md).
