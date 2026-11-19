# Tactic TCN experiment

Dataset: `data\processed\datasets\2026-11-16`. Receptive field of the conv stack: 25.0 s (pooling then covers the whole clip).

Input: shared clip data `runs\clip-data-2026-11-16` (clip-data-1, regions mirage-spectator-v1): 72 detector channels + 32 ResNet numbers per row (compressed per fold on its training clips); no input augmentation.

| Split | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| train | 15 | 24 | 14 | 9 | 20 |
| test | 15 | 6 | 6 | 5 | 13 |

Excluded (uncertain boundary): spirit-falcons-mirage-cologne-major.jx.s005.

## Cross-validation (training matches only)

Each training match is held out in turn and scored (mean macro-F1 over seeds) at the epoch that was best on the other match, so no match picks its own epoch. Optimistic picks one epoch on all folds and scores the same folds; use it only as a comparison. Final epochs: used when training on all training matches.

| Model | Cross-fold | held out: aurora-vitality-mirage-esl-pro-league-season-24 (epoch) | held out: furia-falcons-mirage-cologne-major-2026 (epoch) | held out: mouz-betboom-mirage-iem-bejing-2026 (epoch) | held out: spirit-falcons-mirage-cologne-major (epoch) | Optimistic | Final epochs |
|---|---|---|---|---|---|---|---|
| majority (B_DIRECT) | 0.065 | 0.055 | 0.074 | 0.080 | 0.050 | – | – |
| tcn | 0.836 | 0.955 (140) | 0.906 (50) | 0.783 (30) | 0.700 (140) | 0.867 | 140 |
| control | 0.823 | 0.877 (150) | 0.810 (140) | 0.885 (150) | 0.720 (130) | 0.835 | 150 |

### Per class, pooled over held-out matches

At the cross-fold epochs every training clip is predicted once, by models that never saw its match. Pooled macro-F1 can differ from the per-fold mean above. Per-clip predictions: `cv_predictions.csv`.

#### tcn

Macro-F1 per seed: 0.850 ± 0.028; seed ensemble: 0.833 (accuracy 0.841).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.79 | 0.73 | 0.76 | 15 |
| B_DIRECT | 0.95 | 0.88 | 0.91 | 24 |
| A_SPLIT | 0.68 | 0.93 | 0.79 | 14 |
| B_SPLIT | 0.80 | 0.89 | 0.84 | 9 |
| RETAKE | 0.94 | 0.80 | 0.86 | 20 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 11 | 0 | 4 | 0 | 0 |
| B_DIRECT | 0 | 21 | 0 | 2 | 1 |
| A_SPLIT | 1 | 0 | 13 | 0 | 0 |
| B_SPLIT | 0 | 0 | 1 | 8 | 0 |
| RETAKE | 2 | 1 | 1 | 0 | 16 |

#### control

Macro-F1 per seed: 0.824 ± 0.032; seed ensemble: 0.845 (accuracy 0.866).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.72 | 0.87 | 0.79 | 15 |
| B_DIRECT | 1.00 | 0.96 | 0.98 | 24 |
| A_SPLIT | 0.75 | 0.64 | 0.69 | 14 |
| B_SPLIT | 0.80 | 0.89 | 0.84 | 9 |
| RETAKE | 0.95 | 0.90 | 0.92 | 20 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 13 | 0 | 1 | 1 | 0 |
| B_DIRECT | 0 | 23 | 0 | 1 | 0 |
| A_SPLIT | 4 | 0 | 9 | 0 | 1 |
| B_SPLIT | 0 | 0 | 1 | 8 | 0 |
| RETAKE | 1 | 0 | 1 | 0 | 18 |

## Test match (scored once)

Majority baseline macro-F1: 0.047.

#### tcn

Macro-F1 per seed: 0.897 ± 0.009; seed ensemble: 0.927 (accuracy 0.933).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.93 | 0.93 | 0.93 | 15 |
| B_DIRECT | 1.00 | 0.83 | 0.91 | 6 |
| A_SPLIT | 1.00 | 1.00 | 1.00 | 6 |
| B_SPLIT | 0.71 | 1.00 | 0.83 | 5 |
| RETAKE | 1.00 | 0.92 | 0.96 | 13 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 14 | 0 | 0 | 1 | 0 |
| B_DIRECT | 0 | 5 | 0 | 1 | 0 |
| A_SPLIT | 0 | 0 | 6 | 0 | 0 |
| B_SPLIT | 0 | 0 | 0 | 5 | 0 |
| RETAKE | 1 | 0 | 0 | 0 | 12 |

#### control

Macro-F1 per seed: 0.829 ± 0.018; seed ensemble: 0.832 (accuracy 0.844).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.92 | 0.73 | 0.81 | 15 |
| B_DIRECT | 0.83 | 0.83 | 0.83 | 6 |
| A_SPLIT | 0.60 | 1.00 | 0.75 | 6 |
| B_SPLIT | 0.80 | 0.80 | 0.80 | 5 |
| RETAKE | 1.00 | 0.92 | 0.96 | 13 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 11 | 0 | 4 | 0 | 0 |
| B_DIRECT | 0 | 5 | 0 | 1 | 0 |
| A_SPLIT | 0 | 0 | 6 | 0 | 0 |
| B_SPLIT | 0 | 1 | 0 | 4 | 0 |
| RETAKE | 1 | 0 | 0 | 0 | 12 |
