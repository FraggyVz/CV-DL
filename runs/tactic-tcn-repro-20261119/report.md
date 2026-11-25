# Tactic TCN experiment

Dataset: `data\processed\datasets\2026-11-16-train-only`. Receptive field of the conv stack: 25.0 s (pooling then covers the whole clip).

Input: the original detection features (141 per step), augmented in training.

| Split | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| train | 15 | 24 | 14 | 9 | 20 |
| test | 0 | 0 | 0 | 0 | 0 |

Excluded (uncertain boundary): spirit-falcons-mirage-cologne-major.jx.s005.

## Cross-validation (training matches only)

Each training match is held out in turn and scored (mean macro-F1 over seeds) at the epoch that was best on the other match, so no match picks its own epoch. Optimistic picks one epoch on all folds and scores the same folds; use it only as a comparison. Final epochs: used when training on all training matches.

| Model | Cross-fold | held out: aurora-vitality-mirage-esl-pro-league-season-24 (epoch) | held out: furia-falcons-mirage-cologne-major-2026 (epoch) | held out: mouz-betboom-mirage-iem-bejing-2026 (epoch) | held out: spirit-falcons-mirage-cologne-major (epoch) | Optimistic | Final epochs |
|---|---|---|---|---|---|---|---|
| majority (B_DIRECT) | 0.065 | 0.055 | 0.074 | 0.080 | 0.050 | – | – |
| tcn | 0.679 | 0.550 (90) | 0.871 (80) | 0.674 (80) | 0.620 (80) | 0.696 | 80 |
| control | 0.645 | 0.557 (20) | 0.785 (20) | 0.618 (20) | 0.621 (20) | 0.645 | 20 |

### Per class, pooled over held-out matches

At the cross-fold epochs every training clip is predicted once, by models that never saw its match. Pooled macro-F1 can differ from the per-fold mean above. Per-clip predictions: `cv_predictions.csv`.

#### tcn

Macro-F1 per seed: 0.677 ± 0.008; seed ensemble: 0.693 (accuracy 0.707).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.69 | 0.73 | 0.71 | 15 |
| B_DIRECT | 0.80 | 0.67 | 0.73 | 24 |
| A_SPLIT | 0.71 | 0.71 | 0.71 | 14 |
| B_SPLIT | 0.46 | 0.67 | 0.55 | 9 |
| RETAKE | 0.79 | 0.75 | 0.77 | 20 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 11 | 1 | 2 | 1 | 0 |
| B_DIRECT | 0 | 16 | 0 | 5 | 3 |
| A_SPLIT | 3 | 0 | 10 | 0 | 1 |
| B_SPLIT | 0 | 3 | 0 | 6 | 0 |
| RETAKE | 2 | 0 | 2 | 1 | 15 |

#### control

Macro-F1 per seed: 0.646 ± 0.019; seed ensemble: 0.617 (accuracy 0.634).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A_DIRECT | 0.53 | 0.67 | 0.59 | 15 |
| B_DIRECT | 0.88 | 0.62 | 0.73 | 24 |
| A_SPLIT | 0.47 | 0.50 | 0.48 | 14 |
| B_SPLIT | 0.46 | 0.67 | 0.55 | 9 |
| RETAKE | 0.78 | 0.70 | 0.74 | 20 |

Confusion (rows true, columns predicted):

| | A_DIRECT | B_DIRECT | A_SPLIT | B_SPLIT | RETAKE |
|---|---|---|---|---|---|
| A_DIRECT | 10 | 0 | 4 | 1 | 0 |
| B_DIRECT | 1 | 15 | 0 | 5 | 3 |
| A_SPLIT | 5 | 0 | 7 | 1 | 1 |
| B_SPLIT | 1 | 2 | 0 | 6 | 0 |
| RETAKE | 2 | 0 | 4 | 0 | 14 |

The test match was not scored (run with --evaluate-test once settings are fixed).
