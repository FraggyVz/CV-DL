# GRU development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.387 +/- 0.023 | 0.516 +/- 0.083 | 0.556 +/- 0.059 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.348 +/- 0.013 | 0.493 +/- 0.062 | 0.409 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.343 +/- 0.052 | 0.532 +/- 0.053 | 0.450 +/- 0.082 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.336 +/- 0.047 | 0.439 +/- 0.048 | 0.386 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.354 +/- 0.020 | 0.495 +/- 0.032 | 0.450 +/- 0.011 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.367, accuracy 0.476.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 184 | other folds | 106 |
| spirit-falcons-mirage-cologne-major | 1 | 184 | other folds | 198 |
| spirit-falcons-mirage-cologne-major | 2 | 184 | other folds | 8 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 193 | other folds | 15 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 193 | other folds | 131 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 193 | other folds | 184 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 195 | other folds | 60 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 195 | other folds | 183 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 195 | other folds | 134 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 185 | other folds | 179 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 185 | other folds | 196 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 185 | other folds | 98 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.397 |
| B_DIRECT | 11 | 0.717 |
| A_SPLIT | 2 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 3 | 0.821 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.505 |
| B_DIRECT | 5 | 0.592 |
| A_SPLIT | 6 | 0.291 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 4 | 0.350 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.644 |
| B_DIRECT | 5 | 0.341 |
| A_SPLIT | 1 | 0.000 |
| B_SPLIT | 3 | 0.000 |
| RETAKE | 7 | 0.732 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.409 |
| B_DIRECT | 3 | 0.524 |
| A_SPLIT | 5 | 0.083 |
| B_SPLIT | 2 | 0.167 |
| RETAKE | 6 | 0.499 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
