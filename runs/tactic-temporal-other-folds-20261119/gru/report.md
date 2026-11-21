# GRU development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.330 +/- 0.014 | 0.409 +/- 0.007 | 0.476 +/- 0.039 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.349 +/- 0.009 | 0.451 +/- 0.028 | 0.409 +/- 0.000 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.314 +/- 0.029 | 0.363 +/- 0.034 | 0.417 +/- 0.047 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.360 +/- 0.035 | 0.382 +/- 0.033 | 0.456 +/- 0.050 | 0.055 | 0.158 |
| Equal fold average | 0.338 +/- 0.011 | 0.401 +/- 0.010 | 0.440 +/- 0.018 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.353, accuracy 0.439.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 17 | other folds | 28 |
| spirit-falcons-mirage-cologne-major | 1 | 17 | other folds | 5 |
| spirit-falcons-mirage-cologne-major | 2 | 17 | other folds | 35 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 44 | other folds | 2 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 44 | other folds | 10 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 44 | other folds | 1 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 50 | other folds | 8 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 50 | other folds | 1 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 50 | other folds | 61 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 50 | other folds | 16 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 50 | other folds | 6 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 50 | other folds | 35 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.333 |
| B_DIRECT | 11 | 0.652 |
| A_SPLIT | 2 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.373 |
| B_DIRECT | 5 | 0.481 |
| A_SPLIT | 6 | 0.482 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 4 | 0.408 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.667 |
| B_DIRECT | 5 | 0.419 |
| A_SPLIT | 1 | 0.000 |
| B_SPLIT | 3 | 0.000 |
| RETAKE | 7 | 0.485 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.462 |
| B_DIRECT | 3 | 0.730 |
| A_SPLIT | 5 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.606 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
