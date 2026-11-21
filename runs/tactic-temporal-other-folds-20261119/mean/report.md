# Mean pooling (Plan A) development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.531 +/- 0.022 | 0.672 +/- 0.022 | 0.635 +/- 0.022 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.655 +/- 0.017 | 0.688 +/- 0.013 | 0.652 +/- 0.021 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.583 +/- 0.081 | 0.695 +/- 0.000 | 0.650 +/- 0.041 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.566 +/- 0.000 | 0.667 +/- 0.026 | 0.684 +/- 0.000 | 0.055 | 0.158 |
| Equal fold average | 0.584 +/- 0.016 | 0.680 +/- 0.004 | 0.655 +/- 0.006 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.627, accuracy 0.646.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 153 | other folds | 29 |
| spirit-falcons-mirage-cologne-major | 1 | 153 | other folds | 53 |
| spirit-falcons-mirage-cologne-major | 2 | 153 | other folds | 35 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 110 | other folds | 48 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 110 | other folds | 25 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 110 | other folds | 165 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 87 | other folds | 127 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 87 | other folds | 134 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 87 | other folds | 175 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 110 | other folds | 34 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 110 | other folds | 36 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 110 | other folds | 23 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.413 |
| B_DIRECT | 11 | 0.778 |
| A_SPLIT | 2 | 0.000 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.642 |
| B_DIRECT | 5 | 0.571 |
| A_SPLIT | 6 | 0.632 |
| B_SPLIT | 2 | 0.571 |
| RETAKE | 4 | 0.857 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.694 |
| B_DIRECT | 5 | 0.626 |
| A_SPLIT | 1 | 0.222 |
| B_SPLIT | 3 | 0.667 |
| RETAKE | 7 | 0.707 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.600 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.571 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.909 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal mean pooling ignores movement order and elapsed time.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
