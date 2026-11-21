# Temporal average pooling (3 bins) + linear development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.563 +/- 0.000 | 0.758 +/- 0.058 | 0.667 +/- 0.000 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.709 +/- 0.032 | 0.747 +/- 0.015 | 0.712 +/- 0.021 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.740 +/- 0.091 | 0.810 +/- 0.018 | 0.717 +/- 0.085 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.587 +/- 0.021 | 0.735 +/- 0.030 | 0.702 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.650 +/- 0.013 | 0.763 +/- 0.010 | 0.699 +/- 0.012 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.691, accuracy 0.707.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 122 | other folds | 67 |
| spirit-falcons-mirage-cologne-major | 1 | 122 | other folds | 54 |
| spirit-falcons-mirage-cologne-major | 2 | 122 | other folds | 71 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 73 | other folds | 58 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 73 | other folds | 79 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 73 | other folds | 140 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 73 | other folds | 94 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 73 | other folds | 98 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 73 | other folds | 187 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 73 | other folds | 51 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 73 | other folds | 50 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 73 | other folds | 51 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.571 |
| B_DIRECT | 11 | 0.778 |
| A_SPLIT | 2 | 0.000 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.557 |
| B_DIRECT | 5 | 0.690 |
| A_SPLIT | 6 | 0.710 |
| B_SPLIT | 2 | 0.635 |
| RETAKE | 4 | 0.952 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.737 |
| B_DIRECT | 5 | 0.631 |
| A_SPLIT | 1 | 0.889 |
| B_SPLIT | 3 | 0.667 |
| RETAKE | 7 | 0.778 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.622 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.690 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.873 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Thirds keeps only coarse order: early, middle and late averages. It does not see elapsed time, so a short and a long clip look alike.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
