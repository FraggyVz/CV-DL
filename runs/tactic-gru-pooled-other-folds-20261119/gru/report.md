# GRU development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.791 +/- 0.034 | 0.835 +/- 0.000 | 0.810 +/- 0.039 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.654 +/- 0.074 | 0.762 +/- 0.023 | 0.652 +/- 0.077 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.711 +/- 0.025 | 0.802 +/- 0.054 | 0.750 +/- 0.041 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.567 +/- 0.068 | 0.733 +/- 0.052 | 0.684 +/- 0.074 | 0.055 | 0.158 |
| Equal fold average | 0.681 +/- 0.049 | 0.783 +/- 0.000 | 0.724 +/- 0.057 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.736, accuracy 0.756.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 127 | other folds | 142 |
| spirit-falcons-mirage-cologne-major | 1 | 127 | other folds | 120 |
| spirit-falcons-mirage-cologne-major | 2 | 127 | other folds | 109 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 130 | other folds | 155 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 130 | other folds | 138 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 130 | other folds | 23 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 126 | other folds | 45 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 126 | other folds | 8 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 126 | other folds | 33 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 120 | other folds | 125 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 120 | other folds | 196 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 120 | other folds | 52 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.857 |
| B_DIRECT | 11 | 0.840 |
| A_SPLIT | 2 | 0.667 |
| B_SPLIT | 2 | 0.867 |
| RETAKE | 3 | 0.722 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.589 |
| B_DIRECT | 5 | 0.700 |
| A_SPLIT | 6 | 0.660 |
| B_SPLIT | 2 | 0.656 |
| RETAKE | 4 | 0.667 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.667 |
| B_DIRECT | 5 | 0.770 |
| A_SPLIT | 1 | 0.556 |
| B_SPLIT | 3 | 0.711 |
| RETAKE | 7 | 0.850 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.626 |
| B_DIRECT | 3 | 0.690 |
| A_SPLIT | 5 | 0.611 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.909 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
