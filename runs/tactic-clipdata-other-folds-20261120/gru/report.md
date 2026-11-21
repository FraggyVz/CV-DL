# Clip data: GRU development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.651 +/- 0.041 | 0.705 +/- 0.028 | 0.730 +/- 0.022 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.796 +/- 0.022 | 0.866 +/- 0.054 | 0.773 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.905 +/- 0.068 | 0.949 +/- 0.039 | 0.917 +/- 0.047 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.884 +/- 0.057 | 0.930 +/- 0.040 | 0.895 +/- 0.043 | 0.055 | 0.158 |
| Equal fold average | 0.809 +/- 0.034 | 0.863 +/- 0.027 | 0.829 +/- 0.032 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.867, accuracy 0.878.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 36 | other folds | 39 |
| spirit-falcons-mirage-cologne-major | 1 | 36 | other folds | 54 |
| spirit-falcons-mirage-cologne-major | 2 | 36 | other folds | 31 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 45 | other folds | 25 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 45 | other folds | 13 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 45 | other folds | 24 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 45 | other folds | 25 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 45 | other folds | 22 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 45 | other folds | 47 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 49 | other folds | 52 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 49 | other folds | 186 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 49 | other folds | 45 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.822 |
| B_DIRECT | 11 | 0.861 |
| A_SPLIT | 2 | 0.467 |
| B_SPLIT | 2 | 0.378 |
| RETAKE | 3 | 0.726 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.682 |
| B_DIRECT | 5 | 0.963 |
| A_SPLIT | 6 | 0.524 |
| B_SPLIT | 2 | 0.933 |
| RETAKE | 4 | 0.878 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.832 |
| B_DIRECT | 5 | 0.970 |
| A_SPLIT | 1 | 0.889 |
| B_SPLIT | 3 | 0.886 |
| RETAKE | 7 | 0.949 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.952 |
| B_DIRECT | 3 | 0.905 |
| A_SPLIT | 5 | 0.914 |
| B_SPLIT | 2 | 0.778 |
| RETAKE | 6 | 0.873 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
