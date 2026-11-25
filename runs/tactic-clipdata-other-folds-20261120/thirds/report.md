# Clip data: Temporal average pooling (3 segments) + linear classifier development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.723 +/- 0.044 | 0.761 +/- 0.015 | 0.778 +/- 0.059 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.744 +/- 0.033 | 0.873 +/- 0.021 | 0.742 +/- 0.043 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.917 +/- 0.018 | 0.939 +/- 0.043 | 0.933 +/- 0.024 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.716 +/- 0.024 | 0.928 +/- 0.054 | 0.860 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.775 +/- 0.018 | 0.875 +/- 0.016 | 0.828 +/- 0.029 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.837, accuracy 0.854.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 6 | other folds | 15 |
| spirit-falcons-mirage-cologne-major | 1 | 6 | other folds | 15 |
| spirit-falcons-mirage-cologne-major | 2 | 6 | other folds | 25 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 6 | other folds | 22 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 6 | other folds | 14 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 6 | other folds | 32 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 7 | other folds | 6 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 7 | other folds | 11 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 7 | other folds | 4 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 32 | other folds | 6 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 32 | other folds | 10 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 32 | other folds | 4 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.756 |
| B_DIRECT | 11 | 0.857 |
| A_SPLIT | 2 | 0.500 |
| B_SPLIT | 2 | 0.679 |
| RETAKE | 3 | 0.821 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.767 |
| B_DIRECT | 5 | 0.750 |
| A_SPLIT | 6 | 0.681 |
| B_SPLIT | 2 | 0.667 |
| RETAKE | 4 | 0.857 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.915 |
| B_DIRECT | 5 | 0.939 |
| A_SPLIT | 1 | 0.889 |
| B_SPLIT | 3 | 0.867 |
| RETAKE | 7 | 0.974 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.905 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.926 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 1.000 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal average pooling (3 segments) keeps only coarse order: early, middle and late averages. It does not see elapsed time, so a short and a long clip look alike.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
