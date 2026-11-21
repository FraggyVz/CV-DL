# Clip data: Mean pooling (Plan A) development results

Main result: other-folds. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Other-folds (the TCN's rule) scores each match at the epoch best on the other folds' seed-mean curve; slightly optimistic.

| Validation match | other-folds macro-F1 | optimistic (selected on validation) macro-F1 | other-folds accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.663 +/- 0.026 | 0.723 +/- 0.021 | 0.762 +/- 0.039 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.807 +/- 0.027 | 0.856 +/- 0.028 | 0.803 +/- 0.021 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.873 +/- 0.045 | 0.939 +/- 0.025 | 0.883 +/- 0.024 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.815 +/- 0.025 | 0.902 +/- 0.071 | 0.825 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.790 +/- 0.022 | 0.855 +/- 0.017 | 0.818 +/- 0.019 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.803, accuracy 0.817.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 21 | other folds | 18 |
| spirit-falcons-mirage-cologne-major | 1 | 21 | other folds | 15 |
| spirit-falcons-mirage-cologne-major | 2 | 21 | other folds | 109 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 21 | other folds | 3 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 21 | other folds | 46 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 21 | other folds | 162 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 109 | other folds | 21 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 109 | other folds | 20 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 109 | other folds | 22 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 17 | other folds | 31 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 17 | other folds | 21 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 17 | other folds | 200 |

## Per-class F1: spirit-falcons-mirage-cologne-major

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.667 |
| B_DIRECT | 11 | 0.898 |
| A_SPLIT | 2 | 0.500 |
| B_SPLIT | 2 | 0.467 |
| RETAKE | 3 | 0.786 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.776 |
| B_DIRECT | 5 | 0.843 |
| A_SPLIT | 6 | 0.755 |
| B_SPLIT | 2 | 0.756 |
| RETAKE | 4 | 0.905 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.843 |
| B_DIRECT | 5 | 0.909 |
| A_SPLIT | 1 | 0.889 |
| B_SPLIT | 3 | 0.800 |
| RETAKE | 7 | 0.923 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

other-folds scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.794 |
| B_DIRECT | 3 | 0.800 |
| A_SPLIT | 5 | 0.884 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 6 | 0.800 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Other-folds is slightly optimistic: the epoch for a match comes from the other folds' models, and each of those trained on the scored match. Nested selection is the stricter estimate. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal mean pooling ignores movement order and elapsed time.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
