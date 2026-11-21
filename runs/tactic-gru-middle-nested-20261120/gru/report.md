# GRU development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.392 +/- 0.034 | 0.446 +/- 0.035 | 0.508 +/- 0.022 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.316 +/- 0.042 | 0.451 +/- 0.066 | 0.364 +/- 0.064 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.315 +/- 0.053 | 0.390 +/- 0.035 | 0.383 +/- 0.047 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.284 +/- 0.031 | 0.375 +/- 0.011 | 0.368 +/- 0.043 | 0.055 | 0.158 |
| Equal fold average | 0.327 +/- 0.004 | 0.415 +/- 0.032 | 0.406 +/- 0.021 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.321, accuracy 0.390.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 28 | training matches | 55 |
| spirit-falcons-mirage-cologne-major | 1 | 16 | training matches | 15 |
| spirit-falcons-mirage-cologne-major | 2 | 11 | training matches | 8 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 54 | training matches | 15 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 10 | training matches | 16 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 43 | training matches | 28 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 10 | training matches | 21 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 3 | training matches | 9 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 15 | training matches | 12 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 13 | training matches | 5 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 20 | training matches | 13 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 5 | training matches | 25 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.435 |
| B_DIRECT | 11 | 0.632 |
| A_SPLIT | 2 | 0.133 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 3 | 0.758 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.443 |
| B_DIRECT | 5 | 0.508 |
| A_SPLIT | 6 | 0.268 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 4 | 0.361 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.614 |
| B_DIRECT | 5 | 0.397 |
| A_SPLIT | 1 | 0.222 |
| B_SPLIT | 3 | 0.000 |
| RETAKE | 7 | 0.343 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.463 |
| B_DIRECT | 3 | 0.389 |
| A_SPLIT | 5 | 0.095 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.475 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
