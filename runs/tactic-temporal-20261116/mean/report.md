# Mean pooling (Plan A) development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.578 +/- 0.044 | 0.672 +/- 0.022 | 0.651 +/- 0.022 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.565 +/- 0.071 | 0.688 +/- 0.013 | 0.591 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.429 +/- 0.061 | 0.676 +/- 0.027 | 0.517 +/- 0.062 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.500 +/- 0.024 | 0.667 +/- 0.026 | 0.596 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.518 +/- 0.034 | 0.676 +/- 0.008 | 0.589 +/- 0.025 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 53 | training matches | 29 |
| spirit-falcons-mirage-cologne-major | 1 | 63 | training matches | 53 |
| spirit-falcons-mirage-cologne-major | 2 | 54 | training matches | 35 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 33 | training matches | 48 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 57 | training matches | 25 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 175 | training matches | 165 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 26 | training matches | 127 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 39 | training matches | 31 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 60 | training matches | 29 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 51 | training matches | 34 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 56 | training matches | 36 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 46 | training matches | 23 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.378 |
| B_DIRECT | 11 | 0.778 |
| A_SPLIT | 2 | 0.267 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.422 |
| B_DIRECT | 5 | 0.468 |
| A_SPLIT | 6 | 0.609 |
| B_SPLIT | 2 | 0.468 |
| RETAKE | 4 | 0.857 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.522 |
| B_DIRECT | 5 | 0.557 |
| A_SPLIT | 1 | 0.133 |
| B_SPLIT | 3 | 0.413 |
| RETAKE | 7 | 0.519 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.482 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.468 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.800 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal mean pooling ignores movement order and elapsed time.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
