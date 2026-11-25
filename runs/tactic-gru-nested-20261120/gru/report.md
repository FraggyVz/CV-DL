# GRU development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.502 +/- 0.097 | 0.736 +/- 0.094 | 0.587 +/- 0.059 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.691 +/- 0.036 | 0.755 +/- 0.025 | 0.682 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.661 +/- 0.032 | 0.794 +/- 0.056 | 0.717 +/- 0.047 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.546 +/- 0.051 | 0.672 +/- 0.036 | 0.614 +/- 0.050 | 0.055 | 0.158 |
| Equal fold average | 0.600 +/- 0.028 | 0.739 +/- 0.023 | 0.650 +/- 0.005 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.697, accuracy 0.720.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 6 | training matches | 23 |
| spirit-falcons-mirage-cologne-major | 1 | 17 | training matches | 39 |
| spirit-falcons-mirage-cologne-major | 2 | 36 | training matches | 109 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 65 | training matches | 56 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 51 | training matches | 17 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 67 | training matches | 23 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 4 | training matches | 7 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 42 | training matches | 8 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 71 | training matches | 33 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 19 | training matches | 2 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 84 | training matches | 2 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 9 | training matches | 13 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.472 |
| B_DIRECT | 11 | 0.716 |
| A_SPLIT | 2 | 0.222 |
| B_SPLIT | 2 | 0.433 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.567 |
| B_DIRECT | 5 | 0.822 |
| A_SPLIT | 6 | 0.627 |
| B_SPLIT | 2 | 0.711 |
| RETAKE | 4 | 0.730 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.635 |
| B_DIRECT | 5 | 0.744 |
| A_SPLIT | 1 | 0.467 |
| B_SPLIT | 3 | 0.611 |
| RETAKE | 7 | 0.850 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.548 |
| B_DIRECT | 3 | 0.786 |
| A_SPLIT | 5 | 0.381 |
| B_SPLIT | 2 | 0.222 |
| RETAKE | 6 | 0.792 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
