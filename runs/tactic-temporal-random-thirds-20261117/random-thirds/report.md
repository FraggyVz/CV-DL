# Random-thirds control development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.577 +/- 0.059 | 0.704 +/- 0.030 | 0.667 +/- 0.039 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.535 +/- 0.076 | 0.652 +/- 0.014 | 0.545 +/- 0.064 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.534 +/- 0.084 | 0.691 +/- 0.025 | 0.600 +/- 0.071 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.460 +/- 0.037 | 0.616 +/- 0.074 | 0.596 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.526 +/- 0.012 | 0.666 +/- 0.026 | 0.602 +/- 0.005 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 59 | training matches | 58 |
| spirit-falcons-mirage-cologne-major | 1 | 62 | training matches | 78 |
| spirit-falcons-mirage-cologne-major | 2 | 75 | training matches | 85 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 45 | training matches | 59 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 63 | training matches | 55 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 64 | training matches | 60 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 45 | training matches | 96 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 61 | training matches | 65 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 60 | training matches | 40 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 61 | training matches | 59 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 45 | training matches | 47 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 60 | training matches | 74 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.606 |
| B_DIRECT | 11 | 0.758 |
| A_SPLIT | 2 | 0.300 |
| B_SPLIT | 2 | 0.556 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.520 |
| B_DIRECT | 5 | 0.490 |
| A_SPLIT | 6 | 0.561 |
| B_SPLIT | 2 | 0.397 |
| RETAKE | 4 | 0.705 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.494 |
| B_DIRECT | 5 | 0.591 |
| A_SPLIT | 1 | 0.356 |
| B_SPLIT | 3 | 0.490 |
| RETAKE | 7 | 0.739 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.278 |
| B_DIRECT | 3 | 0.690 |
| A_SPLIT | 5 | 0.586 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.745 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Control: the same three averages and head as thirds, but frames are assigned to the groups at random, so it has thirds' extra inputs without its frame order.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
