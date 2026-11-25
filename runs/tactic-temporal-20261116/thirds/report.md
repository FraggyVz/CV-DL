# Thirds development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.643 +/- 0.056 | 0.758 +/- 0.058 | 0.698 +/- 0.022 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.625 +/- 0.064 | 0.747 +/- 0.015 | 0.636 +/- 0.074 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.713 +/- 0.150 | 0.810 +/- 0.018 | 0.733 +/- 0.094 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.500 +/- 0.074 | 0.735 +/- 0.030 | 0.614 +/- 0.066 | 0.055 | 0.158 |
| Equal fold average | 0.620 +/- 0.015 | 0.763 +/- 0.010 | 0.671 +/- 0.010 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 75 | training matches | 67 |
| spirit-falcons-mirage-cologne-major | 1 | 74 | training matches | 54 |
| spirit-falcons-mirage-cologne-major | 2 | 153 | training matches | 71 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 61 | training matches | 58 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 62 | training matches | 79 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 66 | training matches | 140 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 64 | training matches | 94 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 49 | training matches | 98 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 86 | training matches | 73 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 63 | training matches | 51 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 59 | training matches | 50 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 53 | training matches | 51 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.635 |
| B_DIRECT | 11 | 0.778 |
| A_SPLIT | 2 | 0.333 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.505 |
| B_DIRECT | 5 | 0.524 |
| A_SPLIT | 6 | 0.605 |
| B_SPLIT | 2 | 0.490 |
| RETAKE | 4 | 1.000 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.815 |
| B_DIRECT | 5 | 0.655 |
| A_SPLIT | 1 | 0.667 |
| B_SPLIT | 3 | 0.667 |
| RETAKE | 7 | 0.763 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.554 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.361 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.836 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Thirds keeps only coarse order: early, middle and late averages. It does not see elapsed time, so a short and a long clip look alike.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
