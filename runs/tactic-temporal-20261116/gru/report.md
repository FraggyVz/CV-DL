# GRU development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.376 +/- 0.033 | 0.409 +/- 0.007 | 0.524 +/- 0.039 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.353 +/- 0.026 | 0.451 +/- 0.028 | 0.409 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.286 +/- 0.036 | 0.356 +/- 0.037 | 0.367 +/- 0.062 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.363 +/- 0.029 | 0.382 +/- 0.033 | 0.456 +/- 0.050 | 0.055 | 0.158 |
| Equal fold average | 0.345 +/- 0.008 | 0.400 +/- 0.008 | 0.439 +/- 0.028 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 12 | training matches | 28 |
| spirit-falcons-mirage-cologne-major | 1 | 5 | training matches | 5 |
| spirit-falcons-mirage-cologne-major | 2 | 37 | training matches | 35 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 8 | training matches | 2 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 37 | training matches | 10 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 12 | training matches | 1 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 15 | training matches | 8 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 6 | training matches | 1 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 22 | training matches | 17 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 15 | training matches | 16 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 17 | training matches | 6 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 30 | training matches | 35 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.413 |
| B_DIRECT | 11 | 0.710 |
| A_SPLIT | 2 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 3 | 0.758 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.373 |
| B_DIRECT | 5 | 0.436 |
| A_SPLIT | 6 | 0.522 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 4 | 0.435 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.667 |
| B_DIRECT | 5 | 0.317 |
| A_SPLIT | 1 | 0.000 |
| B_SPLIT | 3 | 0.000 |
| RETAKE | 7 | 0.448 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.462 |
| B_DIRECT | 3 | 0.739 |
| A_SPLIT | 5 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.615 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
