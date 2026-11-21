# Temporal average pooling (3 segments) + linear classifier development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.561 +/- 0.044 | 0.740 +/- 0.041 | 0.667 +/- 0.000 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.652 +/- 0.091 | 0.734 +/- 0.006 | 0.652 +/- 0.107 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.609 +/- 0.054 | 0.836 +/- 0.029 | 0.633 +/- 0.062 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.435 +/- 0.004 | 0.632 +/- 0.020 | 0.579 +/- 0.000 | 0.055 | 0.158 |
| Equal fold average | 0.564 +/- 0.005 | 0.736 +/- 0.015 | 0.633 +/- 0.028 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.646, accuracy 0.683.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 58 | training matches | 39 |
| spirit-falcons-mirage-cologne-major | 1 | 69 | training matches | 70 |
| spirit-falcons-mirage-cologne-major | 2 | 80 | training matches | 59 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 60 | training matches | 61 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 58 | training matches | 54 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 73 | training matches | 73 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 58 | training matches | 70 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 63 | training matches | 71 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 62 | training matches | 69 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 53 | training matches | 44 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 71 | training matches | 62 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 59 | training matches | 76 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.579 |
| B_DIRECT | 11 | 0.772 |
| A_SPLIT | 2 | 0.356 |
| B_SPLIT | 2 | 0.433 |
| RETAKE | 3 | 0.667 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.549 |
| B_DIRECT | 5 | 0.792 |
| A_SPLIT | 6 | 0.481 |
| B_SPLIT | 2 | 0.522 |
| RETAKE | 4 | 0.915 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.492 |
| B_DIRECT | 5 | 0.655 |
| A_SPLIT | 1 | 0.556 |
| B_SPLIT | 3 | 0.656 |
| RETAKE | 7 | 0.686 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.515 |
| B_DIRECT | 3 | 0.750 |
| A_SPLIT | 5 | 0.000 |
| B_SPLIT | 2 | 0.000 |
| RETAKE | 6 | 0.909 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal average pooling (3 segments) keeps only coarse order: early, middle and late averages. It does not see elapsed time, so a short and a long clip look alike.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
