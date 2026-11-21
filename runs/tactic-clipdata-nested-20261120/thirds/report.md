# Clip data: Temporal average pooling (3 segments) + linear classifier development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.692 +/- 0.080 | 0.761 +/- 0.015 | 0.778 +/- 0.059 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.770 +/- 0.027 | 0.873 +/- 0.021 | 0.773 +/- 0.037 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.885 +/- 0.027 | 0.939 +/- 0.043 | 0.917 +/- 0.024 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.706 +/- 0.061 | 0.928 +/- 0.054 | 0.789 +/- 0.074 | 0.055 | 0.158 |
| Equal fold average | 0.763 +/- 0.048 | 0.875 +/- 0.016 | 0.814 +/- 0.048 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.811, accuracy 0.829.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 5 | training matches | 15 |
| spirit-falcons-mirage-cologne-major | 1 | 7 | training matches | 15 |
| spirit-falcons-mirage-cologne-major | 2 | 9 | training matches | 25 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 9 | training matches | 22 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 7 | training matches | 14 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 9 | training matches | 32 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 49 | training matches | 6 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 9 | training matches | 11 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 9 | training matches | 4 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 12 | training matches | 6 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 19 | training matches | 10 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 5 | training matches | 4 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.711 |
| B_DIRECT | 11 | 0.879 |
| A_SPLIT | 2 | 0.333 |
| B_SPLIT | 2 | 0.679 |
| RETAKE | 3 | 0.857 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.792 |
| B_DIRECT | 5 | 0.750 |
| A_SPLIT | 6 | 0.734 |
| B_SPLIT | 2 | 0.667 |
| RETAKE | 4 | 0.905 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.868 |
| B_DIRECT | 5 | 0.939 |
| A_SPLIT | 1 | 0.778 |
| B_SPLIT | 3 | 0.867 |
| RETAKE | 7 | 0.974 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.806 |
| B_DIRECT | 3 | 0.767 |
| A_SPLIT | 5 | 0.822 |
| B_SPLIT | 2 | 0.267 |
| RETAKE | 6 | 0.867 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal average pooling (3 segments) keeps only coarse order: early, middle and late averages. It does not see elapsed time, so a short and a long clip look alike.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
