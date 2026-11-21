# Clip data: GRU development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.670 +/- 0.049 | 0.705 +/- 0.028 | 0.746 +/- 0.059 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.807 +/- 0.010 | 0.866 +/- 0.054 | 0.788 +/- 0.021 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.900 +/- 0.053 | 0.949 +/- 0.039 | 0.917 +/- 0.024 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.791 +/- 0.098 | 0.928 +/- 0.043 | 0.772 +/- 0.089 | 0.055 | 0.158 |
| Equal fold average | 0.792 +/- 0.033 | 0.862 +/- 0.028 | 0.806 +/- 0.037 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.856, accuracy 0.866.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 43 | training matches | 39 |
| spirit-falcons-mirage-cologne-major | 1 | 24 | training matches | 54 |
| spirit-falcons-mirage-cologne-major | 2 | 50 | training matches | 31 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 42 | training matches | 25 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 20 | training matches | 13 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 35 | training matches | 24 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 61 | training matches | 25 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 26 | training matches | 22 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 16 | training matches | 47 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 40 | training matches | 52 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 16 | training matches | 59 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 16 | training matches | 45 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.790 |
| B_DIRECT | 11 | 0.859 |
| A_SPLIT | 2 | 0.433 |
| B_SPLIT | 2 | 0.444 |
| RETAKE | 3 | 0.821 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.722 |
| B_DIRECT | 5 | 0.963 |
| A_SPLIT | 6 | 0.539 |
| B_SPLIT | 2 | 0.933 |
| RETAKE | 4 | 0.878 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.815 |
| B_DIRECT | 5 | 0.970 |
| A_SPLIT | 1 | 0.833 |
| B_SPLIT | 3 | 0.933 |
| RETAKE | 7 | 0.949 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.657 |
| B_DIRECT | 3 | 0.952 |
| A_SPLIT | 5 | 0.667 |
| B_SPLIT | 2 | 0.889 |
| RETAKE | 6 | 0.792 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
The GRU reads frames in order with their elapsed time, but has far more weights than training clips, so it can fit noise.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
