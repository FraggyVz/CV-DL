# Clip data: Mean pooling (Plan A) development results

Main result: nested. Macro-F1 includes all five classes; values are mean +/- SD across seeds.
Nested picks each epoch by leave-one-match-out on the training matches only, for the same seed, then refits on all of them.

| Validation match | nested macro-F1 | optimistic (selected on validation) macro-F1 | nested accuracy | Majority macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0.684 +/- 0.008 | 0.714 +/- 0.031 | 0.746 +/- 0.045 | 0.050 | 0.143 |
| furia-falcons-mirage-cologne-major-2026 | 0.813 +/- 0.018 | 0.847 +/- 0.018 | 0.803 +/- 0.021 | 0.074 | 0.227 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.885 +/- 0.027 | 0.939 +/- 0.025 | 0.900 +/- 0.000 | 0.080 | 0.250 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.803 +/- 0.021 | 0.902 +/- 0.071 | 0.825 +/- 0.025 | 0.055 | 0.158 |
| Equal fold average | 0.796 +/- 0.016 | 0.851 +/- 0.020 | 0.818 +/- 0.017 | 0.065 | 0.195 |

The overall row averages folds per seed before calculating the spread.

Seed ensemble (mean probability over seeds, pooled over all 82 held-out clips): macro-F1 0.790, accuracy 0.805.

## Epochs used

| Validation match | Seed | Scored epoch | Epoch from fold | Own optimistic epoch |
|---|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 0 | 23 | training matches | 18 |
| spirit-falcons-mirage-cologne-major | 1 | 9 | training matches | 15 |
| spirit-falcons-mirage-cologne-major | 2 | 10 | training matches | 10 |
| furia-falcons-mirage-cologne-major-2026 | 0 | 53 | training matches | 3 |
| furia-falcons-mirage-cologne-major-2026 | 1 | 60 | training matches | 46 |
| furia-falcons-mirage-cologne-major-2026 | 2 | 27 | training matches | 17 |
| mouz-betboom-mirage-iem-bejing-2026 | 0 | 108 | training matches | 21 |
| mouz-betboom-mirage-iem-bejing-2026 | 1 | 59 | training matches | 20 |
| mouz-betboom-mirage-iem-bejing-2026 | 2 | 23 | training matches | 22 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0 | 20 | training matches | 31 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 1 | 24 | training matches | 21 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 2 | 37 | training matches | 200 |

## Per-class F1: spirit-falcons-mirage-cologne-major

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.775 |
| B_DIRECT | 11 | 0.836 |
| A_SPLIT | 2 | 0.556 |
| B_SPLIT | 2 | 0.468 |
| RETAKE | 3 | 0.786 |

## Per-class F1: furia-falcons-mirage-cologne-major-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 5 | 0.752 |
| B_DIRECT | 5 | 0.889 |
| A_SPLIT | 6 | 0.721 |
| B_SPLIT | 2 | 0.800 |
| RETAKE | 4 | 0.905 |

## Per-class F1: mouz-betboom-mirage-iem-bejing-2026

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 4 | 0.878 |
| B_DIRECT | 5 | 0.909 |
| A_SPLIT | 1 | 0.889 |
| B_SPLIT | 3 | 0.800 |
| RETAKE | 7 | 0.949 |

## Per-class F1: aurora-vitality-mirage-esl-pro-league-season-24

nested scores, averaged over seeds. Clips are validation counts, not multiplied by seeds.

| Class | Clips | Mean F1 |
|---|---:|---:|
| A_DIRECT | 3 | 0.821 |
| B_DIRECT | 3 | 0.756 |
| A_SPLIT | 5 | 0.903 |
| B_SPLIT | 2 | 0.700 |
| RETAKE | 6 | 0.836 |

Excluded clips: spirit-falcons-mirage-cologne-major.jx.s005.

See fold JSON files for class counts, weights, per-class scores and confusion matrices
(rows are true labels; columns are predictions, in config.json class order).

Nested selection keeps each validation match out of its own model and epoch choice, but every match is still development data: the settings were chosen while looking at them. Inner fits train on one match fewer than the refit. This is not an independent test score.
The optimistic column selects and scores the epoch on the same validation match.
Seed spread describes initialisation variability, not uncertainty across unseen matches.
Temporal mean pooling ignores movement order and elapsed time.
Clip data rows are the shared clip data (docs/clip-data.md): detector channels per map region, drawn after looking at training labels, plus ResNet numbers compressed per fold. Rows already carry elapsed time and the regions reached so far.
No synthetic pretraining, detection, temporal localisation or final test evaluation.
