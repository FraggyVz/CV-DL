# Temporal tactic models: development results

Inputs: clip data (data\processed\clip-data-2026-11-16-train-only; clip-data-1, regions mirage-spectator-v1): 104 numbers per row (72 detector channels + 32 ResNet PCA numbers, PCA and scaling fitted on each fit's training clips only).

All models share the frame features, 4 whole-match folds, other-folds epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is mean pooling on these rows.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| Mean pooling + linear | 525 | 0.790 +/- 0.022 | 0.818 +/- 0.019 | 0.855 +/- 0.017 |
| Temporal average pooling (3 segments) + linear classifier | 1565 | 0.775 +/- 0.018 | 0.828 +/- 0.029 | 0.875 +/- 0.016 |
| GRU | 2349 | 0.809 +/- 0.034 | 0.829 +/- 0.032 | 0.863 +/- 0.027 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | Mean pooling + linear | Temporal average pooling (3 segments) + linear classifier | GRU | Majority |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.663 +/- 0.026 | 0.723 +/- 0.044 | 0.651 +/- 0.041 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.807 +/- 0.027 | 0.744 +/- 0.033 | 0.796 +/- 0.022 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.873 +/- 0.045 | 0.917 +/- 0.018 | 0.905 +/- 0.068 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.815 +/- 0.025 | 0.716 +/- 0.024 | 0.884 +/- 0.057 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | Mean pooling + linear | Temporal average pooling (3 segments) + linear classifier | GRU |
|---|---:|---:|---:|---:|
| A_DIRECT | 15 | 0.770 | 0.836 | 0.822 |
| B_DIRECT | 24 | 0.862 | 0.824 | 0.925 |
| A_SPLIT | 14 | 0.757 | 0.749 | 0.698 |
| B_SPLIT | 9 | 0.706 | 0.553 | 0.744 |
| RETAKE | 20 | 0.853 | 0.913 | 0.856 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.034 (largest SD above).
- The same model differs by up to 0.254 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
