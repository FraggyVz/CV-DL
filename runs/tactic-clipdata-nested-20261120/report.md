# Temporal tactic models: development results

Inputs: clip data (data\processed\clip-data-2026-11-16-train-only; clip-data-1, regions mirage-spectator-v1): 104 numbers per row (72 detector channels + 32 ResNet PCA numbers, PCA and scaling fitted on each fit's training clips only).

All models share the frame features, 4 whole-match folds, nested epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is mean pooling on these rows.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| Mean pooling + linear | 525 | 0.796 +/- 0.016 | 0.818 +/- 0.017 | 0.851 +/- 0.020 |
| Temporal average pooling (3 segments) + linear classifier | 1565 | 0.763 +/- 0.048 | 0.814 +/- 0.048 | 0.875 +/- 0.016 |
| GRU | 2349 | 0.792 +/- 0.033 | 0.806 +/- 0.037 | 0.862 +/- 0.028 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | Mean pooling + linear | Temporal average pooling (3 segments) + linear classifier | GRU | Majority |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.684 +/- 0.008 | 0.692 +/- 0.080 | 0.670 +/- 0.049 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.813 +/- 0.018 | 0.770 +/- 0.027 | 0.807 +/- 0.010 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.885 +/- 0.027 | 0.885 +/- 0.027 | 0.900 +/- 0.053 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.803 +/- 0.021 | 0.706 +/- 0.061 | 0.791 +/- 0.098 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | Mean pooling + linear | Temporal average pooling (3 segments) + linear classifier | GRU |
|---|---:|---:|---:|---:|
| A_DIRECT | 15 | 0.806 | 0.794 | 0.746 |
| B_DIRECT | 24 | 0.847 | 0.834 | 0.936 |
| A_SPLIT | 14 | 0.767 | 0.667 | 0.618 |
| B_SPLIT | 9 | 0.692 | 0.620 | 0.800 |
| RETAKE | 20 | 0.869 | 0.901 | 0.860 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.048 (largest SD above).
- The same model differs by up to 0.230 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
