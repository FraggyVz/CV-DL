# Temporal tactic models: development results

Inputs: images (512 numbers per frame).

All models share the frame features, 4 whole-match folds, other-folds epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| Mean pooling + linear | 2565 | 0.584 +/- 0.016 | 0.655 +/- 0.006 | 0.680 +/- 0.004 |
| Temporal average pooling (3 bins) + linear | 7685 | 0.650 +/- 0.013 | 0.699 +/- 0.012 | 0.763 +/- 0.010 |
| GRU | 23109 | 0.338 +/- 0.011 | 0.440 +/- 0.018 | 0.401 +/- 0.010 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | Mean pooling + linear | Temporal average pooling (3 bins) + linear | GRU | Majority |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.531 +/- 0.022 | 0.563 +/- 0.000 | 0.330 +/- 0.014 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.655 +/- 0.017 | 0.709 +/- 0.032 | 0.349 +/- 0.009 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.583 +/- 0.081 | 0.740 +/- 0.091 | 0.314 +/- 0.029 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.566 +/- 0.000 | 0.587 +/- 0.021 | 0.360 +/- 0.035 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | Mean pooling + linear | Temporal average pooling (3 bins) + linear | GRU |
|---|---:|---:|---:|---:|
| A_DIRECT | 15 | 0.587 | 0.622 | 0.459 |
| B_DIRECT | 24 | 0.681 | 0.712 | 0.571 |
| A_SPLIT | 14 | 0.357 | 0.572 | 0.120 |
| B_SPLIT | 9 | 0.510 | 0.525 | 0.000 |
| RETAKE | 20 | 0.785 | 0.817 | 0.541 |

## Shuffled-label check

Labels are shuffled within each match and the whole procedure is refitted.
A sound procedure scores near chance on shuffled labels.

| Model | Real macro-F1 | Shuffled macro-F1, mean (min to max) | Shuffles at or above real |
|---|---:|---:|---:|
| Mean pooling + linear | 0.584 | 0.179 (0.067 to 0.314) | 0 of 20 |
| Temporal average pooling (3 bins) + linear | 0.650 | 0.168 (0.106 to 0.295) | 0 of 20 |
| GRU | 0.338 | 0.180 (0.096 to 0.278) | 0 of 20 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.016 (largest SD above).
- The same model differs by up to 0.177 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
