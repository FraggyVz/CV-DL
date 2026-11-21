# Temporal tactic models: development results

Inputs: images (512 numbers per frame).

All models share the frame features, 4 whole-match folds, nested epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| Temporal average pooling (3 segments) + linear classifier | 7685 | 0.564 +/- 0.005 | 0.633 +/- 0.028 | 0.736 +/- 0.015 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | Temporal average pooling (3 segments) + linear classifier | Majority |
|---|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.561 +/- 0.044 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.652 +/- 0.091 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.609 +/- 0.054 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.435 +/- 0.004 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | Temporal average pooling (3 segments) + linear classifier |
|---|---:|---:|
| A_DIRECT | 15 | 0.534 |
| B_DIRECT | 24 | 0.742 |
| A_SPLIT | 14 | 0.348 |
| B_SPLIT | 9 | 0.403 |
| RETAKE | 20 | 0.794 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.005 (largest SD above).
- The same model differs by up to 0.217 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
