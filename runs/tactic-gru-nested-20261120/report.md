# Temporal tactic models: development results

Inputs: images (512 numbers per frame).

All models share the frame features, 4 whole-match folds, nested epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| GRU | 5613 | 0.600 +/- 0.028 | 0.650 +/- 0.005 | 0.739 +/- 0.023 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | GRU | Majority |
|---|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.502 +/- 0.097 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.691 +/- 0.036 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.661 +/- 0.032 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.546 +/- 0.051 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | GRU |
|---|---:|---:|
| A_DIRECT | 15 | 0.556 |
| B_DIRECT | 24 | 0.767 |
| A_SPLIT | 14 | 0.424 |
| B_SPLIT | 9 | 0.494 |
| RETAKE | 20 | 0.760 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.028 (largest SD above).
- The same model differs by up to 0.189 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
