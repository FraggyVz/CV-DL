# Temporal tactic models: development results

Inputs: images (512 numbers per frame).

All models share the frame features, 4 whole-match folds, other-folds epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| GRU | 5533 | 0.354 +/- 0.020 | 0.450 +/- 0.011 | 0.495 +/- 0.032 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | GRU | Majority |
|---|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.387 +/- 0.023 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.348 +/- 0.013 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.343 +/- 0.052 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.336 +/- 0.047 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | GRU |
|---|---:|---:|
| A_DIRECT | 15 | 0.489 |
| B_DIRECT | 24 | 0.543 |
| A_SPLIT | 14 | 0.094 |
| B_SPLIT | 9 | 0.042 |
| RETAKE | 20 | 0.601 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.020 (largest SD above).
- The same model differs by up to 0.051 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
