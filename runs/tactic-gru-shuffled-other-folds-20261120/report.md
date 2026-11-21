# Temporal tactic models: development results

Inputs: images (512 numbers per frame).

All models share the frame features, 4 whole-match folds, other-folds epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| GRU | 5613 | 0.681 +/- 0.049 | 0.724 +/- 0.057 | 0.783 +/- 0.000 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | GRU | Majority |
|---|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.791 +/- 0.034 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.654 +/- 0.074 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.711 +/- 0.025 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.567 +/- 0.068 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | GRU |
|---|---:|---:|
| A_DIRECT | 15 | 0.685 |
| B_DIRECT | 24 | 0.750 |
| A_SPLIT | 14 | 0.623 |
| B_SPLIT | 9 | 0.558 |
| RETAKE | 20 | 0.787 |

## Shuffled-label check

Labels are shuffled within each match and the whole procedure is refitted.
A sound procedure scores near chance on shuffled labels.

| Model | Real macro-F1 | Shuffled macro-F1, mean (min to max) | Shuffles at or above real |
|---|---:|---:|---:|
| GRU | 0.681 | 0.176 (0.101 to 0.264) | 0 of 20 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.049 (largest SD above).
- The same model differs by up to 0.223 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
