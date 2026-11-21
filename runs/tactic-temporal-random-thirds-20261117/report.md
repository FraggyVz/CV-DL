# Temporal tactic models: development results

All models share the frame features, 4 whole-match folds, nested epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| random-thirds | 7685 | 0.526 +/- 0.012 | 0.602 +/- 0.005 | 0.666 +/- 0.026 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | random-thirds | Majority |
|---|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.577 +/- 0.059 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.535 +/- 0.076 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.534 +/- 0.084 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.460 +/- 0.037 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | random-thirds |
|---|---:|---:|
| A_DIRECT | 15 | 0.474 |
| B_DIRECT | 24 | 0.632 |
| A_SPLIT | 14 | 0.451 |
| B_SPLIT | 9 | 0.361 |
| RETAKE | 20 | 0.714 |

## Shuffled-label check

Labels are shuffled within each match and the whole procedure is refitted.
A sound procedure scores near chance on shuffled labels.

| Model | Real macro-F1 | Shuffled macro-F1, mean (min to max) | Shuffles at or above real |
|---|---:|---:|---:|
| random-thirds | 0.526 | 0.142 (0.067 to 0.276) | 0 of 20 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.012 (largest SD above).
- The same model differs by up to 0.117 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
