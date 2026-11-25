# Temporal tactic models: development results

All models share the frame features, 4 whole-match folds, nested epoch choice,
seeds, class weights and metrics. Values are mean +/- SD across seeds; the overall row
averages matches equally per seed first. `mean` is Plan A, refitted on the same cache.

| Model | Trainable weights | Macro-F1 | Accuracy | Optimistic macro-F1 |
|---|---:|---:|---:|---:|
| mean | 2565 | 0.518 +/- 0.034 | 0.589 +/- 0.025 | 0.676 +/- 0.008 |
| thirds | 7685 | 0.620 +/- 0.015 | 0.671 +/- 0.010 | 0.763 +/- 0.010 |
| gru | 23109 | 0.345 +/- 0.008 | 0.439 +/- 0.028 | 0.400 +/- 0.008 |
| majority guess | 0 | 0.065 | 0.195 | - |

## Macro-F1 for each validation match

| Validation match | Clips | mean | thirds | gru | Majority |
|---|---:|---:|---:|---:|---:|
| spirit-falcons-mirage-cologne-major | 21 | 0.578 +/- 0.044 | 0.643 +/- 0.056 | 0.376 +/- 0.033 | 0.050 |
| furia-falcons-mirage-cologne-major-2026 | 22 | 0.565 +/- 0.071 | 0.625 +/- 0.064 | 0.353 +/- 0.026 | 0.074 |
| mouz-betboom-mirage-iem-bejing-2026 | 20 | 0.429 +/- 0.061 | 0.713 +/- 0.150 | 0.286 +/- 0.036 | 0.080 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 19 | 0.500 +/- 0.024 | 0.500 +/- 0.074 | 0.363 +/- 0.029 | 0.055 |

## F1 for each class

Mean over validation matches and seeds. Clips are totals across matches.

| Class | Clips | mean | thirds | gru |
|---|---:|---:|---:|---:|
| A_DIRECT | 15 | 0.451 | 0.627 | 0.478 |
| B_DIRECT | 24 | 0.638 | 0.677 | 0.550 |
| A_SPLIT | 14 | 0.369 | 0.492 | 0.131 |
| B_SPLIT | 9 | 0.420 | 0.489 | 0.000 |
| RETAKE | 20 | 0.711 | 0.817 | 0.564 |

## Shuffled-label check

Labels are shuffled within each match and the whole procedure is refitted.
A sound procedure scores near chance on shuffled labels.

| Model | Real macro-F1 | Shuffled macro-F1, mean (min to max) | Shuffles at or above real |
|---|---:|---:|---:|
| mean | 0.518 | 0.146 (0.065 to 0.281) | 0 of 20 |
| thirds | 0.620 | 0.155 (0.074 to 0.271) | 0 of 20 |
| gru | 0.345 | 0.171 (0.108 to 0.283) | 0 of 20 |

## Reading these numbers

There are only 82 clips from 4 matches. Small differences between a model
and the baseline are within noise and do not show that one is better:

- Changing only the random seed moves overall macro-F1 by up to 0.034 (largest SD above).
- The same model differs by up to 0.213 macro-F1 between validation matches.
- One clip is several points of F1 for a class with two or three validation clips.

Treat a gap smaller than those as a tie. These are development scores on the training
matches, not test results; the test split is never opened here.
Repeated frames in this run: 0.
