# Submitted model results

Every model scored on the 2026-11-16 dataset, with how each number was produced. The
[README](../README.md#reproduce-the-submitted-results) has the headline scores and the
steps to download the dataset and rerun training. What the results mean, and where the
models go wrong, is in [findings and limitations](challenges-and-findings.md).

## Macro-F1 for every model

Cross-validation on the 82 training clips (Spirit #5 excluded): each of the four training
matches is held out in turn, seeds 0–2. The main column is **nested epoch selection**: the
epoch is chosen from the three training matches only, so the scored match never helps
choose it. Values are the mean over the four held-out matches and three seeds, ± the seed
SD. The TCN's **other-folds rule** (each match scored at the epoch best on the other
matches, whose models were trained on it) is shown for reference only; it scored most
models higher.

| Model                                 | Input             | Nested macro-F1 (main) | Other-folds | Test (scored once)                                                 | Details                                                 |
| ------------------------------------- | ----------------- | ---------------------- | ----------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| Mean pooling                          | ResNet images     | 0.518 ± 0.034          | 0.584       | not run                                                            | [tactic baseline](tactic-baseline.md)                   |
| Temporal average pooling (3 segments) | ResNet images     | 0.620 ± 0.015          | 0.650       | not run                                                            | [temporal models](tactic-temporal.md)                   |
| GRU                                   | ResNet images     | 0.600 ± 0.028          | 0.681       | not run                                                            | [temporal models](tactic-temporal.md#the-gru)           |
| TCN                                   | Detector features | not yet run (slow)     | 0.679       | 0.919                                                              | [Tactic TCN](tactic-tcn.md)                             |
| Pooled classifier                     | Detector features | not yet run (slow)     | 0.645       | 0.862                                                              | [Tactic TCN](tactic-tcn.md)                             |
| Mean pooling                          | Clip data         | 0.796 ± 0.016          | 0.790       | not run                                                            | [clip-data inputs](tactic-temporal.md#clip-data-inputs) |
| Temporal average pooling (3 segments) | Clip data         | 0.763 ± 0.048          | 0.775       | not run                                                            | [clip-data inputs](tactic-temporal.md#clip-data-inputs) |
| GRU                                   | Clip data         | 0.792 ± 0.033          | 0.809       | [0.890 (ensemble 0.963)](../notebooks/tactic-final-test.ipynb)     | [clip-data inputs](tactic-temporal.md#clip-data-inputs) |
| **TCN**                               | Clip data         | **0.849 ± 0.021**      | 0.834       | [**0.897** (ensemble 0.927)](../notebooks/tactic-final-test.ipynb) | [shared clip data](clip-data.md)                        |
| Pooled classifier                     | Clip data         | 0.809 ± 0.035          | 0.823       | not reported                                                       | [shared clip data](clip-data.md)                        |

Test results for the clip-data TCN and GRU are in `notebooks/tactic-final-test.ipynb`
(macro-F1 pooled over the 45 test clips, mean over three seeds, with the seed ensemble in
brackets); the TCN was selected by nested cross-validation before the GRU was scored.

## Where the numbers come from

The pooled classifier is the TCN's control: the same rows pooled over the clip before a
small network, so step order cannot matter. The detector-features test scores come from
the earlier run (cross-validation 0.667 for the TCN and 0.625 for the pooled classifier);
the other-folds column is the 2026-11-19 rerun (`runs/tactic-tcn-repro-20261119`). Clip
data (ri3n's `build-clip-data`) is detector channels per map region plus 32 ResNet PCA
numbers fitted per fold; image runs use 16 sampled frames per clip. The nested TCN and
pooled-classifier rows come from `scripts/nested_tcn.py`; the other-folds clip-data rows
from a train-only rerun of `train-tactics`.

Every value and the per-match, per-class and error breakdowns are computed in
[`notebooks/tactic-model-comparison.ipynb`](../notebooks/tactic-model-comparison.ipynb).
The notebook is saved with its outputs. The result files it reads are tracked in Git, so a
clone can rerun it with the same scores. Figures 2.2 and 2.3 and a few review and input
checks also need the train-only dataset and clip-data copies; without them they show "—"
or "unverified".

## Other comparisons

- Training choices on ResNet images (nested): the Last-state GRU scores 0.345, the Compact
  last-state GRU 0.327 and the Compact pooled GRU 0.600; temporal average pooling without
  class weights 0.564, against 0.620 with them ([details](tactic-temporal.md#the-gru)).
- Majority class: 0.065 (test 0.047).
- Random-segment average pooling + linear classifier (control): the same three segments
  with the frames assigned at random, 0.526 (nested); evidence that frame order helps on
  these development matches ([details](tactic-temporal.md#the-random-segment-control)).
- Position features: the ResNet-row models on detector positions (8×8 grid) instead of
  image features: mean 0.595, temporal average pooling 0.557, first GRU 0.384 (nested)
  ([details](tactic-temporal.md#position-features)).
- The clip data's map regions were adjusted after looking at training labels, so the
  clip-data scores may be optimistic.

## How to read them

Report cross-validation as the estimate for new matches. The test score is optimistic:
the test teams also appear in training, and some classes have only 5–6 test clips. In
cross-validation, 7 of the TCN's 13 errors are direct vs split on the same site. Compare
models only on the same dataset, exclusions, folds and seeds.
