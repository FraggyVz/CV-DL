# Temporal models: thirds and GRU

Two models that look at frame order, trained on the same frozen ResNet-18 frame features
and the same protocol as the [tactic baseline](tactic-baseline.md), so the scores can be compared.

| Model | What it does | Why it is here |
|---|---|---|
| `mean` | Plan A: average all frames, then a linear layer | The baseline, refitted on the same cache |
| `thirds` | Average the early, middle and late third of the frames, join them, then a linear layer | A cheap check of whether frame order helps at all |
| `random-thirds` | Exactly like `thirds`, but each clip's frames go into the three groups at random | Control: thirds' extra inputs without its frame order |
| `gru` | A small GRU reads the frames in order with their times, then a linear layer | A learned temporal model |

## Run it

1. Install as for the [tactic baseline](tactic-baseline.md#install-do-this-once).
2. Train all three models, with the shuffled-label check:
   ```powershell
   python -m stratscout tactic-temporal --manifest data/processed/datasets/2026-11-16/train/manifest.csv --output runs/tactic-temporal --exclude-uncertain --shuffles 20
   ```
3. Open `report.md` in the output folder, or `notebooks/tactic-gru-results.ipynb` for figures.
4. Optionally run the control on the same features, into its own folder:
   ```powershell
   python -m stratscout tactic-temporal --manifest data/processed/datasets/2026-11-16/train/manifest.csv --output runs/tactic-temporal-random-thirds --exclude-uncertain --models random-thirds --features runs/tactic-temporal/frame-features.pt --shuffles 20
   ```
   The notebook reads it from `CONTROL_RUN` and puts it beside the other models.

**Common mistake:** an existing output folder causes an error; choose a new name.
`--shuffles 20` takes most of the time; leave it out for a quick run.

| Option | What happens |
|---|---|
| `--models gru` | Train only the listed models (`mean`, `thirds`, `random-thirds`, `gru`); the default is `mean thirds gru` |
| `--inputs positions` | Use detector position features instead of image features; see [position features](#position-features) |
| `--drop-channels bomb_badge` | Position inputs only: remove a channel, for an ablation |
| `--features runs/tactic-temporal/frame-features.pt` | Reuse an earlier cache; it is refused if the manifest, frame count, size or feature source differ |
| `--shuffles N` | Refit each model N times on labels shuffled within each match |
| `--fixed-epochs`, `--seeds`, `--epochs`, `--patience`, `--size`, `--frames` | Same meaning as in the baseline |

## What is shared with the baseline

Everything except the head and its inputs. The code is the baseline's, not a copy:

- the same 16 evenly spaced frames per clip and frozen ResNet-18 features;
- the same whole-match folds and [epoch choice](tactic-baseline.md#how-the-epoch-is-chosen);
- seeds 0, 1, 2; class weights and majority guess from training folds only;
- the same optimiser settings, metrics and report layout;
- only the training split. NaVi and Vitality-Falcons are never opened.

## The frame-feature cache

`frame-features.pt` keeps one 512-number feature for every sampled frame, with each
frame's time and repeat flag. The baseline's `features.pt` kept only the clip average.

- Averaging a clip's frames in the cache gives exactly the baseline's clip feature, so
  `mean` reproduces the baseline's scores. A test checks this.
- Any model can reuse it. A TCN can load it with `load_frame_features` in
  `stratscout.tactic_features`, or extract it with `extract_frame_features`.

## How time is handled

The loader samples 16 frames evenly across each clip, whatever its length. Frame order
alone would hide how long the clip lasted: 16 frames from a 6-second rush and from a
50-second slow attack look the same. That is how the old classifiers lost timing.

The GRU therefore gets each frame's real time as two extra inputs:

- seconds since the start of the clip;
- seconds since the previous frame.

Both come from the loader's `timestamps_s` and are divided by a fixed 30 seconds to keep
them small. Nothing is fitted to the data.

**Why not sample at a fixed rate instead?** It would change the frames, so the result
could no longer be compared with the baseline on identical features. It would also make
long clips into long sequences, which is harder to learn from 83 clips.

**Repeated frames.** A clip shorter than 16 frames repeats some frames; the loader marks
them in `repeat_mask`. The GRU and thirds drop the repeats, so no time step is counted
twice, and the GRU stops at each clip's last real frame. `mean` keeps them, to match the
baseline exactly. The 2026-11-16 clips are all long enough to have no repeats.

`thirds` does not get the times. It only tests order.

## The random-thirds control

`thirds` has three times as many inputs as `mean`. A gain could therefore come from the
extra inputs rather than from frame order. `random-thirds` separates the two:

- it drops repeats and cuts each clip into three groups of the same sizes as `thirds`;
- but the frames are shuffled first, so each group holds frames from any time in the clip;
- the shuffle uses a fixed seed made from the clip ID, so every run gives the same groups.

It has the same input size and head as `thirds`; only the time order is removed.

## The GRU

| Part | Setting |
|---|---|
| Standardise | Each of the 512 features, using the training fold's frames only; no weights |
| Project | One linear layer, 512 to 32, shared by every frame |
| GRU | One layer, 34 inputs (32 + 2 time inputs), 32 hidden units |
| Output | Linear layer on the last hidden state |
| Size | 23,109 trainable weights; `thirds` has 7,685 and `mean` 2,565 |

**Why standardise.** The first version fed the raw features to the GRU and scored far
below the baseline. Its training loss had barely moved when the stopping rule ended the
search, because clips differ by much less than the features' shared level. Standardising
fixed the training. The change was made after seeing development scores, so the GRU has
had one more design round than `thirds`; read its score with that in mind.

Apart from that fix, the GRU is one untuned configuration: the baseline's learning rate
and weight decay, no dropout, sizes chosen once in advance.

## Results on 2026-11-16

82 clips from four training matches (Spirit #5 left out as uncertain); four whole-match
folds, nested epoch choice, seeds 0, 1, 2. Runs: `runs/tactic-temporal-20261116` and, for
the control, `runs/tactic-temporal-random-thirds-20261117`. Each model was run once with
these settings; random-thirds was planned before its score was seen and not tuned after.

| Model | Macro-F1 (mean +/- SD over seeds) | Accuracy | Optimistic macro-F1 | Shuffled labels: macro-F1, mean |
|---|---:|---:|---:|---:|
| `mean` (baseline) | 0.518 +/- 0.034 | 0.589 | 0.676 | 0.146 |
| `thirds` | 0.620 +/- 0.015 | 0.671 | 0.763 | 0.155 |
| `random-thirds` | 0.526 +/- 0.012 | 0.602 | 0.666 | 0.142 |
| `gru` | 0.345 +/- 0.008 | 0.439 | 0.400 | 0.171 |
| Majority guess | 0.065 | 0.195 | - | - |

| Validation match | Clips | `mean` | `thirds` | `random-thirds` | `gru` |
|---|---:|---:|---:|---:|---:|
| Spirit vs Falcons | 21 | 0.578 | 0.643 | 0.577 | 0.376 |
| Furia vs Falcons | 22 | 0.565 | 0.625 | 0.535 | 0.353 |
| MOUZ vs BetBoom | 20 | 0.429 | 0.713 | 0.534 | 0.286 |
| Aurora vs Vitality | 19 | 0.500 | 0.500 | 0.460 | 0.363 |

| Class | Clips | `mean` | `thirds` | `random-thirds` | `gru` |
|---|---:|---:|---:|---:|---:|
| A_DIRECT | 15 | 0.451 | 0.627 | 0.474 | 0.478 |
| B_DIRECT | 24 | 0.638 | 0.677 | 0.632 | 0.550 |
| A_SPLIT | 14 | 0.369 | 0.492 | 0.451 | 0.131 |
| B_SPLIT | 9 | 0.420 | 0.489 | 0.361 | 0.000 |
| RETAKE | 20 | 0.711 | 0.817 | 0.714 | 0.564 |

Per-match and per-class values are macro-F1 or F1, averaged over seeds (and over matches
for classes). No shuffled refit, out of 20 per model, reached its model's real score.

**The thirds gain comes from frame order, not from the extra inputs.** `random-thirds`
has the same inputs and head as `thirds` and scores like `mean` (0.526 against 0.518;
ahead in only 6 of 12 match-and-seed pairs). `thirds` beats `random-thirds` on all four
matches and in 11 of 12 pairs, by 0.094 on average.

**Most of the gain over the baseline comes from one match.** `thirds` is 0.102 ahead of
`mean` on average, but 0.285 of that is on MOUZ vs BetBoom, whose seed spread is the
largest (+/- 0.150). Without MOUZ the gain is 0.042, and on Aurora vs Vitality there is
none. Against `random-thirds`, the gain without MOUZ is 0.065. With four matches this is
a consistent sign that order helps, not a settled size of the gain.

**The GRU is below the baseline on every match** and almost never predicts a split
correctly. It is one untuned configuration on 82 clips. Its optimistic score, with the
epoch picked on the scored match itself, is also about 0.40, so the epoch choice is not
what holds it back: no stopping point on these runs reaches the baseline. Its training
loss falls to almost zero within about 20 epochs, which points to too many weights for
the data rather than to the protocol.

## Position features

`--inputs positions` trains the same models on player positions from the minimap detector
instead of image features. The detector is the one checked in the
[detector check](detector-check.md); it passed on all four training matches.

```powershell
python -m stratscout tactic-temporal --manifest data/processed/datasets/2026-11-16/train/manifest.csv --output runs/tactic-positions --exclude-uncertain --inputs positions --shuffles 20
```

Each sampled frame becomes 260 numbers, saved in `position-features.pt`:

| Part | Numbers | Meaning |
|---|---:|---|
| T observed | 64 | Visible T icons in each cell of an 8x8 grid over the radar crop |
| CT observed | 64 | Visible CT icons per cell |
| T inferred carrier | 64 | 1 where the detector infers a hidden bomb carrier under the badge |
| Bomb badge | 64 | 1 where the bomb badge is, carried, dropped or planted |
| Totals | 4 | T seen, CT seen, inferred carrier present, bomb badge present |

- The frames, `timestamps_s` and `repeat_mask` are the loader's, so the cache lines up
  with `frame-features.pt`. The detector reads the clips at native size, with the empty-map
  background taken from each clip's source match video.
- There is no tracker identity, and a missed player is not filled in: it lowers a count.
- The inferred carrier is never added to the T counts. It can misfire on a planted bomb,
  so it has its own channel.
- Model settings are unchanged; only the input is narrower (260 numbers instead of 512).

## What you get

| File | Contents |
|---|---|
| `report.md`, `summary.json` | The models side by side: overall, per match, per class, shuffled-label check |
| `frame-features.pt`, `encoder.pt`, `samples.json` | The shared cache, encoder and sampled times |
| `mean/`, `thirds/`, `gru/` | One folder per model with the [baseline's files](tactic-baseline.md#what-you-get); `random-thirds/` when run |
| `<model>/shuffled-labels.json` | Each shuffled refit's score, when `--shuffles` is used |

## Results with position features (2026-11-16)

Run: `runs/tactic-positions-20261117`; bomb-badge ablation:
`runs/tactic-positions-no-bomb-20261117` (planned before the scores were seen, no shuffles).
Same clips, folds, epoch choice, seeds and settings as the image run. One planned run;
nothing was changed after seeing the scores.

| Model | Images: macro-F1 | Positions: macro-F1 | Images: accuracy | Positions: accuracy | Positions minus images, pairs ahead |
|---|---:|---:|---:|---:|---:|
| `mean` | 0.518 +/- 0.034 | 0.595 +/- 0.003 | 0.589 | 0.646 | +0.077, 10 of 12 |
| `thirds` | 0.620 +/- 0.015 | 0.557 +/- 0.014 | 0.671 | 0.643 | -0.063, 4 of 12 |
| `gru` | 0.345 +/- 0.008 | 0.384 +/- 0.040 | 0.439 | 0.446 | +0.039, 6 of 12 |

| Validation match | `mean` images / positions | `thirds` images / positions | `gru` images / positions |
|---|---|---|---|
| Spirit vs Falcons | 0.578 / 0.655 | 0.643 / 0.517 | 0.376 / 0.350 |
| Furia vs Falcons | 0.565 / 0.537 | 0.625 / 0.628 | 0.353 / 0.414 |
| MOUZ vs BetBoom | 0.429 / 0.541 | 0.713 / 0.523 | 0.286 / 0.453 |
| Aurora vs Vitality | 0.500 / 0.647 | 0.500 / 0.561 | 0.363 / 0.318 |

| Class | Clips | `mean` images / positions | `thirds` images / positions | `gru` images / positions |
|---|---:|---|---|---|
| A_DIRECT | 15 | 0.451 / 0.640 | 0.627 / 0.654 | 0.478 / 0.412 |
| B_DIRECT | 24 | 0.638 / 0.682 | 0.677 / 0.720 | 0.550 / 0.574 |
| A_SPLIT | 14 | 0.369 / 0.564 | 0.492 / 0.452 | 0.131 / 0.254 |
| B_SPLIT | 9 | 0.420 / 0.390 | 0.489 / 0.289 | 0.000 / 0.159 |
| RETAKE | 20 | 0.711 / 0.698 | 0.817 / 0.671 | 0.564 / 0.521 |

No shuffled refit (20 per model) reached its real score; shuffled position scores averaged
about 0.17.

**What it shows.**

- **`mean` gains most:** positions beat images for the plain average, on three of four
  matches and in 10 of 12 pairs. The gain is on A_DIRECT and A_SPLIT: A splits predicted
  as A direct fall from 21 to 12, and the reverse from 19 to 12. B_SPLIT does not improve
  (0.42 to 0.39), and RETAKE stays level.
- **`thirds` loses its lead:** on positions it is below `mean`, and below its own
  image-feature score on Spirit and MOUZ. B_SPLIT falls to 0.29. So "order helps" was
  shown for image features only; with positions it does not hold on these runs.
- **The GRU is still far below the others** with either input.
- **The bomb badge adds little:**

  | Model | With bomb badge | Without | Difference |
  |---|---:|---:|---:|
  | `mean` | 0.595 | 0.591 | +0.004 |
  | `thirds` | 0.557 | 0.532 | +0.025 |
  | `gru` | 0.384 | 0.370 | +0.014 |

  Its clearest effect is on RETAKE (about +0.1 F1 for every model), as expected from a
  planted badge.
- **The linear heads did not underfit:** their final training loss is well below the
  1.61 of a model that learned nothing (`mean` 0.35, `thirds` 0.22 on positions; 0.63
  and 0.24 on images).

With four matches and one run each, a 0.06-0.08 gap moves between matches and is only a
sign, not a settled ranking. The detector gaps (about 1 frame in 9 misses a player; about
1 T position in 6 is inferred) are in these inputs.

## Limits

There are 83 clips from four matches. A small gap between a model and the baseline is
within noise: changing only the seed, or the validation match, moves the score by more.
The report lists both spreads beside the results.
The models share the baseline's [limits](tactic-baseline.md#limits): these are
development scores on the training matches, not test results.
The GRU has about 280 weights per training clip, so it can memorise the training fold.
