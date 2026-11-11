# Plan A: first tactic model

Plan A extracts 512 ImageNet ResNet-18 features per frame, averages them across the
clip, then trains a linear layer for the five classes in the [tactic guide](tactic-definitions.md).

## Install (do this once)

1. Open a terminal in the repository root with the Python 3.13 project environment active.
2. Install the matching CPU packages:
   ```powershell
   python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
   ```
3. Install the project dependencies:
   ```powershell
   python -m pip install -r requirements-dev.txt
   ```

**Common mistake:** `torchvision` missing? Run step 2 in the same environment you use for training.

## Run it

1. Put the exported MP4 clips beside their `train/manifest.csv`; see the [clip loader](clip-loader.md).
2. Run the default cross-fold comparison:
   ```powershell
   python -m stratscout tactic-baseline --manifest data/processed/datasets/2026-11-01/train/manifest.csv --output runs/plan-a-cross-fold --exclude-uncertain
   ```
3. Open `report.md` inside that output folder.

To choose the epoch count yourself beforehand, use a separate run (40 is an example):
```powershell
python -m stratscout tactic-baseline --manifest data/processed/datasets/2026-11-01/train/manifest.csv --output runs/plan-a-fixed-40 --exclude-uncertain --fixed-epochs 40
```

**Common mistake:** an existing output folder causes an error; choose a new name.
Without `--exclude-uncertain`, uncertain boundaries cause an error. The flag omits them with a warning, currently Spirit segment #5; excluded IDs and counts are saved.
The first run downloads about 45 MB of ImageNet weights; later runs reuse that download.

| Setting | What happens |
|---|---|
| Frames | 16 evenly spaced frames; resize the whole map to 224x224, with no crop |
| Preprocessing | RGB scaled to 0-1, then fixed ImageNet mean/std once; no augmentation |
| Model | ResNet and BatchNorm stay frozen in evaluation mode; only the last layer learns |
| Training | Fresh head for each Furia/Spirit swap and seed (0, 1, 2); features reused within the run |
| Class balance | All five classes required in each training fold; weights and majority guess use training counts only |
| Optimiser | Full-batch AdamW; learning rate 0.01, weight decay 0.001 |
| Cross-fold (default, or `--cross-fold`) | Each seed uses the OTHER fold's best epoch, then refits to exactly that epoch |
| Epoch search | Up to 200 epochs, patience 30; best validation macro-F1, lower loss breaks ties |
| Fixed epochs | `--fixed-epochs N` overrides search/patience; no early stopping, score only epoch N |

## What you get

| File | Contents |
|---|---|
| `report.md`, `summary.json` | Main and optimistic scores, majority scores; mean, SD and range across seeds in JSON |
| `fold-*.json` | Epoch choices, counts/weights, per-class precision/recall/F1, accuracy, confusion matrices |
| `predictions.csv` | Main-mode validation predictions and probabilities, one per clip per seed |
| `config.json`, `input-manifest.csv`, `samples.json` | Settings, versions, exclusions, source references, hashes and sampled times |
| `features.pt` | Fixed clip features |
| `encoder.pt`, `fold-*-seed-*.pt`, `*-history.json` | Shared frozen encoder, classifier checkpoints and training histories |

Keep the run folder together. `load_baseline(checkpoint, context=RunContext())` in `stratscout.tactic_model` restores it offline and returns saved settings. Use its size/frame
count with the loader, then `normalise_frames(frames, context=context).unsqueeze(0)` for inference.

## How to read the results

1. Read the main macro-F1 column; the overall row averages folds per seed before calculating SD.
2. Compare it with majority macro-F1 and accuracy; inspect each match's per-class F1 and clip counts.
3. Treat "optimistic (selected on validation)" as a comparison: it picks and scores on the same match.
   Fixed mode does not run that comparison. Matrix rows are truth, columns predictions; class order is in `config.json`.
4. For the planned size check, use `--size 320` and a new output folder; keep other settings unchanged and report both sizes, not the best seed.

## Limits

No flipping or mirroring is an existing project constraint. There is no synthetic training.
Mean pooling ignores frame order and elapsed time; hand-cut clips already supply the boundaries.
Only two development matches: cross-fold still reuses both, because each validation match
trains the other fold's model. It is not nested validation or an independent test score.
Choose fixed N before seeing results; repeated size/epoch tuning still uses these matches.
Seed SD is not uncertainty on unseen matches. NaVi stays held out; this command accepts only
the training split. These are development classification scores, not final test results or temporal AP/mAP.
