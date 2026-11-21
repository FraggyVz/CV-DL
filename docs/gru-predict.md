# Predicting tactics from clips (web app guide)

How to call the tactic models from Python, for example from the web app. The code is in
`stratscout/gru_predict.py`; the command-line versions are `predict-tcn` and
`predict-gru`.

## Which model

| Model | Role | Called with |
|---|---|---|
| TCN | **The selected model and the default.** Show its answer. | `models=("tcn",)` (default) |
| Compact pooled GRU | Optional second opinion, or a backup if the TCN is unavailable. | `models=("gru",)` or `("tcn", "gru")` |

Asking for the GRU as well never changes the TCN's result. Both models read the same clip
features, computed once per call, and each applies its own saved ResNet compression and
scaling. Each model averages its three saved seeds.

## What a clip must be

- The **spectator-radar crop** (the minimap only), as `process-recordings` prepares it.
- **10 fps**. Another rate still runs but adds a warning, because the models were trained
  on 10 fps clips.
- **One tactic**, cut from its start to its end. `export-segments` clips are ready to use.
- An `.mp4` file. Names must be unique within one call.

## Model files

| Model | Folder | Files |
|---|---|---|
| TCN | `runs/tactic-tcn-clipdata-final/models` | `tcn-seed0.pt` to `tcn-seed2.pt`, and `../report.json` (regions version) |
| GRU | `runs/tactic-gru-clipdata-final-20261121/models` | `gru-seed0.pt` to `gru-seed2.pt` |

The TCN's folder `runs/tactic-tcn-clipdata-final` is tracked in git, so a clone already
has the TCN models. The GRU's folder is not yet: copy
`runs/tactic-gru-clipdata-final-20261121` to the same path in your checkout, or pass
other folders with `tcn_models=` and `gru_models=`. Both models were trained with map regions `mirage-spectator-v1`
(`stratscout/mirage_regions.json`). Loading refuses a regions file with another version
or channel layout, and a GRU folder whose seeds have different settings.

Models are loaded once and cached between calls; a cached call takes about 3 ms. The
cache reloads if the checkpoint files change. `gru_predict.clear_model_cache()` empties it.

## Packages

Python 3.13 and `requirements.txt`: `torch` and `torchvision` (CPU wheels are enough),
`numpy` and `opencv-python`. On first use, torchvision downloads the ImageNet ResNet-18
weights (`resnet18-f37072fd.pth`, about 45 MB) into the torch cache. Pre-download them
on a server with no internet access.

## Background (needed for a single clip)

The detector compares each frame with an image of the **empty map**. In training, that
image was the median of frames from all of a match's clips. One uploaded clip is not
enough to build it: players who stand still look like part of the map. Choose one of
three options:

1. **A saved background (recommended for uploads).** Build it once per broadcast layout
   from the training clips of a match with the same layout, then pass it to every call:

   ```powershell
   python -m stratscout save-background --dataset data/processed/datasets/2026-11-16-train-only --match aurora --output runs/backgrounds/esl-pro-league.png
   ```

   Only `train/manifest.csv` is read. `--match` takes a training match folder or a unique
   part of its name. The image is the same median of 8 frames per clip that training
   used. On a training clip, detection with it gives exactly the training detections.
   Pass a path or an RGB `uint8` array of shape (height, width, 3). It must be the same
   pixel size as the clips, or the call refuses it.

   | Training match | Event (broadcast layout) |
   |---|---|
   | `aurora-vitality-mirage-esl-pro-league-season-24` | ESL Pro League Season 24 |
   | `furia-falcons-mirage-cologne-major-2026` | Cologne Major 2026 |
   | `spirit-falcons-mirage-cologne-major` | Cologne Major |
   | `mouz-betboom-mirage-iem-bejing-2026` | IEM Beijing 2026 |

   A layout not in this list has no training background. Use option 2 for it.
2. **`background_from=[videos]`**: build the background from other videos of the same
   match, ideally the prepared match video (`minimap-10fps.mp4`) or many of its clips.
3. **Neither**: as `predict-tcn` does, clips are grouped by the part of the file name
   before the first dot, and each group builds its own background. With fewer than 5
   clips in a group, every result in it carries a warning.

## Calling it

```python
from stratscout.gru_predict import predict_clips

background = "runs/backgrounds/esl-pro-league.png"

# TCN only (the default): what the app should show.
results = predict_clips(["upload.mp4"], background=background)

# TCN and GRU: the GRU as a second opinion.
results = predict_clips(["upload.mp4"], models=("tcn", "gru"), background=background)

# GRU only: backup.
results = predict_clips(["upload.mp4"], models=("gru",), background=background)

tactic = results[0]["predictions"]["tcn"]["tactic"]
```

Other arguments:
- `min_confidence=0.6` sets the `uncertain` threshold.
- `device="cpu"` (default) or `"cuda"`.
- `log=print` receives progress lines; pass `lambda *a: None` to silence them.
- `paths` may also be a folder: every `.mp4` under it is used.

The result is a plain list with one dict per clip, ready for `json.dumps`. For example,
with both models:

```json
[
  {
    "clip": "aurora-vitality-mirage-esl-pro-league-season-24.jx.s002",
    "file": "...\\train\\clips\\RETAKE\\aurora-vitality-mirage-esl-pro-league-season-24.jx.s002.mp4",
    "warnings": [],
    "predictions": {
      "tcn": {
        "tactic": "RETAKE",
        "confidence": 0.9207,
        "uncertain": false,
        "runner_up": "B_SPLIT",
        "probabilities": {"A_DIRECT": 0.0157, "B_DIRECT": 0.017, "A_SPLIT": 0.02, "B_SPLIT": 0.0267, "RETAKE": 0.9207}
      },
      "gru": {
        "tactic": "RETAKE",
        "confidence": 0.9952,
        "uncertain": false,
        "runner_up": "B_DIRECT",
        "probabilities": {"A_DIRECT": 0.0006, "B_DIRECT": 0.0037, "A_SPLIT": 0.0002, "B_SPLIT": 0.0003, "RETAKE": 0.9952}
      }
    }
  }
]
```

(A training clip, so both models are confident. This shows the format, not accuracy.)

- `tactic` is the class with the highest seed-averaged probability.
- `confidence` is that probability.
- `uncertain` is true when the confidence is below `min_confidence`. Show such a result
  as "unsure", with `runner_up`.
- Probabilities are rounded to 4 decimals, as in `predict-tcn`'s CSV.
- `warnings` lists problems the app should show: a non-10-fps clip, or a background
  built from too few clips.

Errors raise `ValueError`, or `FileExistsError` when a CSV output already exists. Causes
include a missing file, a wrong regions version, a background of another size, and an
unknown model name.

## Time per clip (CPU)

Measured on the development laptop's CPU: three training clips of 19 to 33 s, with a
saved background and both models.

| Step | Time |
|---|---|
| Detection (every frame, colour detector) | about 11 s per clip (about 0.4 s per second of clip) |
| ResNet-18 features (every 2nd frame) | about 3 s per clip, including building the ResNet once per call (about 3 to 5 s) |
| TCN, 3 seeds | about 0.1 s per clip |
| GRU, 3 seeds | about 0.07 s per clip |
| Loading the models | 0.1 s first time, then cached |

Almost all of the time is feature extraction. Adding the GRU costs well under a second.
Run the call in a background worker rather than inside the web request.

## Command line

```powershell
python -m stratscout predict-tcn clips/ --output runs/predictions-tcn.csv
python -m stratscout predict-gru clips/ --with-tcn --background runs/backgrounds/esl-pro-league.png --output runs/predictions-both.csv
```

`predict-gru` takes `predict-tcn`'s options (`--models`, `--output`, `--background-from`,
`--min-confidence`, `--regions` and `--device`). It adds three more:
- `--background` for a saved image;
- `--with-tcn` for the TCN's rows;
- `--tcn-models` for another TCN folder.

It prints the same table, and its CSV has the same columns with a `model` column first.

The final GRU was made with:

```powershell
python -m stratscout train-gru-final --manifest data/processed/datasets/2026-11-16-train-only/train/manifest.csv --clip-data data/processed/clip-data-2026-11-16-train-only --output runs/tactic-gru-clipdata-final-20261121 --exclude-uncertain --seeds 0 1 2
```

This command:
- chose the epoch count (49) by leave-one-match-out cross-validation on the 4 training
  matches, taking the best mean macro-F1 as `train-tactics` does for the TCN;
- trained each seed on all 82 training clips for exactly 49 epochs;
- read no test clip.

## Limits

- Detections are unvalidated colour estimates (see the README). Predictions inherit
  their misses, for example stacked players or players standing on site letters.
- The GRU's cross-validation score at its chosen epoch (0.833 mean macro-F1) is
  optimistic, because the epoch was picked on the same matches. It is not a test score.
- The checks above were run on training clips. They show the code path reproduces the
  training features exactly; they do not measure accuracy on new matches.
