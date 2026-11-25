# Clip loader

Use the exported MP4 clips and `manifest.csv`; no original recording or demo is needed.
This example makes the Furia/Spirit swaps: train on one match, validate on the other.

```python
from torch.utils.data import Subset
from stratscout.context import RunContext
from stratscout.tactic_data import TacticClipDataset, whole_match_folds

dataset = TacticClipDataset(
    "data/processed/datasets/2026-11-01/train/manifest.csv",
    context=RunContext(),
    n_frames=16,
    exclude_uncertain=True,  # Explicitly omit uncertain clips for this run.
)
for fold in whole_match_folds(dataset):
    train_data = Subset(dataset, fold.train_indices)
    validation_data = Subset(dataset, fold.validation_indices)
    # Initialise a fresh model for each fold; training code is separate.
```

By default, the loader rejects uncertain boundaries, currently Spirit segment #5.
Pass `exclude_uncertain=True` to omit them with a warning, as shown above.
Report `dataset.excluded_clip_ids` with the resulting class counts for each fold.
The loader does not change the labels or exported clips.

Each sample is a dictionary:

| Key | Meaning |
|---|---|
| `frames` | RGB `uint8` frames, shape `(T, H, W, 3)`, values 0–255 |
| `target` | Class index in `dataset.classes` |
| `timestamps_s` | Sampled frame times in seconds from the start of the clip |
| `frame_indices` | Zero-based frame indices within the clip |
| `repeat_mask` | True where a short clip repeats the previous sampled frame |
| `duration_s`, `fps` | Annotated clip duration and video-header frame rate |
| `metadata` | Clip/match identity and source references; not model inputs |

The loader applies no augmentation or normalisation. It samples frames evenly and
keeps their timestamps. Models can use their own preprocessing; normalise only once.
The project's existing constraint is no flipping or mirroring: it swaps the map layout.

Keep all clips and frame-rate variants of a match together. Fit preprocessing and
class weights on each fold's training subset only. Load the test manifest with
`split="test"` only for final evaluation; test data cannot create validation folds.
Frame-count or timing mismatches raise an error rather than silently changing the input.
