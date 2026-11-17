"""Frozen per-frame ResNet-18 features, extracted once and shared by every tactic model.

The cache keeps each sampled frame's feature, time and repeat flag, so a model can
pool, order or time them as it likes. Averaging a clip's frames gives exactly the
Plan A clip feature. Nothing here fits statistics or opens another split.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .context import RunContext
from .storage import save_checkpoint
from .tactic_data import TacticClipDataset
from .tactic_model import WEIGHTS, build_baseline, file_sha256, normalise_frames


def extract_frame_features(dataset: TacticClipDataset, output, *, context: RunContext) -> dict:
    """Encode every sampled frame of every clip; save the encoder as ``output/encoder.pt``.

    Returns ``features`` (clips, frames, 512) in dataset order, the per-clip ``samples``
    records and ``encoder_sha256``. Frame order is time order within each clip.
    """
    output = Path(output)
    model = build_baseline(context=context).eval()
    save_checkpoint(
        {k: v.cpu() for k, v in model.encoder.state_dict().items()}, output / "encoder.pt"
    )
    features, samples = [], []
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        tensor = normalise_frames(sample["frames"], context=context)
        features.append(model.encode_frames(tensor.unsqueeze(0)).cpu())
        samples.append(
            {
                **sample["metadata"],
                "label": record.label,
                "target": sample["target"],
                "clip_sha256": file_sha256(record.path),
                "fps": sample["fps"],
                "duration_s": sample["duration_s"],
                "frame_indices": sample["frame_indices"].tolist(),
                "timestamps_s": sample["timestamps_s"].tolist(),
                "repeat_mask": sample["repeat_mask"].tolist(),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"Extracted features: {index + 1}/{len(dataset)}", flush=True)
    return {
        "features": torch.cat(features),
        "samples": samples,
        "encoder_sha256": file_sha256(output / "encoder.pt"),
    }


def _identity(dataset: TacticClipDataset, weights=WEIGHTS) -> dict:
    """What a cache was made from. ``weights`` names the feature source: the ResNet
    weights, or the position detector (see ``tactic_positions.KIND``)."""
    return {
        "clip_ids": [r.clip_id for r in dataset.records],
        "manifest_sha256": file_sha256(dataset.manifest),
        "n_frames": dataset.n_frames,
        "size": list(dataset.size) if dataset.size else None,
        "weights": weights,
    }


def save_frame_features(path, frames: dict, dataset: TacticClipDataset, weights=WEIGHTS) -> None:
    """Write the per-frame cache with the times, repeat flags and inputs it was made from."""
    samples = frames["samples"]
    save_checkpoint(
        {
            "features": frames["features"],
            "timestamps_s": torch.tensor([s["timestamps_s"] for s in samples], dtype=torch.float64),
            "repeat_mask": torch.tensor([s["repeat_mask"] for s in samples], dtype=torch.bool),
            "samples": samples,
            "encoder_sha256": frames["encoder_sha256"],
            **({"feature_names": frames["feature_names"]} if "feature_names" in frames else {}),
            **_identity(dataset, weights),
        },
        path,
    )


def load_frame_features(path, dataset: TacticClipDataset, weights=WEIGHTS) -> dict:
    """Reuse a saved cache only if it was made from this manifest, clip order, sampling
    and feature source."""
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    for key, expected in _identity(dataset, weights).items():
        if saved.get(key) != expected:
            raise ValueError(f"{path}: cached {key} does not match this run; extract again")
    if saved["features"].shape[:2] != (len(dataset), dataset.n_frames):
        raise ValueError(f"{path}: cached features have the wrong shape; extract again")
    return saved
