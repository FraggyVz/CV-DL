"""Plan A: frozen frame features, temporal mean pooling and a five-class linear head."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TACTIC_CLASSES
from .context import RunContext

WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def normalise_frames(frames: np.ndarray, *, context: RunContext) -> torch.Tensor:
    """Convert loader RGB uint8 (T,H,W,3) to ImageNet-normalised (T,3,H,W).

    Resize the whole map in the loader first. No cropping, flipping or fitted
    preprocessing is used here. Reject floats to prevent normalising twice.
    """
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 frames with shape (T,H,W,3)")
    if any(size == 0 for size in frames.shape):
        raise ValueError("Frames must not be empty")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    tensor = tensor.to(device=context.device, dtype=torch.float32) / 255.0
    mean = tensor.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(RGB_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


class TacticBaseline(nn.Module):
    """The encoder stays in evaluation mode, even when the classifier is training.

    Inputs are normalised tensors (B,T,3,H,W). Temporal order and elapsed time
    are deliberately ignored by this baseline; metadata is never an input.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int = 512):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.classifier = nn.Linear(feature_dim, len(TACTIC_CLASSES))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # BatchNorm must not learn from either match.
        return self

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3 or min(frames.shape) == 0:
            raise ValueError("Expected non-empty normalised frames with shape (B,T,3,H,W)")
        batch, time = frames.shape[:2]
        with torch.no_grad():
            features = self.encoder(frames.flatten(0, 1))
            return features.reshape(batch, time, -1).mean(dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(frames))


def build_baseline(*, context: RunContext, pretrained: bool = True) -> TacticBaseline:
    """Download fixed ImageNet weights on first use; never silently train from scratch.

    ``pretrained=False`` is for restoring the saved encoder, without a download.
    """
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Install torchvision from requirements.txt before running Plan A"
        ) from exc
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    encoder.fc = nn.Identity()
    return TacticBaseline(encoder).to(context.device)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_baseline(checkpoint, *, context: RunContext) -> tuple[TacticBaseline, dict]:
    """Restore a head and its sibling encoder.pt offline; keep the run folder together."""
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model"] != "resnet18-mean-linear" or saved["classes"] != list(TACTIC_CLASSES):
        raise ValueError("Checkpoint model or class order does not match Plan A")
    encoder_path = checkpoint.parent / "encoder.pt"
    if file_sha256(encoder_path) != saved["encoder_sha256"]:
        raise ValueError("encoder.pt does not match this classifier checkpoint")
    model = build_baseline(context=context, pretrained=False)
    model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
    model.classifier.load_state_dict(saved["classifier"])
    return model.eval(), saved
