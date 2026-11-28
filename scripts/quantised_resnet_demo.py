"""Compare the float ResNet-18 encoder with an int8 (quantised) copy of it.

Demo only: production code is unchanged. It
1. quantises the project's frozen ResNet-18 with post-training static quantisation,
   calibrated on frames from a few training clips (never the test clips);
2. reports encoder speed and size on one batch of 64 frames;
3. checks whether the saved TCN and GRU ensembles give the same predictions on the three
   bundled sample clips when their ResNet features come from the int8 encoder.

Run from the repository root (needs the 2026-11-16 training clips under data/):
    $env:PYTHONPATH = "."; python scripts/quantised_resnet_demo.py
"""

import io
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from stratscout import clip_data, gru_predict
from stratscout.context import RunContext
from stratscout.tactic_model import build_baseline

warnings.filterwarnings("ignore")  # torch.ao.quantization prints a deprecation notice

SAMPLES = sorted(Path("deploy/streamlit/samples").glob("*.mp4"))
TRAIN = Path("data/processed/datasets/2026-11-16/train/clips")
CALIBRATION_CLIPS = 8


def quantise(encoder: torch.nn.Module) -> torch.nn.Module:
    """Post-training static int8 quantisation, calibrated on real training frames."""
    engines = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in engines else "onednn"
    torch.backends.quantized.engine = engine
    example = torch.randn(1, 3, 224, 224)
    prepared = prepare_fx(encoder, get_default_qconfig_mapping(engine), example_inputs=(example,))

    def observe(batch):
        with torch.no_grad():
            return prepared(batch).numpy()

    clips = sorted(TRAIN.glob("*/*.mp4"))[
        :: max(1, len(list(TRAIN.glob("*/*.mp4"))) // CALIBRATION_CLIPS)
    ]
    for path in clips[:CALIBRATION_CLIPS]:
        frames = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        clip_data.resnet_rows(path, np.arange(0, frames, 2), observe)
    return convert_fx(prepared)


def as_encode(model: torch.nn.Module):
    def encode(batch):
        with torch.no_grad():
            return model(batch.cpu()).numpy()

    return encode


def size_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


def seconds(encode, batch, repeats: int = 3) -> float:
    encode(batch)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        encode(batch)
    return (time.perf_counter() - start) / repeats


def predict(label: str):
    start = time.perf_counter()
    results = gru_predict.predict_clips(SAMPLES, models=("tcn", "gru"), log=lambda *a: None)
    print(f"   {label}: three clips in {time.perf_counter() - start:.1f} s, end to end")
    return {r["clip"]: r["predictions"] for r in results}


def main() -> None:
    torch.manual_seed(0)
    float_model = build_baseline(context=RunContext(device="cpu")).encoder.eval()
    int8_model = quantise(build_baseline(context=RunContext(device="cpu")).encoder.eval())
    float_encode, int8_encode = as_encode(float_model), as_encode(int8_model)

    batch = torch.randn(64, 3, 224, 224)
    t_float, t_int8 = seconds(float_encode, batch), seconds(int8_encode, batch)
    print("\n1. Encoder alone, one batch of 64 frames")
    print(f"   float32 {t_float:.2f} s, int8 {t_int8:.2f} s ({t_float / t_int8:.1f}x faster)")
    print(f"   size: float32 {size_mb(float_model):.0f} MB, int8 {size_mb(int8_model):.0f} MB")

    print("\n2. Saved TCN and GRU on the three sample clips")
    before = predict("float32 ResNet")
    clip_data.resnet_encoder = lambda device="cpu": int8_encode  # swap the encoder
    gru_predict.clear_model_cache()
    after = predict("int8 ResNet   ")

    print()
    for clip in before:
        for model in ("tcn", "gru"):
            p, q = before[clip][model], after[clip][model]
            same = "same" if p["tactic"] == q["tactic"] else "CHANGED"
            print(
                f"   {clip.split('.')[-1]} {model.upper()}: {p['tactic']} {p['confidence']:.3f}"
                f" -> {q['tactic']} {q['confidence']:.3f} ({same})"
            )


if __name__ == "__main__":
    main()
