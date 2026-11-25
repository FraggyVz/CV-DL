"""What each stage of one round's classification produced, for the web app's model tabs.

``build_pipeline`` reruns the saved models on a segment's already-computed clip features
and records every stage on the way to the prediction: the detector's player positions
and map-region signals, the ResNet-18 PCA numbers, the scaled model input, the TCN's
residual-block activations or the GRU's projection and hidden state, the mean and max
pooling, and each seed's class probabilities. The values describe this clip's forward
pass only; they are not a feature-attribution explanation.

Time is binned to at most ``MAX_COLUMNS`` columns and values are rounded, so one segment
is a few hundred kilobytes of JSON. Importing this module loads nothing.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import TACTIC_CLASSES

FORMAT = "stratscout-web-pipeline-1"
MAX_COLUMNS = 120
MAX_POSITIONS = 900
TEAM = 5  # clip_data stores player counts as a fraction of a five-player team


def _round(values, decimals: int) -> np.ndarray:
    """Rounded as float64, so JSON gets 0.3 rather than float32's 0.30000001192092896."""
    return np.round(np.asarray(values, np.float64), decimals)


def _bins(steps: int, columns: int = MAX_COLUMNS) -> List[np.ndarray]:
    return [b for b in np.array_split(np.arange(steps), min(steps, columns)) if len(b)]


def _binned(matrix: np.ndarray, bins) -> np.ndarray:
    """(steps, channels) -> (channels, columns), each column the mean of its steps."""
    matrix = np.asarray(matrix, np.float32)
    return np.stack([matrix[b].mean(0) for b in bins], 1)


def _heatmap(matrix: np.ndarray, bins, labels: Sequence[str], decimals: int = 1) -> Dict:
    values = _binned(matrix, bins)
    return {
        "labels": list(labels),
        "values": _round(values, decimals).tolist(),
        "min": round(float(values.min()), decimals) if values.size else 0.0,
        "max": round(float(values.max()), decimals) if values.size else 0.0,
    }


def _series(values: np.ndarray, bins, decimals: int = 3) -> List[float]:
    return _round(_binned(np.asarray(values)[:, None], bins)[0], decimals).tolist()


def _memory_change(outputs: np.ndarray, bins, decimals: int = 3) -> List[float]:
    """Size (Euclidean length) of the change in the GRU's memory from each row to the next,
    averaged per time column. The first row has no earlier row, so it adds nothing; a
    column holding only that row is 0."""
    outputs = np.asarray(outputs, np.float64)
    change = np.full(len(outputs), np.nan)
    change[1:] = np.linalg.norm(np.diff(outputs, axis=0), axis=1)
    columns = [change[b][~np.isnan(change[b])] for b in bins]
    return _round([part.mean() if len(part) else 0.0 for part in columns], decimals).tolist()


def _peak_columns(peak_steps: np.ndarray, bins) -> List[int]:
    """How many channels took their maximum in each time column."""
    column_of = np.zeros(bins[-1][-1] + 1, int)
    for i, b in enumerate(bins):
        column_of[b] = i
    return np.bincount(column_of[peak_steps], minlength=len(bins)).tolist()


def _probabilities(p) -> Dict[str, float]:
    return {c: round(float(v), 4) for c, v in zip(TACTIC_CLASSES, p)}


def _radar_frame(path: Path) -> str:
    """The clip's middle frame as a JPEG data URI, for the detection overlay."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frames // 2, 0))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return ""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else ""


def _detection_stage(path: Path, detection: Dict, detector: np.ndarray, t_s, bins, regions):
    from .clip_data import detector_channel_names

    names = detector_channel_names(regions)
    column = {n: i for i, n in enumerate(names)}
    players = np.asarray(detection["players"], np.float32).reshape(-1, 5)
    if len(players) > MAX_POSITIONS:
        players = players[np.linspace(0, len(players) - 1, MAX_POSITIONS).astype(int)]
    frames = max(int(detection["n_frames"]) - 1, 1)
    positions = [
        [round(float(k) / frames, 3), round(float(x), 4), round(float(y), 4), int(team)]
        for k, x, y, team, carrier in players
        if not carrier
    ]
    region_rows = list(regions.names)
    return {
        "radar_image": _radar_frame(path),
        "positions": positions,
        "regions": [
            {"name": n, "rects": _round(r, 4).tolist()}
            for n, r in zip(regions.names, regions.rects)
        ],
        "t_regions": _heatmap(
            TEAM * detector[:, [column[f"t_in_{n}"] for n in region_rows]], bins, region_rows
        ),
        "ct_regions": _heatmap(
            TEAM * detector[:, [column[f"ct_in_{n}"] for n in region_rows]], bins, region_rows
        ),
        "t_reached": _heatmap(
            detector[:, [column[f"t_reached_{n}"] for n in region_rows]], bins, region_rows
        ),
        "counts": {
            "t_seen": _series(TEAM * detector[:, column["t_seen"]], bins, 1),
            "ct_seen": _series(TEAM * detector[:, column["ct_seen"]], bins, 1),
            "bomb_planted_est": _series(detector[:, column["bomb_planted_est"]], bins),
        },
        "channels": len(names),
        "channel_names": names,
    }


def _tcn_stage(members, features: Dict, bins) -> Dict:
    import torch

    from .tactic_tcn import _raw_features, masked_pool, pad_batch, receptive_field

    seeds = []
    record = None
    for i, (model, norm, cfg) in enumerate(members):
        raw = _raw_features(features, cfg, norm.compressor)
        scaled = norm(raw)
        x, mask = pad_batch([scaled])
        with torch.no_grad():
            h = model.inp(x) * mask
            blocks = []
            for block in model.blocks:
                h = block(h, mask)
                blocks.append(h[0].T.numpy())
            pooled = masked_pool(h, mask)
            probs = torch.softmax(model.head(pooled), -1)[0].numpy()
        seeds.append(_probabilities(probs))
        if i == 0:  # Internals are shown for the first seed; every seed has the same shape.
            channels = h.shape[1]
            pcs = 0 if norm.compressor is None else len(norm.compressor.components)
            labels = [f"detector {j + 1}" for j in range(raw.shape[1] - pcs)] + [
                f"ResNet PC {j + 1}" for j in range(pcs)
            ]
            peak = h[0].argmax(-1).numpy()
            record = {
                "resnet_pcs": _heatmap(raw[:, raw.shape[1] - pcs :], bins, labels[-pcs:])
                if pcs
                else None,
                "input": _heatmap(scaled, bins, labels),
                "input_width": int(scaled.shape[1]),
                "blocks": [
                    {
                        "dilation": int(d),
                        "activations": _heatmap(
                            b, bins, [f"channel {c + 1}" for c in range(channels)]
                        ),
                    }
                    for d, b in zip(cfg.dilations, blocks)
                ],
                "pooled": {
                    "mean": _round(pooled[0, :channels].numpy(), 3).tolist(),
                    "max": _round(pooled[0, channels:].numpy(), 3).tolist(),
                },
                "peak_columns": _peak_columns(peak, bins),
                "architecture": {
                    "channels": int(channels),
                    "kernel": int(cfg.kernel),
                    "dilations": [int(d) for d in cfg.dilations],
                    "receptive_field_steps": int(receptive_field(cfg.kernel, cfg.dilations)),
                    "receptive_field_s": round(
                        receptive_field(cfg.kernel, cfg.dilations) * cfg.step / 10.0, 1
                    ),
                    "step_s": cfg.step / 10.0,
                },
            }
    ensemble = np.mean([[s[c] for c in TACTIC_CLASSES] for s in seeds], 0)
    return {**record, "seeds": seeds, "ensemble": _probabilities(ensemble)}


def _gru_stage(members, features: Dict, bins) -> Dict:
    import torch

    from .gru_predict import gru_rows
    from .tactic_temporal import TIME_SCALE_S

    packed = gru_rows(features)
    seeds = []
    record = None
    for i, (head, info) in enumerate(members):
        head.eval()
        with torch.no_grad():
            probs = head(packed).softmax(1)[0].numpy()
            seeds.append(_probabilities(probs))
            if i:
                continue
            rows, _ = head.transform(packed)
            g = head.gru
            steps_in = torch.cat([rows, packed[:, :, head.width :]], 2)[0]
            width = g.feature_dim
            standardised = g.standardise(steps_in[:, :width])
            projected = g.project(standardised)
            outputs, _ = g.gru(torch.cat([projected, steps_in[:, width : width + 2]], 1)[None])
            outputs = outputs[0]
            mean, peak = outputs.mean(0), outputs.amax(0)
            check = g.classifier(torch.cat([mean, peak])[None]).softmax(1)[0].numpy()
        if not np.allclose(check, probs, atol=1e-4):
            raise ValueError("the GRU walkthrough does not reproduce the model's output")
        if float(outputs.abs().max()) > 1.0:
            raise ValueError("the GRU memory left the range -1 to +1")
        hidden = outputs.shape[1]
        pcs = int(head.pcs)
        detector_width = int(head.detector_width)
        labels = [f"detector {j + 1}" for j in range(detector_width)] + [
            f"ResNet PC {j + 1}" for j in range(pcs)
        ]
        settings = info.get("settings", {}) if isinstance(info, dict) else {}
        sizes = settings.get("gru", {})
        time_inputs = int(g.gru.input_size - g.project.out_features)
        scale = f"{TIME_SCALE_S:g}"
        record = {
            "input": _heatmap(rows[0].numpy(), bins, labels),
            "input_width": int(rows.shape[2]),
            "projection": _heatmap(
                projected.numpy(), bins, [f"projection {c + 1}" for c in range(projected.shape[1])]
            ),
            # sequence_inputs: seconds since the clip started, then since the previous row.
            "time_inputs": _heatmap(
                steps_in[:, width : width + time_inputs].numpy(),
                bins,
                [f"seconds since start / {scale}", f"seconds since previous row / {scale}"][
                    :time_inputs
                ],
                3,
            ),
            "hidden": _heatmap(
                outputs.numpy(), bins, [f"memory {c + 1}" for c in range(hidden)], 2
            ),
            "memory_change": _memory_change(outputs.numpy(), bins),
            "row_times_s": _round(steps_in[:, width].numpy() * TIME_SCALE_S, 1).tolist(),
            "pooled": {
                "mean": _round(mean.numpy(), 3).tolist(),
                "max": _round(peak.numpy(), 3).tolist(),
            },
            "peak_columns": _peak_columns(outputs.argmax(0).numpy(), bins),
            "architecture": {
                "projection_in": int(g.project.in_features),
                "projection": int(projected.shape[1]),
                "time_inputs": time_inputs,
                "time_scale_s": float(TIME_SCALE_S),
                "gru_input": int(g.gru.input_size),
                "hidden": int(hidden),
                "readout": str(g.readout),
                "readout_width": int(g.classifier.in_features),
                "classes": int(g.classifier.out_features),
                "settings": sizes,
                "parameters": int(sum(p.numel() for p in head.parameters())),
            },
        }
    ensemble = np.mean([[s[c] for c in TACTIC_CLASSES] for s in seeds], 0)
    return {**record, "seeds": seeds, "ensemble": _probabilities(ensemble)}


def build_pipeline(path: Path, detection: Dict, features: Dict, tcn_members, gru_members) -> Dict:
    """Every stage of one prepared segment's TCN and GRU classification (see the module
    docstring). ``features`` is gru_predict.clip_features's entry for this clip."""
    from .clip_data import REGIONS_PATH, Regions

    regions = Regions.load(REGIONS_PATH)
    detector = np.asarray(features["detector"], np.float32)
    t_s = np.asarray(features["t_s"], np.float32)
    bins = _bins(len(detector))
    return {
        "format": FORMAT,
        "classes": list(TACTIC_CLASSES),
        "steps": int(len(detector)),
        "columns": len(bins),
        "duration_s": round(float(t_s[-1]) if len(t_s) else 0.0, 1),
        "column_times_s": [round(float(t_s[b].mean()), 1) for b in bins],
        "detection": _detection_stage(path, detection, detector, t_s, bins, regions),
        "tcn": _tcn_stage(tcn_members, features, bins),
        "gru": _gru_stage(gru_members, features, bins),
    }
