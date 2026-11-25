"""Shared clip data: one model-ready package per tactical clip, for any strategy model.

Two upstream sources are combined on one real-time timeline (every `step`-th frame of a
10 fps clip, so 5 rows per second):

* the spectator-radar player and bomb detector (`live_preview.detect`: icon splitting,
  death markers and the C4 badge), read from the detection cache that train-tactics fills;
* frozen ImageNet ResNet-18 frame features, as in the tactic baseline (`tactic_model`),
  computed here on the same frames as the detector rows.

Detector rows are turned into named channels: players per map region (mirage_regions.json),
whether Ts have reached each region, region flow, team counts/centroids/spread, distances
to the sites, the bomb state from `BombStateTracker` and an estimated plant, elapsed time
and quality flags. A per-clip summary row is derived from them. ResNet features are stored
raw (512 per row); compress them per cross-validation fold with ResnetCompressor, fitted
on training clips only, so held-out matches never shape the features.

Labels are copied into the index for convenience but never used to build any feature.
Importing this module reads nothing.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TACTIC_CLASSES
from .storage import write_new_text

REGIONS_PATH = Path(__file__).with_name("mirage_regions.json")
DATA_VERSION = "clip-data-1"
NEVER_S = 60.0  # "first reached" time used when a region is never reached
BOMB_STATES = ("carried", "alone", "dropped", "missing")


# --------------------------------------------------------------------------- regions


@dataclass
class Regions:
    version: str
    names: Tuple[str, ...]
    rects: Tuple[np.ndarray, ...]  # per region: (k, 4) x0, y0, x1, y1
    site_a: np.ndarray
    site_b: np.ndarray

    @classmethod
    def load(cls, path=REGIONS_PATH) -> "Regions":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=raw["version"],
            names=tuple(r["name"] for r in raw["regions"]),
            rects=tuple(np.asarray(r["rects"], np.float32).reshape(-1, 4) for r in raw["regions"]),
            site_a=np.asarray(raw["site_centres"]["a"], np.float32),
            site_b=np.asarray(raw["site_centres"]["b"], np.float32),
        )

    def index(self, xy: np.ndarray) -> np.ndarray:
        """(n, 2) normalised points -> region index per point (first match), -1 for none."""
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        out = np.full(len(xy), -1, int)
        for i, rects in enumerate(self.rects):
            hit = np.zeros(len(xy), bool)
            for x0, y0, x1, y1 in rects:
                hit |= (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            out[(out == -1) & hit] = i
        return out


# --------------------------------------------------------------------------- detector channels


def bomb_states(bomb: np.ndarray, fps: float, size: Tuple[int, int]) -> np.ndarray:
    """Per frame index into BOMB_STATES, replaying the detector's BombStateTracker on the
    cached badge positions (normalised; size = crop width, height in pixels)."""
    from .bomb import BombStateTracker

    tracker = BombStateTracker(fps)
    w, h = size
    out = np.zeros(len(bomb), int)
    for k, (present, x, y, carried) in enumerate(bomb):
        if present:
            state = tracker.update(float(x) * w, float(y) * h, bool(carried))
        else:
            state = tracker.update(None, None, False)
        out[k] = BOMB_STATES.index(state)
    return out


def detector_channel_names(regions: Regions) -> List[str]:
    r = regions.names
    return (
        [f"t_in_{n}" for n in r]
        + [f"ct_in_{n}" for n in r]
        + [f"t_reached_{n}" for n in r]
        + [f"t_flow_{n}" for n in r]
        + [f"ct_flow_{n}" for n in r]
        + ["t_seen", "ct_seen", "t_cx", "t_cy", "ct_cx", "ct_cy", "t_spread", "ct_spread"]
        + ["t_dist_a", "t_dist_b"]
        + ["bomb_x", "bomb_y"]
        + [f"bomb_{s}" for s in BOMB_STATES]
        + ["bomb_planted_est", "bomb_dist_a", "bomb_dist_b"]
        + ["t_clip"]
        + ["carrier_inferred", "t_unplaced"]
    )


def _smooth3(a: np.ndarray) -> np.ndarray:
    """Median over each row and its neighbours in time, to stop one-frame flickers."""
    if len(a) < 3:
        return a.copy()
    padded = np.concatenate([a[:1], a, a[-1:]], 0)
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), 0)


def detector_channels(
    det: Dict[str, np.ndarray],
    regions: Regions,
    size: Tuple[int, int],
    step: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Cached detections -> (frames used, X of shape (rows, channels), channel names).
    Uses every step-th frame from frame 0, the same rows the ResNet features use."""
    players, bomb, fps = det["players"], det["bomb"], float(det["fps"])
    frames = np.arange(0, len(bomb), step)
    names = detector_channel_names(regions)
    n_r = len(regions.names)
    x = np.zeros((len(frames), len(names)), np.float32)
    region_of = regions.index(players[:, 1:3]) if len(players) else np.zeros(0, int)
    states = bomb_states(bomb, fps, size)
    col = {n: i for i, n in enumerate(names)}
    for i, f in enumerate(frames):
        here = players[:, 0] == f
        for team, prefix in ((0, "t"), (1, "ct")):
            sel = here & (players[:, 3] == team)
            pts = players[sel, 1:3]
            counts = np.bincount(region_of[sel][region_of[sel] >= 0], minlength=n_r)
            x[
                i,
                col[f"{prefix}_in_{regions.names[0]}"] : col[f"{prefix}_in_{regions.names[0]}"]
                + n_r,
            ] = counts / 5.0
            x[i, col[f"{prefix}_seen"]] = len(pts) / 5.0
            if len(pts):
                c = pts.mean(0)
                x[i, col[f"{prefix}_cx"]], x[i, col[f"{prefix}_cy"]] = c
                x[i, col[f"{prefix}_spread"]] = float(np.sqrt(((pts - c) ** 2).sum(1).mean()))
                if team == 0:
                    x[i, col["t_dist_a"]] = float(np.linalg.norm(c - regions.site_a))
                    x[i, col["t_dist_b"]] = float(np.linalg.norm(c - regions.site_b))
                    x[i, col["t_unplaced"]] = float(np.mean(region_of[sel] < 0))
        x[i, col["carrier_inferred"]] = float(np.any(here & (players[:, 4] > 0)))
        present, bx, by, _ = bomb[f]
        state = states[f]
        x[i, col[f"bomb_{BOMB_STATES[state]}"]] = 1.0
        if present:
            x[i, col["bomb_x"]], x[i, col["bomb_y"]] = bx, by
            xy = np.array([bx, by], np.float32)
            x[i, col["bomb_dist_a"]] = float(np.linalg.norm(xy - regions.site_a))
            x[i, col["bomb_dist_b"]] = float(np.linalg.norm(xy - regions.site_b))
        x[i, col["t_clip"]] = f / fps / 30.0
    # Derived over time: smoothed T presence -> reached yet; region flow per team.
    t0, ct0 = col[f"t_in_{regions.names[0]}"], col[f"ct_in_{regions.names[0]}"]
    t_in = _smooth3(x[:, t0 : t0 + n_r])
    ct_in = _smooth3(x[:, ct0 : ct0 + n_r])
    r0 = col[f"t_reached_{regions.names[0]}"]
    x[:, r0 : r0 + n_r] = np.maximum.accumulate(t_in > 0, axis=0)
    for prefix, smooth in (("t", t_in), ("ct", ct_in)):
        f0 = col[f"{prefix}_flow_{regions.names[0]}"]
        x[1:, f0 : f0 + n_r] = np.diff(smooth, axis=0)
    x[:, col["bomb_planted_est"]] = planted_estimate(bomb, regions, fps)[frames]
    return frames, x, names


def planted_estimate(
    bomb: np.ndarray, regions: Regions, fps: float, window_s: float = 3.0, seen_s: float = 1.0
) -> np.ndarray:
    """Per frame 0/1: from the first frame where, within the last window_s, the badge was
    seen uncarried on a site for at least seen_s in total, then stays 1. Counting frames
    in a window tolerates a planted icon that flickers in and out of detection, which
    keeps resetting BombStateTracker's stillness timer. A bomb dropped on a site also
    counts; the badge alone cannot tell them apart."""
    site_ids = [regions.names.index(s) for s in ("a_site", "b_site") if s in regions.names]
    if not len(bomb) or not site_ids:
        return np.zeros(len(bomb), np.float32)
    ok = (bomb[:, 0] > 0) & (bomb[:, 3] == 0) & np.isin(regions.index(bomb[:, 1:3]), site_ids)
    window = max(int(round(window_s * fps)), 1)
    recent = np.convolve(ok.astype(int), np.ones(window, int))[: len(ok)]
    return np.maximum.accumulate(recent >= int(round(seen_s * fps))).astype(np.float32)


# --------------------------------------------------------------------------- summary row


def summary_names(regions: Regions) -> List[str]:
    r = regions.names
    return (
        [f"first_t_{n}_s" for n in r]
        + [f"max_t_{n}" for n in r]
        + ["a_split_gap_s", "b_split_gap_s"]
        + ["planted_est", "planted_a", "planted_b", "plant_time_s"]
        + ["t_spread_mean", "t_spread_max", "t_path_length"]
        + ["duration_s"]
    )


def clip_summary(x: np.ndarray, names: Sequence[str], regions: Regions, rows_per_s: float):
    """One row per clip from the detector channels. Times are seconds from clip start;
    a region never reached gets NEVER_S, and a split gap is NEVER_S if either approach
    is never used."""
    col = {n: i for i, n in enumerate(names)}
    r = regions.names
    out: Dict[str, float] = {}
    first = {}
    for n in r:
        reached = np.nonzero(x[:, col[f"t_reached_{n}"]] > 0)[0]
        first[n] = reached[0] / rows_per_s if len(reached) else NEVER_S
        out[f"first_t_{n}_s"] = first[n]
    for n in r:
        out[f"max_t_{n}"] = float(x[:, col[f"t_in_{n}"]].max()) if len(x) else 0.0

    def gap(a: float, b: float) -> float:
        return abs(a - b) if a < NEVER_S and b < NEVER_S else NEVER_S

    a_main = min(first.get("a_ramp", NEVER_S), first.get("palace", NEVER_S))
    out["a_split_gap_s"] = gap(a_main, first.get("connector", NEVER_S))
    out["b_split_gap_s"] = gap(first.get("apartments", NEVER_S), first.get("short", NEVER_S))
    planted = np.nonzero(x[:, col["bomb_planted_est"]] > 0)[0]
    out["planted_est"] = float(len(planted) > 0)
    site = regions.index(x[planted[:1]][:, [col["bomb_x"], col["bomb_y"]]]) if len(planted) else []
    out["planted_a"] = float(len(site) and regions.names[site[0]] == "a_site")
    out["planted_b"] = float(len(site) and regions.names[site[0]] == "b_site")
    out["plant_time_s"] = planted[0] / rows_per_s if len(planted) else NEVER_S
    seen = x[:, col["t_seen"]] > 0
    spread = x[seen, col["t_spread"]]
    out["t_spread_mean"] = float(spread.mean()) if len(spread) else 0.0
    out["t_spread_max"] = float(spread.max()) if len(spread) else 0.0
    cxy = x[seen][:, [col["t_cx"], col["t_cy"]]]
    out["t_path_length"] = (
        float(np.linalg.norm(np.diff(cxy, axis=0), axis=1).sum()) if len(cxy) > 1 else 0.0
    )
    out["duration_s"] = len(x) / rows_per_s
    s_names = summary_names(regions)
    return np.array([out[n] for n in s_names], np.float32), s_names


# --------------------------------------------------------------------------- ResNet features


def resnet_encoder(device: str = "cpu"):
    """The tactic baseline's frozen ImageNet ResNet-18 without its final layer: an
    (n, 3, 224, 224) normalised batch -> (n, 512). Weights download on first use."""
    from .context import RunContext
    from .tactic_model import build_baseline

    encoder = build_baseline(context=RunContext(device=device)).encoder.eval()

    def encode(batch):
        import torch

        with torch.no_grad():
            return encoder(batch).cpu().numpy()

    return encode


def resnet_rows(
    path: Path,
    frames: np.ndarray,
    encode: Callable,
    *,
    size: int = 224,
    batch: int = 64,
    device: str = "cpu",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """ResNet features for the given frame indices of one clip -> ((rows, 512) float16,
    crop (width, height)). Frames are resized whole to size x size with INTER_AREA and
    ImageNet-normalised, as in the tactic baseline."""
    import cv2

    from .context import RunContext
    from .tactic_model import normalise_frames

    context = RunContext(device=device)
    wanted = set(int(f) for f in frames)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    crop = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    images, out, k = [], [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if k in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
                if len(images) == batch:
                    out.append(encode(normalise_frames(np.stack(images), context=context)))
                    images = []
            k += 1
    finally:
        cap.release()
    if images:
        out.append(encode(normalise_frames(np.stack(images), context=context)))
    feats = np.concatenate(out, 0) if out else np.zeros((0, 512), np.float32)
    if len(feats) != len(frames):
        raise ValueError(f"{path}: read {len(feats)} of {len(frames)} frames")
    return feats.astype(np.float16), crop


class ResnetCompressor:
    """PCA of ResNet rows to k numbers. Fit on the training clips of one fold only."""

    def __init__(self, k: int = 32):
        self.k = k
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None

    def fit(self, arrays: Sequence[np.ndarray]) -> "ResnetCompressor":
        stacked = np.concatenate([np.asarray(a, np.float32) for a in arrays], 0)
        self.mean = stacked.mean(0)
        _, _, vt = np.linalg.svd(stacked - self.mean, full_matrices=False)
        self.components = vt[: self.k]
        return self

    def __call__(self, a: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise ValueError("fit the compressor on training clips first")
        return ((np.asarray(a, np.float32) - self.mean) @ self.components.T).astype(np.float32)


def assemble(clip: Dict, compressor: Optional[ResnetCompressor] = None):
    """A loaded clip -> (X rows x channels, names, S, summary names). With a fitted
    compressor the k ResNet numbers are appended to every row and their clip mean to S."""
    x, names = clip["detector"], list(clip["detector_names"])
    s, s_names = clip["summary"], list(clip["summary_names"])
    if compressor is not None:
        if clip.get("resnet") is None:
            raise ValueError(f"{clip['clip_id']} was built without ResNet features")
        r = compressor(clip["resnet"])
        x = np.concatenate([x, r], 1)
        names += [f"resnet_pc{i + 1}" for i in range(r.shape[1])]
        s = np.concatenate([s, r.mean(0)])
        s_names += [f"resnet_pc{i + 1}_mean" for i in range(r.shape[1])]
    return x, names, s, s_names


def pad(sequences: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Variable-length (rows, channels) arrays -> X (clips, max_rows, channels) and mask."""
    longest = max(len(s) for s in sequences)
    x = np.zeros((len(sequences), longest, sequences[0].shape[1]), np.float32)
    mask = np.zeros((len(sequences), longest), np.float32)
    for i, s in enumerate(sequences):
        x[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return x, mask


# --------------------------------------------------------------------------- build and load


def build_clip_data(
    dataset,
    output,
    *,
    cache_dir="runs/tactic-cache",
    with_resnet: bool = True,
    regions_path=REGIONS_PATH,
    step: int = 2,
    device: str = "cpu",
    log=print,
    rows: Optional[List[Dict]] = None,
    dets: Optional[Dict] = None,
    encode: Optional[Callable] = None,
    crop_size: Tuple[int, int] = (494, 500),
) -> Dict:
    """Write <output>/clips/<clip_id>.npz, index.csv, channels.json and report.md for
    every clip in an export-segments dataset (both splits, uncertain clips flagged, not
    dropped). rows/dets/encode let tests skip video reading."""
    from .tactic_features import cached_detections, load_dataset

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output folder")
    rows = load_dataset(dataset) if rows is None else rows
    regions = Regions.load(regions_path)
    if dets is None:
        log(f"Detections for {len(rows)} clips (cache: {cache_dir}; new clips are detected)")
        dets = cached_detections(rows, cache_dir, log=log)
    if with_resnet and encode is None:
        log("Loading the frozen ResNet-18 (ImageNet weights)")
        encode = resnet_encoder(device)
    (output / "clips").mkdir(parents=True)
    index, stats = [], []
    for i, row in enumerate(rows, 1):
        det = dets[row["clip_id"]]
        frames = np.arange(0, len(det["bomb"]), step)
        resnet, size = None, _crop_size(row.get("path"), crop_size)
        if with_resnet:
            log(f"  {i}/{len(rows)} ResNet + detector channels: {row['clip_id']}")
            resnet, size = resnet_rows(row["path"], frames, encode, device=device)
        frames, x, names = detector_channels(det, regions, size, step)
        rows_per_s = float(det["fps"]) / step
        s, s_names = clip_summary(x, names, regions, rows_per_s)
        arrays = {
            "frames": frames,
            "t_s": (frames / float(det["fps"])).astype(np.float32),
            "detector": x,
            "summary": s,
        }
        if resnet is not None:
            arrays["resnet"] = resnet
        np.savez_compressed(output / "clips" / f"{row['clip_id']}.npz", **arrays)
        index.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "match": row["match"],
                "label": row["label"],
                "uncertain": bool(row.get("uncertain", False)),
                "rows": len(x),
                "duration_s": row["duration_s"],
            }
        )
        stats.append((row, x, names, s, s_names))
    channels = {
        "data_version": DATA_VERSION,
        "regions_version": regions.version,
        "regions": list(regions.names),
        "rows_per_second": float(dets[rows[0]["clip_id"]]["fps"]) / step,
        "detector": names,
        "summary": s_names,
        "resnet": {"stored": with_resnet, "raw_dim": 512, "compress": "ResnetCompressor per fold"},
        "never_s": NEVER_S,
    }
    write_new_text(output / "channels.json", json.dumps(channels, indent=2) + "\n")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(index[0]))
    writer.writeheader()
    writer.writerows(index)
    write_new_text(output / "index.csv", buffer.getvalue())
    report = clip_data_report(stats, regions, dataset, with_resnet)
    write_new_text(output / "report.md", report)
    log(f"Clip data: {output} ({len(index)} clips, {len(names)} detector channels per row)")
    return {"clips": len(index), "channels": channels}


def _crop_size(path, default: Tuple[int, int]) -> Tuple[int, int]:
    """The clip's frame (width, height), or default when there is no readable file."""
    if path is None or not Path(path).exists():
        return default
    import cv2

    cap = cv2.VideoCapture(str(path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size if min(size) > 0 else default


def load_clip_data(folder) -> List[Dict]:
    """Every clip in a build_clip_data folder, as dicts of index fields and arrays."""
    folder = Path(folder)
    channels = json.loads((folder / "channels.json").read_text(encoding="utf-8"))
    with (folder / "index.csv").open(encoding="utf-8", newline="") as handle:
        index = list(csv.DictReader(handle))
    clips = []
    for row in index:
        with np.load(folder / "clips" / f"{row['clip_id']}.npz") as data:
            arrays = {k: data[k] for k in data.files}
        clips.append(
            {
                **row,
                "uncertain": row["uncertain"] == "True",
                "y": TACTIC_CLASSES.index(row["label"]),
                "detector_names": channels["detector"],
                "summary_names": channels["summary"],
                "resnet": arrays.get("resnet"),
                **{k: v for k, v in arrays.items() if k != "resnet"},
            }
        )
    return clips


def clip_data_report(stats, regions: Regions, dataset, with_resnet: bool) -> str:
    """Plain checks of what the data says, per class. Labels are used here only to
    describe the data, never to build it."""
    lines = [
        "# Clip data",
        "",
        f"Dataset: `{dataset}`. Regions: `{regions.version}`. ResNet features stored: "
        f"{'yes' if with_resnet else 'no'}. Built by `{DATA_VERSION}`.",
        "",
        "## Region coverage",
        "",
        "Share of player detections that fall in a named region (the rest are unplaced).",
        "",
    ]
    placed = [1 - float(x[:, names.index("t_unplaced")].mean()) for _, x, names, _, _ in stats]
    lines.append(f"T detections placed, mean over clips: {np.mean(placed):.1%}.")
    lines += [
        "",
        "## Does the data separate the classes? (training split only)",
        "",
        "Share of clips where Ts reach a region, the median split gap when both approaches "
        "are used, and the share with an estimated plant.",
        "",
        "| Class | Clips | reach connector | reach ramp/palace | reach short | reach apartments "
        "| A gap s (median) | B gap s (median) | plant est. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in TACTIC_CLASSES:
        group = [
            (s, sn) for row, _, _, s, sn in stats if row["split"] == "train" and row["label"] == c
        ]
        if not group:
            continue
        sn = group[0][1]
        s = np.stack([g[0] for g in group])

        def reached(*names):
            cols = [sn.index(f"first_t_{n}_s") for n in names]
            return float(np.mean(np.min(s[:, cols], 1) < NEVER_S))

        def median_gap(name):
            g = s[:, sn.index(name)]
            g = g[g < NEVER_S]
            return f"{np.median(g):.1f}" if len(g) else "–"

        lines.append(
            f"| {c} | {len(group)} | {reached('connector'):.0%} | {reached('a_ramp', 'palace'):.0%} "
            f"| {reached('short'):.0%} | {reached('apartments'):.0%} | {median_gap('a_split_gap_s')} "
            f"| {median_gap('b_split_gap_s')} | {np.mean(s[:, sn.index('planted_est')]):.0%} |"
        )
    lines += [
        "",
        "Regions and the plant are estimates from an unvalidated colour detector and a draft "
        "region map; see docs/clip-data.md.",
    ]
    return "\n".join(lines) + "\n"
