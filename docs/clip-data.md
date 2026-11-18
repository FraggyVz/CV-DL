# Shared clip data

One model-ready package per tactical clip, built once and read by every strategy model
(the TCN, the GRU, the tactic baseline and any new model). It combines the two upstream
sources that already exist in the project, on one timeline:

| Source | Code | Gives |
|---|---|---|
| Player and bomb detector | `live_preview.detect` → `icon_split.find_icons_split`, `bomb.find_bomb`, `bomb.BombStateTracker` | Where every T and CT icon is, and the C4 badge, on every frame |
| Frozen ResNet-18 | `tactic_model.build_baseline` (ImageNet weights, final layer removed) | 512 general image features per frame |

The models stay unchanged. This is only the data they read. The walkthrough notebook is
[`notebooks/clip-data-walkthrough.ipynb`](../notebooks/clip-data-walkthrough.ipynb).

## Build it

```powershell
.\.venv\Scripts\python.exe -m stratscout build-clip-data --dataset data/processed/datasets/2026-11-16 --output runs/clip-data-2026-11-16
```

- **Time:** about 5 s per clip, mostly ResNet. The 128 clips of 2026-11-16 take about
  11 minutes on CPU. The first run downloads the ResNet-18 weights (45 MB) once.
- **Detections** come from `runs/tactic-cache/`, the same cache `train-tactics` uses.
  A clip not in the cache is detected first.
- **`--no-resnet`** builds the detector channels only, in seconds.
- **The output folder must be new.** Choose a new name each time.

## What you get

```
runs/clip-data-2026-11-16/
  clips/<clip_id>.npz   one file per clip (arrays below)
  index.csv             clip_id, split, match, label, uncertain, rows, duration_s
  channels.json         the name of every column, region names and versions
  report.md             checks of what the data says per class
```

Each `.npz` holds:

| Array | Shape | Meaning |
|---|---|---|
| `detector` | rows × 72 | The detector channels below, one row every 0.2 s |
| `resnet` | rows × 512 | Raw ResNet-18 features of the **same** frames (float16) |
| `summary` | 30 | The whole clip in one row (below) |
| `frames`, `t_s` | rows | Which clip frame each row is, and its time in seconds |

**Rows are real time.** Every second frame of the 10 fps clip, so 5 rows per second. A
20 s clip has 100 rows and a 40 s clip 200. Nothing is stretched to a fixed length.

## Map regions

The detector gives positions. The classes are defined by **named map areas** ("ramp
and/or palace plus mid → connector = A split"), so positions are mapped to the regions in
[`stratscout/mirage_regions.json`](../stratscout/mirage_regions.json), drawn on the
spectator radar crop (494 × 500 px):

| Region | Role in the class rules |
|---|---|
| `a_ramp`, `palace` | The direct A approaches |
| `connector` | Mid → A, the second A approach (includes jungle) |
| `apartments` | The direct B approach |
| `short` | Mid → B, the second B approach: the stairs up from mid and the catwalk into B |
| `mid` | Includes top mid |
| `a_site`, `b_site` | The sites |
| `t_spawn`, `ct_side` | Context: spawn and CT rotations |

Each region is one or more rectangles in normalised crop coordinates (0–1). A point
belongs to the first region in the list that contains it. On 2026-11-16, **every**
player detection falls in a region.

**Status:** version `mirage-spectator-v1` is a **draft**. The team should check the
boundaries, especially connector/jungle, short and the edges of mid, before results
depend on them. Change the JSON, bump `version`, and rebuild.

**How v1 was drawn.** v0 was drawn from the empty-map image only. The radar is rotated
compared with the usual Mirage overview (B top-left, A bottom-centre, T spawn on the
right), and v0 put `short` on the CT side of B. The 2026-11-16 report showed B_SPLIT
clips reaching that `short` no more often than B_DIRECT (22% against 17%). A map of
where B_SPLIT Ts go and B_DIRECT Ts don't, using **training clips only**, showed the
real route from mid up the stairs and across into B. v1 moves `short` onto that route
and gives v0's area to `ct_side`. The test matches were not looked at. Region changes
are still a design choice informed by training labels, so report the version with
every result.

## Detector channels (72 per row)

| Channels | Count | Meaning |
|---|---|---|
| `t_in_<region>`, `ct_in_<region>` | 20 | Players of that team in the region ÷ 5 |
| `t_reached_<region>` | 10 | 1 from the first row a T is in the region, then stays 1 (route history) |
| `t_flow_<region>`, `ct_flow_<region>` | 20 | Change in that region's count since the last row (which way the team moves) |
| `t_seen`, `ct_seen` | 2 | Players detected ÷ 5 |
| `t_cx`, `t_cy`, `ct_cx`, `ct_cy` | 4 | Each team's average position |
| `t_spread`, `ct_spread` | 2 | How far players are from their team's average; large for a split |
| `t_dist_a`, `t_dist_b` | 2 | Distance from the T average to each site centre |
| `bomb_x`, `bomb_y` | 2 | The C4 badge, when visible |
| `bomb_carried`, `bomb_alone`, `bomb_dropped`, `bomb_missing` | 4 | The bomb state from `BombStateTracker` (one of them is 1) |
| `bomb_planted_est` | 1 | 1 from the first row where the badge has been seen uncarried on a site for at least 1 s within the last 3 s, then stays 1 |
| `bomb_dist_a`, `bomb_dist_b` | 2 | Distance from the badge to each site |
| `t_clip` | 1 | Seconds since the clip started ÷ 30 |
| `carrier_inferred`, `t_unplaced` | 2 | Quality: a hidden carrier was inferred; share of Ts outside every region |

Notes:

- **Flicker guard.** Region counts are smoothed over 3 rows (0.6 s) before `reached` and
  `flow` are computed, so one bad frame cannot mark a region as reached.
- **No identity and nothing invented.** A missed player lowers a count; it is never filled
  in. The tracker is not used.
- **The plant is an estimate.** A planted C4 icon flickers in and out of detection, and
  each gap of more than 2 frames resets `BombStateTracker`'s 2-second stillness timer.
  So on training RETAKE clips the tracker reached "dropped" in only 20%. The plant
  rule therefore counts frames in a 3 s window, which finds a plant in 75% of training
  RETAKE clips. The tracker's four states are kept as channels too. A bomb dropped on
  a site also counts as planted; the badge alone cannot tell the two apart.

## Summary row (30 per clip)

| Values | Meaning |
|---|---|
| `first_t_<region>_s` (10) | Seconds until a T first reaches the region; 60 means never |
| `max_t_<region>` (10) | Most Ts seen in the region at once ÷ 5 |
| `a_split_gap_s` | Seconds between the first T on ramp/palace and the first T in connector; 60 if one is never used |
| `b_split_gap_s` | The same for apartments and short |
| `planted_est`, `planted_a`, `planted_b`, `plant_time_s` | Estimated plant, its site and time |
| `t_spread_mean`, `t_spread_max`, `t_path_length` | How split up the Ts are, and how far their average moves |
| `duration_s` | Clip length |

The split gaps measure the tactic guide's split rule directly: final approaches starting
within about 10 s of each other. They are **features, not labels**. Models learn from them.

## ResNet features: compress per fold

The 512 raw ResNet numbers are stored as they are. Before a model uses them, compress
them to a few numbers (32 is suggested) with `ResnetCompressor`, **fitted on that fold's
training clips only**. Fitting on all clips would let the held-out match shape the
features.

```python
from stratscout.clip_data import ResnetCompressor, assemble, load_clip_data, pad

clips = load_clip_data("runs/clip-data-2026-11-16")
train = [
    c for c in clips if c["split"] == "train" and c["match"] != held_out and not c["uncertain"]
]
compressor = ResnetCompressor(32).fit([c["resnet"] for c in train])

x, names, s, s_names = assemble(train[0], compressor)  # rows x (72 + 32), 30 + 32 summary
X, mask = pad([assemble(c, compressor)[0] for c in train])  # (clips, max rows, 104) + mask
```

`assemble(clip)` without a compressor gives the detector channels only.

## Rules for models that use it

1. Same dataset, the same exclusions (`uncertain` in `index.csv`), the same test matches
   and the same leave-one-match-out folds for every model.
2. Variable-length rows plus `mask`; never stretch clips to a fixed length.
3. Fit scaling and `ResnetCompressor` on each fold's training clips only.
4. Labels are in `index.csv` for training targets only. No feature is built from them.
5. Record the `data_version` and `regions_version` from `channels.json` with results.

## Limits

- **The detector is an unvalidated colour detector.** jx's detector check measures it on
  Furia and Spirit. Stacked or hidden icons are missed, and inferred carriers are about a
  fifth of T detections.
- **Draft regions.** See above.
- **No time since round start.** The manifest gives each clip's round and start time in
  the recording, but the recordings are not on every machine. When they are, round starts
  from `rounds.py` can add it.
- **ResNet-18 was trained on everyday photos.** A 224 × 224 resize shrinks the about 18 px
  icons to about 8 px. Its features describe the whole map's appearance, not players.
- **Spatial ResNet maps (7 × 7 × 512) are not stored.** They would be about 1 GB for this
  dataset. Only the pooled 512 per row is kept.
- **Mirage, the spectator radar layout and these recordings only.**
