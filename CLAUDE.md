# StratScout — project handoff

> **Current implementation update (12 October 2026):** reusable code from both
> notebooks now lives in `stratscout/`. The notebooks import named workflow steps.
> Use [README](README.md), [architecture](docs/architecture.md) and
> [AGENTS.md](AGENTS.md) for current setup and contributor instructions. The numbered
> sections, cell references and measured results below describe the original
> prototype at `00e024f`; they are historical context, not current validation.
> The companion schema mismatch and tracker/annotation faults remain. B-prime is
> disabled by default because its model-selection split leaks test data.

> Context file for Claude Code. Read this first; it carries the decisions, the measured
> numbers, and the bugs already paid for. Treat the "Hard-won lessons" section as load-bearing —
> every item there cost real debugging time and several are silent failures that look like
> working code.

---

## 1. What this is

A school computer-vision / deep-learning project (Ryan, Singapore Institute of Technology).
**StratScout** recognises Counter-Strike 2 tactics from the minimap, and nothing else.

**The one rule that defines the project:**

> At inference the system sees **pixels only**. `.dem` files may be used during development as
> ground truth, for annotation and for evaluation — **never as an input to the deployed
> pipeline**.

Scope is deliberately narrow: **one map (de_mirage)**.

### The three objectives

| # | Objective | Status |
|---|---|---|
| 1 | Identify the strategy: attacker tactic before the plant, defender response after it | pipeline built, needs real labels |
| 2 | Player tendency heatmaps — where attackers go, how defenders set up | built, nearly free from tracker output |
| 3 | Win rates — overall side advantage, and **per-tactic success rate** | built as *annotation + aggregation*, not a model |

### Two questions per round (decided 2026-10-07; replaces the old five-class list)

```python
T_CLASSES  = ["A_RUSH", "B_RUSH", "A_SPLIT", "B_SPLIT", "DEFAULT"]   # before the plant, every round
CT_CLASSES = ["RETAKE", "TRICKLE", "SAVE"]                          # after the plant, planted rounds only
PHASES = {"pre": T_CLASSES, "post": CT_CLASSES};  CLASSES = T_CLASSES  # headline head
```

- **RUSH** — all five attackers, one route, fast. **SPLIT** — 3 + 2, converging.
- **DEFAULT** — slow map control, spread out, no convergence (settles the old §9 question).
- **RETAKE** — defenders regroup, then arrive on the site together. **TRICKLE** — they arrive one
  at a time. **SAVE** — they give the site up to keep weapons.
- The **site comes from where the bomb was planted**, so no label names it (RETAKE_A/RETAKE_B was
  rejected: it would just re-learn the plant location). `Round` carries `label`, `plant_site`,
  `ct_label`, `positions`/`frames` (pre-plant) and `post_positions`/`post_frames`.
- `RoundLabeller` keys: `1-5` attacker tactic, `a`/`b` plant now at A/B, `6/7/8` defender
  response, `x` clear plant, `t`/`c` winner. It disables matplotlib's own keymaps (`s`, `c`, `q`,
  arrows used to also trigger save-dialog / back / quit).

---

## 2. Current state

### Files (the project folder is a fresh start — all older `.py`/`.ipynb` were deleted)

```
project/
├── stratscout/                 implementation modules and explicit workflows
├── stratscout_prototype.ipynb   thin main experiment interface
├── stratscout_dem_ingest.ipynb  thin companion: radar-only clips + .dem positions
├── tests/                     regression fixtures and synthetic workflow checks
├── demos/                       .dem files, searched recursively (gitignored)
├── mirage_radar.png             clean radar screenshot (286², has the bomb icon) = background for
│                                EVERY plot via plot_background(). Frames the same area as the crop
│                                (corr 0.986 vs footage median, plain resize). NOT used for synthetic
│                                training frames while footage exists (the median is): the baked-in
│                                orange bomb icon would look like a static T player.
├── data/                        rendered demo replays, *.mp4, ONE FILE = ONE MATCH (gitignored)
└── labels/                      round labels, <match_id>.rounds.json per video (created on use)
```

**Ingest:** every `.mp4` (any case) in `Config.data_dir` (`data/`) is picked up; files OpenCV
cannot open, and videos whose radar cannot be found, are skipped with a warning. The file stem is the
`match_id`, which is what train/val/test splits group by — never put two matches in one file or
split one match across files. `Config.video_path=None` → the first video (sorted) is the
"primary" one used by §3.1 verify and §16 examples. Each video gets its own auto-detected crop
box, cached in `crops.json` keyed by file name (a renamed file is matched by byte size + resolution; entries for removed videos are dropped) (`CROPS[name]`; `CROP` = primary's).

`requirements: numpy matplotlib torch opencv-python`

**Implemented direction (2026-10-12):** the lecturer accepts either format, and both
notebooks now import the Python package. Configuration and intermediate results are
passed using per-run contexts. Keep imports one-way; never launch experiments on
import. See [architecture](docs/architecture.md) for module ownership and examples.

### Everything the notebook writes (all outputs, none are inputs)

`pilot_frames.npy` (pooled across all videos), `pilot_frames_sources.json` (row range per match),
`pilot_labels.json`, `crops.json`, `map_points.json`, `labels/<match_id>.rounds.json`,
`rounds_data.npz` (all labelled matches, with `match_ids`),
`runs/prototype/{detector.pt, stratscout.pt, summary.json}`

§4 refuses to re-extract `pilot_frames.npy` while `pilot_labels.json` exists — the labels index
into those frames, so re-extracting would silently scramble them. Delete both to start over.
Without labels, it reuses the file if the set of videos (name + byte size) is unchanged, and
otherwise moves the old one aside as `pilot_frames.bak-<timestamp>.npy` before re-extracting.

### Where the project actually is

- Capture settings **solved and verified** (see §4).
- Full pipeline **runs end to end** on synthetic data.
- Detector **trained once on real POV footage** before a data-source pivot: reached
  `P 0.617 / R 0.512 / 1.94px` at only 10 epochs, still climbing steeply, never re-run longer.
  **That footage and those labels are now obsolete** — the project moved from player-POV to
  demo-replay recording.
- **Real demo replays are in `data/`** (2 as of 2026-10-07), both 1280×720 @ 60fps, both
  auto-cropped to `(4,4,196,196)` at ≤1.9% change. Files are named by match (renamed 2026-10-07); keep
  names fixed once labelling starts — the stem is the `match_id`, and `labels/` + `crops.json` are keyed by it.

  | file | match (read from HUD name tags) | length |
  |---|---|---|
  | `vitality_vs_falcons.mp4` | Vitality vs Falcons | 40.9 min |
  | `aurora_vs_betboom.mp4` | Aurora vs BetBoom | 38.9 min |

- Next: pilot frame extraction (§4) on these, then icon labelling.

---

## 3. Pipeline and data contracts

```
demo replay .mp4
  → auto-crop the radar from the top-left of the screen        (§3 of notebook)
  → Model 1 (CNN) detect orange T / blue CT icons              (§10)
  → FixedSlotTracker  link icons into 10 stable slots          (§11)
  → trajectories (32, 10, 2) → feature sequence                (§13)
  → Model 2 classify the round's tactic                        (§14)
  → heatmaps, tendencies, win rates                            (§17–19)
```

**Contracts — do not change these without updating every stage:**

| Stage | Name | Shape / type | Meaning |
|---|---|---|---|
| ingest | `frames` | `uint8 (N, H, W, 3)` | cropped radar, RGB, 0–255 |
| preprocess | `x` | `float32 (N, 3, 128, 128)` | normalised 0–1, channels first |
| detector | `detections` | list of `{"T": (k,3), "CT": (m,3)}` | rows `(x, y, score)`, normalised |
| tracker | `positions` | `float32 (N, 10, 2)` | slots **0–4 = T, 5–9 = CT** |
| trajectory | `sequence` | `float32 (32, D)` | D=16 for `features`, 20 for `coords` |
| classifier | `logits` | `float32 (B, 5)` attacker + `(B, 3)` defender | order fixed by `T_CLASSES` / `CT_CLASSES` |

Coordinates are **normalised to the crop**, x right, y down. Never pass raw pixels across a
stage boundary — pixels change the moment someone edits the resolution; fractions do not.

**Identity is not promised.** Icons within a team are identical, so slots can swap when two
teammates cross. The classifier reads team *shape*, not individuals.

---

## 4. Capture settings — verified, do not change casually

Recorded from **demo replay**, so both teams are visible: **orange = T (attackers), blue = CT
(defenders)**. The radar sits in the **top-left** of the replay screen.

### Console (an `autoexec.cfg` + `+exec autoexec` launch option; CS2 does not auto-run it)

```
cl_radar_always_centered 0
cl_radar_rotate 0
cl_radar_scale 0.25
cl_hud_radar_scale 1.3
cl_radar_icon_scale_min 0.8
cl_radar_square_with_scoreboard 0
cl_teammate_colors_show 1
```

### Menu-only (not console-settable) — Settings → Game → Radar/Tablet

| Setting | Must be |
|---|---|
| Radar Centers The Player | NO |
| Radar Is Rotating | NO |
| Radar Is Zooming Dynamically | NO |
| Radar Map Zoom | 0.25 |
| **Radar Hud Map Blends With Background** | **NO** |
| **Blur Background** | **NO** |
| **Radar Hud Background Opacity** | **1.00** |
| Toggle Square Shape With Scoreboard | NO |
| Force Square Shape | YES |
| Radar Hud Size | 1.30 |

The four bolded ones were the cause of a real failure: with blending and 63% opacity the 3D
world bleeds through the radar, so `verify_crop` reported **98.3% of pixels changing**. Fixing
them dropped it to **5.7%**, and the residual was the bomb-radius ring, not drift.

**OBS:** fixed resolution (note it down), high bitrate. A teammate recording at a different
resolution produces a different crop box and their footage will not work with yours.

---

## 5. Hard-won lessons — read before debugging anything

These each cost real time. Several are **silent** failures: no error, no crash, just wrong
numbers.

### 5.1 The sub-pixel heatmap bug (detector predicts nothing)

Centring the training Gaussian on the true sub-pixel location means the brightest pixel may only
reach ~0.8. The focal loss then finds **zero positive pixels**, and the detector learns to
predict nothing anywhere. Recall sat at **0.000** and looked like a broken model; it was a broken
label.

**Fix (already in `gaussian_heatmap`):** snap the peak to the nearest integer pixel and force it
to exactly 1.0. Recall went 0.000 → 0.93.

### 5.2 Free-track tracking fragments (was 90% of total error)

A textbook "each unmatched detection starts a new track" tracker turns 5 players into ~20
fragments, because every missed detection orphans a track and every recovery spawns a fresh one.
"Keep the longest five" then grabs fragments.

**Measured:** free tracker **9.74px** → `FixedSlotTracker` **0.80px**, against a detector floor
of 0.51px.

**Fix:** exactly 5 slots per team, created once and never destroyed; a lost slot coasts on its
last velocity; a spare detection goes to whichever slot has been lost longest.

### 5.3 Overlapping icons cannot be split by thresholding

Five players stacked at spawn are one connected blob. Connected-component labelling has no
concept of "too big to be one player" and never will.

**This is the CNN's entire reason for existing** — but it only learns to place several peaks
inside one blob **if your labels show it that**. When reviewing a clustered frame, click all five
and estimate the centres. Labelling a cluster as one player teaches the model to reproduce the
threshold's own failure.

### 5.4 Static artifacts get detected as players

The A/B site letters and the green spawn-zone outlines are saturated and pass a naive colour
filter. Exclude boxes work but are fragile and risk blinding the detector where players actually
go.

**Better fix (in use):** median-frame background subtraction. Anything that never moves is not a
player. Measured on synthetic data: false positives **106 → 4** over 40 frames, with recall
unchanged.

### 5.5 Colour pitfalls

- Red is reserved for **enemies and death markers** in CS2, never teammates — a safe prior for
  rejecting them during hue learning.
- `learn_team_hues` measures the real hues from your own footage. Defaults are T≈17, CT≈105
  (OpenCV hue, 0–179). Verified on test footage as T=16, CT=105.
- Orange and magenta sit only 9–12 hue units from the red reject band. If icons start going
  missing, suspect the hue bands before anything else.

### 5.6 Real demo footage (first measured on data/vitality_vs_falcons.mp4, formerly mirage-demo.mp4, 1280×720 @ 60fps, 41 min)

- **The radar sits flush in the corner**, and its outer frame lets the 3D world bleed through:
  the outer 4px top/left changed in ~95% of frames. With that border included the box was
  `(0,0,198,198)` and the change figure was **8.8%**. `find_radar_box` now trims edge lines that
  are less than 90% stable, giving **`(4,4,196,196)`, 1.9% change**.
- **The end-of-match "MATCH RESULT" screen covers the radar**, and stopping 20 s before the end
  does not avoid it: 2 of the 400 old pilot frames were that screen (98.6% changed). `radar_hidden()`
  now drops any frame where more than 30% of the crop differs from the median map. Expect the same
  at half-time, pauses and menus.
- **T icons look yellow-orange, not pure orange**: `learn_team_hues` measured **T=27, CT=94** on
  the two matches (defaults 17/105). Always learn hues; never trust the defaults on new footage.
- **Classical finder on real frames is weak:** mean 2.7 T / 3.0 CT per frame, all ten players in
  only **9/100** frames (overlaps, bomb-carrier and death-marker icons, real deaths — can't be
  separated without labels). This is the real baseline the CNN must beat.
- The old 198×198 `pilot_frames.npy` came from this video with the untrimmed crop. It is obsolete;
  §4 backs it up and re-extracts the first time it runs on `data/`.

### 5.6b Off-map detections (fixed 2026-10-07)

Synthetic spawn/arrival jitter put attackers on the black area beside T spawn, and the detector
duly detected them (§10.4). Fixes: `MAP_MASK` (walkable = radar brightness > 25) built in §7;
`_jitter_on_map` re-draws offsets until on the map; `snap_to_map` is the last resort; and
`decode_heatmap` zeroes peaks outside `DETECT_MASK` (MAP_MASK dilated 2px). The margin matters:
filtering at the exact map edge cost **2.7 points of recall**, because correct peaks land a pixel
past it. Old recall 0.954 was flattered by isolated off-map players; on-map rounds give 0.923.

**Consistency rule (user demand):** the slides and the results web page show exactly the
figures the notebook draws (29 as of 2026-10-07), no more, no fewer. Add a figure to the notebook first, then to both.

### 5.7 Mechanical traps

- **OpenCV gives BGR.** Convert to RGB or the team channels swap.
- **`frames` must be `uint8`.** The preprocessor divides by 255 itself; hand it floats in [0,1]
  and everything goes black.
- **Split by match, never by frame.** Frames inside a round are near-copies; shuffling them is
  revising with the exam paper. `generate_split` asserts no match-id overlap.
- **`np.argmin` on an empty array throws.** Guard the zero-living-players case in
  `detector_metrics`.
- `extract_clip` clamps to the video length, so `end_s` is an upper bound, not a requirement.
- Interactive tools (`ReviewTool`, `RoundLabeller`, `pick_map_points`) need a **live window** —
  run from a terminal or use `%matplotlib widget`. `%matplotlib inline` renders static images and
  the clicks go nowhere.

---

## 6. Key design decisions and why

### Auto-crop replaces manual clicking

A HUD overlay is drawn at exactly the same pixels every frame; the rendered 3D world never
repeats. So sample ~20 frames across the video and find the top-left region that is both
**perfectly stable** (within ±3 of its median in nearly every frame) **and detailed** (strong
edges). Stability alone catches flat sky; detail alone catches the whole world. The intersection
is the radar.

Tested against a synthetic replay screen with known radar position: **detected within 2px**.
`Config.crop_box` overrides it if ever needed.

### Centre heatmaps, not bounding boxes

Every icon is the same size, so a box adds nothing a centre does not; the tracker consumes
points; and clicking a centre annotates far faster than dragging a rectangle. Model 1 is a
fully-convolutional CenterNet-style detector, **98,482 parameters**, 2 output channels (ch0 = T,
ch1 = CT) — so **team classification comes free** from which channel the peak appears in.

### Focal loss, not cross-entropy

~99.9% of pixels are background. Plain cross-entropy is minimised by predicting "nothing,
anywhere".

### `det_stride = 1` (full-resolution heatmap)

The 3×3 peak suppression works in *heatmap* pixels. At stride 2, two players 4px apart in the
image are only 2 heatmap px apart and one is suppressed as a duplicate.

### The classical colour finder is kept on purpose

It does two jobs: proposes labels so review is ~5× faster than clicking from scratch, and serves
as **the baseline the CNN must beat**. "Classical colour segmentation achieves X%; our CNN
achieves Y%, with the gap concentrated in occluded and overlapping cases" is a far stronger
report result than a CNN number quoted alone.

### `FeatureMLP` is a control, not a contender

It pools over time and discards ordering entirely. **If the temporal models cannot beat it, they
are doing nothing a bag of averages couldn't.** On an earlier, cruder synthetic map it *tied* at
1.000 — which revealed the task didn't need time at all. Watch this comparison on real data.

### Pixels only where pixels are needed (`stratscout_dem_ingest.ipynb`)

> **User decision 2026-10-07:** nothing from this notebook goes into the slides, the results
> page, or the main notebook. It is parked for later review.

Only the detector needs pixels. Model 2, heatmaps and win rates run on positions, so they can be
trained from `.dem` files (`demoparser2`) with no rendering. Rendered matches are needed only for
detector labels and for held-out vision evaluation (Experiment B). This is development-time use
of `.dem`; the deployed path is still pixels-only.

- **Radar-only clips:** 2.5 GB → ~80 MB per match (192×192 @ 8 fps, mp4v; no H.264 in this
  OpenCV build). Main notebook reads them with `data_dir="data_radar"`, `crop_box=(0,0,192,192)`.
- **World → crop mapping:** de_mirage overview `pos_x −3230, pos_y 1713, scale 5.0` (1024 px),
  then a fitted `{s: 1.01, ox: −0.005, oy: −0.005}` in `radar_fit.json`. 99.6% of 33k demo
  positions land on walkable radar pixels. **Fit is relative to crop `(4,4,196,196)`** — refit if
  the crop changes.
- Demo rounds: `round_freeze_end → round_end`, 32 steps, slots sorted by Steam ID (true identity),
  dead players NaN, winner + plant site from the demo. Label key `x` = EXCLUDED (the §9 question).
- **Split by `match_group`** (demo `vitality-vs-falcons-m3-mirage` ↔ video `vitality_vs_falcons`),
  or a match trains on its demo and tests on its own video.
- demoparser2 gotcha: `bomb_planted.user_steamid` is a **str**, `parse_ticks` steamid is
  **uint64** — compare as strings or every plant silently goes missing.
- Not built yet: demo ↔ video time sync (would give free tracker evaluation + icon labels).

### Win rate is annotation, not vision

The radar cannot distinguish a bomb detonation from a defuse, and the round timer is outside the
crop. The scoreboard already states the result. So the winner is recorded with one keypress in
`RoundLabeller` (`t` / `c`) and stored as a **label, never a model input** — the pixels-only rule
stays intact.

The interesting question is conditional: *given an A_SPLIT, how often did it work?* That needs
the classifier, which is what makes it worth building.

---

## 7. Measured results (full run 2026-10-07: real footage for §1–6, synthetic from §7 on)

> **These numbers are not findings.** In synthetic mode the generator draws the dots and the
> detector then learns to find dots the generator drew — a closed loop. They prove the plumbing.

| | |
|---|---|
| Detector | precision 1.000, recall **0.945**, 0.49px error (players kept on-map; misses = stacked clusters) |
| + tracker | 1.01px |
| Q1 attacker (test acc) | all four models **1.000**, MLP control ties — synthetic Q1 too easy to rank them |
| Q2 defender (test acc) | TCN 0.938 · Transformer 0.938 · GRU 0.896 (best on val) · **MLP control 0.771** — timing matters |
| Experiment A → B | Q1 1.000 → 0.950; **Q2 0.896 → 0.729** (crowded post-plant sites are the detector's worst case) |
| B′ (Q1, trained on vision, n=60) | 0.633 / 0.334 — **collapses; a sample-size confound** |
| Overall side win rate | 49.4% T, 95% CI [44.3, 54.6] → *no significant advantage* |

Model sizes: `feature_mlp` 14,501 · `tcn` 61,765 · `gru` 66,661 · `transformer` 154,949.

**A note on the gap's sign.** On an earlier invented map it came out **negative** (vision beat
oracle), which is physically impossible and was pure noise from near-ceiling scores on 60 test
rounds. Real map geometry gave the experiment enough headroom to measure something. If the sign
goes negative again, the experiment is uninformative, not surprising.

---

## 8. Sample size — this drives the collection plan

Wilson 95% intervals, at p=0.5:

| Precision wanted | Rounds needed |
|---|---|
| ±20pp | 25 |
| ±15pp | 43 |
| **±10pp** | **97** |
| ±5pp | 385 |

Per-tactic win rates at ±10pp need ~97 rounds **each** → ~**485 rounds ≈ 19 matches**.
The classifier itself needs roughly **200 labelled rounds** to train meaningfully (8–10 matches).

Heatmaps need volume too: under ~20 rounds use `show_trails()` (individual paths) rather than a
heatmap, which would just be a few paths smeared into a blob.

---

## 9. Class-scheme decision — RESOLVED 2026-10-07 (see §1: two questions per round)

Kept for the record of why:

**The five classes have no bucket for a slow default round.** Most competitive rounds are map
control — take mid, poke B, rotate, commit late — and are neither a rush nor a split. Annotators
will meet these constantly and be forced to file them as something they aren't, which pollutes
every class at once.

Two acceptable resolutions:

1. **Add `DEFAULT`** as a sixth class (or replace `RETAKE` with it and make retake a separate
   binary post-plant model).
2. **Exclude those rounds** and write that down in the report, accepting a smaller dataset.

What is *not* acceptable is forcing them into a bucket silently.

Secondary: `RETAKE` is the only CT-side class in an otherwise T-side head, so expect it to behave
differently in the confusion matrix. `Round.attacking_side` already handles this for win-rate
purposes.

---

## 10. Next steps, in order of risk removed

1. ~~**Record one demo replay.** Confirm auto-crop lands on the radar and §3.1 is low.~~
   **Done 2026-10-07** on vitality_vs_falcons.mp4: crop `(4,4,196,196)`, 1.9% change (see §5.6).
2. **Icon labels (~200 frames).** `auto_label_icons` → `ReviewTool` → retrain the detector.
   Target recall **> 0.85**. Click all five on overlapping spawn clusters.
3. **Round labels.** `RoundLabeller` on each match, **including the winner keypress**. This is
   the long pole — split it across the team. ~25 rounds per match.
4. **At ~100 rounds:** first real Experiment A vs B; heatmaps start meaning something.
5. **At ~400 rounds:** per-tactic win rates get tight enough to quote.

### Known gaps deliberately left open

- **Dead-player handling in the tracker.** Real rounds lose icons permanently; the tracker coasts
  then sits still, which quietly drags the centroid feature. **This is a correctness bug, not a
  refinement** — fix it early.
- **Plant detection from pixels.** The labeller marks the plant by hand; at deployment the planted
  bomb must be spotted on the radar or question 2 cannot be asked live.
- **Class imbalance.** `DEFAULT` will dominate real rounds, `SAVE` will be rare. Add class weights to the cross-entropy once real
  counts are known.
- **Annotator agreement.** Have two people label the same 30 rounds. If humans agree only 70% of
  the time on `A_RUSH` vs `A_SPLIT`, that is roughly the model's ceiling — and knowing it early
  tells you when to stop tuning.
- **Deployment window strategy.** During training, round boundaries come from the annotator. At
  deployment there is none. Options: sliding window (implemented), OCR the round timer, or an
  analyst marks rounds. **Pick deliberately** — this is the one place a `.dem` dependency can
  sneak back in unnoticed.

---

## 11. Ethics / deployment constraint

An earlier idea was a **live in-game overlay predicting enemy positions**. That is third-party
software providing gameplay advantage — against Valve's rules and functionally indistinguishable
from a radar hack, since the model would be inferring un-spotted positions from partial contacts.
The fact that it is ML does not change what it does.

**Offline VOD scouting is fine. An analyst's second-screen review tool is fine. A coach between
maps is fine.** The deliverable should be a desktop app that ingests an `.mp4` and produces a
scouting report; keep "live" scoped to replaying a recorded demo. The tension between "scouting
tool" and "real-time assistance" is a strong write-up in its own right (Ryan takes an Ethics in
Technology module) but should not be the deployment target.

---

## 12. Working preferences

- **Direct, honest critique over diplomatic softening.** Say when something is wrong.
- **Analogy-driven explanations** for technical concepts.
- **Concise.** Targeted edits over full-file replacements; incremental over wholesale rewrites.
- When giving code edits, **say which file and which section**, and give exact indentation —
  several past errors were indentation and use-before-definition.
- Placeholders must be called out as placeholders. Any concrete-looking filename or number is a
  placeholder unless it came from the user's own machine.

### Environment

Windows, VS Code, project at `C:\Users\xryan\source\repos\CV-DL`, Python 3.11.

---

## 13. Notebook section map

| § | Contents |
|---|---|
| 1–2 | Config, data contracts |
| 3 | **Auto-crop** the radar; 3.1 verifies it |
| 4 | Frame extraction (evenly spaced, not fixed-fps) |
| 5 | Classical icon finder: background subtraction, learned hues, saturation-weighted centres |
| 6 | **Visualisers**: `show_detections`, `show_mask_pipeline`, `show_counts` |
| 7 | Synthetic fallback on the real Mirage radar; `pick_map_points` |
| 8 | `Round` dataclass, match-level splits |
| 9 | Preprocessing |
| 10 | **Model 1** detector + training + `show_detector_output` |
| 11 | `FixedSlotTracker` |
| 12 | `auto_label_icons`, `ReviewTool` |
| 13 | Trajectory representations (`features` / `coords`) |
| 14 | **Model 2** — MLP / GRU / TCN / Transformer |
| 15 | **Experiment A vs B vs B′** |
| 16 | `RoundLabeller` (with winner keypress), `rounds_to_dataset` |
| 17 | End-to-end pipeline, scouting report |
| 18 | **Objective 2** — shared helpers; 18.1 T vs CT occupancy; 18.2 per-tactic grid (T/CT rows); 18.3 route trails for all 5 tactics (RETAKE drawn from CT) |
| 19 | **Objective 3** — win rates with Wilson intervals |
| 20–21 | Saving, what to replace next |

Four real anchors on the Mirage radar are **measured**, not guessed:
`b_site (0.230, 0.276)`, `a_site (0.545, 0.770)`, `t_spawn (0.880, 0.360)`,
`ct_spawn (0.297, 0.697)`. The six connecting waypoints are estimates — `pick_map_points()`
replaces them by clicking, saved to `map_points.json`.
