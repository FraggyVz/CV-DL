# StratScout

StratScout investigates tactical recognition from CS2 minimap video. The current
milestone detects **visible T and CT icon centres using colour-based OpenCV**.
It reads a recording and exports an annotated video and timestamped coordinates.
It does not require a `.dem` file at inference.

## Actual status

- Implemented: interactive crop/colour calibration, two-team colour detection,
  streaming video export, run provenance and point-annotation evaluation.
- Automated validation uses generated footage. Initial calibration and a short
  preview on the supplied Falcons/FURIA Mirage recording are available; see the
  [recording review](docs/recording-review.md). **Held-out real-footage accuracy
  is not yet measured.** Obscured carriers and overlapping icons remain limitations.
- No player tracking, tactical recognition, dashboard or automated replay
  recording is provided by the new command-line tool.
- The superseded notebook, teammate-only pilot and historical checkpoints have
  been removed from the working tree. Their original versions remain in Git
  history at `6755650`; their synthetic scores are not real CS2 results.

The eventual scope remains Mirage and five tactical classes: `A_DEFAULT`,
`B_DEFAULT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`. `.dem` data is development ground truth,
not a replacement for the visual pipeline.

## Windows setup

Run from the repository root. These commands use the environment's interpreter
directly, so PowerShell activation is unnecessary:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m stratscout --help
```

The pinned NumPy 2.2.6 and opencv-python 4.12.0.88 wheels were installed and tested
on Windows with Python 3.13. Calibration requires a local desktop; do not replace
OpenCV with its headless package. This detector runs on CPU and needs no CUDA,
PyTorch, Jupyter or trained checkpoint. If Python 3.13 is unavailable, install it
before creating the environment; this dependency set is not validated on 3.14.

## Prepare the first recording

Use the [CS2 recording and radar setup](docs/recording-setup.md) for the minimap
preset, default reset commands and instructions for preserving your playing settings.
Apply the recording preset only for demo capture; restore your playing settings afterwards.

1. Replay a Mirage demo in CS2 and record **at least three rounds**. Include
   movement, deaths, close/overlapping players, bomb/objective markers and quiet frames.
2. Use a spectator/radar view that displays both teams with two distinct colours.
   Player-perspective footage with hidden opponents or individual teammate colours
   is unsuitable for this milestone. Verify the T/CT colour mapping yourself.
3. Keep the full relevant minimap visible, non-rotating, and fixed in orientation,
   zoom, screen position and size. Check distant points in the recording: the map
   background should stay aligned while icons move. Keep the scoreboard closed.
4. Capture the **full CS2 game frame**, not the desktop or just the minimap. Record
   at a **constant frame rate**, fixed resolution and normal replay speed.
   Avoid pauses, seeks, HUD changes and scaling changes inside a recording. Use
   MP4 or MKV that OpenCV can decode; minimise compression around small icons.
5. Place the recording at `data/raw/mirage.mp4` (create the directory if needed).
   Retain the source `.dem` and note match ID, map, replay version, HUD/radar settings,
   recording resolution/frame rate, and any recording/replay time offset beside it.
   Recording time is not demo/game time; synchronisation is later work.

Record a whole match once and derive frames/windows from it. Do not record each
training example separately. Reuse calibration only when the HUD, colours, zoom,
resolution and crop remain the same. Other matches are needed for later generalisation
tests. Raw recordings and generated runs are ignored by Git.

## Calibrate

```powershell
.\.venv\Scripts\python.exe -m stratscout calibrate --video data/raw/mirage.mp4 --profile configs/mirage.json --frame 0
```

Choose a starting `--frame` where the radar is visible. Drag a rectangle tightly
around the minimap, then press Enter. Crop coordinates use exclusive bottom/right
edges. Detection keeps native crop resolution; preview enlargement changes neither
coordinates nor size thresholds.

- Select T or CT using the team slider or **T/C** keys, then click the coloured
  core of a known icon in the enlarged minimap. Both teams must be sampled before saving.
- Use the **Frame** slider or **N/B** for next/previous frame. Check different
  calibration frames and rounds; reserve evaluation rounds before tuning.
- Adjust the selected team's hue/tolerance, minimum saturation/brightness,
  component area, maximum aspect ratio and minimum bounding-box fill. Area uses
  native crop pixels. The mask preview shows pixels passing the colour threshold.
- Overlapping T/CT hue ranges are rejected rather than assigning one blob to both
  teams. Inspect false positives while adjusting colour and size thresholds.
- Press **S** to save, or **Q/Escape** to cancel. Invalid settings cannot be saved.

Existing profiles are never overwritten. Use `configs/mirage-v2.json` for changes.
Resolution changes are rejected automatically; same-resolution HUD changes still
require manual inspection and recalibration.

## Detect and inspect

```powershell
.\.venv\Scripts\python.exe -m stratscout detect --video data/raw/mirage.mp4 --profile configs/mirage.json --output runs/two-team-001
```

The output directory must not exist. Processing reads frames sequentially and
preserves their order and nominal frame rate. It then decodes the overlay to verify
the export. Outputs are:

| File | Contents |
| --- | --- |
| `annotated.mp4` | Full-frame recording with crop outline, T/CT centres and counts; no audio |
| `detections.jsonl` | One record per source frame, including empty T/CT lists |
| `run.json` | Input SHA-256/path/size, profile, versions, timing convention and completion status |

For a short preview, add `--start-frame 7080 --max-frames 300`. At 60 fps this
exports five seconds beginning at 01:58. JSONL indices and timestamps retain their
positions in the source recording; playback of the output MP4 starts at zero.
`run.json` records the requested range and actual exclusive end frame. Omitting
these options processes the whole recording. `--max-frames` is a cap; the run may
end at the advertised source end, but unexpected early decoding still fails.

Example record (illustrative, not a measured result):

```json
{"frame_index": 120, "timestamp_s": 2.0, "T": [{"x": 0.25, "y": 0.60}], "CT": []}
```

Coordinates are relative to the crop: `x = local_pixel_x / crop_width`,
`y = local_pixel_y / crop_height`, right/down positive. Lists contain observed
components, not player identities. There is no padding, five-player cap or fabricated
confidence probability. A count above five is a useful sign of false positives.

Timestamps are `frame_index / source_fps` from the recording start. Variable-frame-rate
video is outside the timing contract: convert it to constant frame rate first.
Odd source dimensions are padded by at most one black row/column in the overlay
for MPEG-4 encoding; detection coordinates are unchanged. Failed runs retain partial
outputs with `status: failed` when metadata can be written. Use a new run directory
after correcting the error.

## Real-footage evaluation

Follow [annotation guidelines](docs/annotation-guidelines.md). Manually review at
least 30 frames: 10 for calibration and 20 from separate held-out rounds. Match
centres one-to-one within 2% of the crop diagonal. The target is at least 90%
precision **and** recall for each team, plus inspection of difficult cases.

```powershell
.\.venv\Scripts\python.exe -m stratscout evaluate --run runs/two-team-001 --annotations data/annotations/mirage-review.json --output runs/two-team-001/evaluation.json
```

The evaluator checks video identity, crop, review status and separation of rounds.
It reports teams separately and will not mark an undersized sample as meeting the
protocol. Results describe the supplied recording only. Do not tune on held-out
frames and then report them as untouched evaluation data.

## Limitations

Colour alone cannot reliably distinguish every bomb, death marker, facing cone or
same-coloured map label. Touching icons may merge; an unresolved blob remains one
detection or is rejected by size filters. Invisible players cannot be recovered.
Inspect and report these errors. No map-area exclusion rectangles hide playable
regions in the new detector, and no tracking fills missing positions. Broadcasts
with changing HUDs, moving radars or inconsistent team colours are outside this version.

## Development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover calibration through simulated window inputs, coordinate/colour handling,
invalid profiles, output preservation, matching and real encoding/decoding of
generated video, including preview ranges with original source timestamps.
They do not prove CS2 performance or replace interactive inspection
of calibration windows on real footage.

| Location | Purpose |
| --- | --- |
| `stratscout/` | New calibration, detection, export and evaluation modules |
| `tests/` | Standard-library `unittest` checks and generated fixtures |
| `docs/` | Implementation order and annotation protocol |
| `requirements.txt` | Pinned detector dependencies; no notebook or training environment |

The current tool has no dependency on the retired prototype. Future tracking and
classification should follow the documented interfaces rather than restore its
broken pipeline. `.gitignore` excludes raw recordings, caches and generated runs.

See [Repository Guidelines](AGENTS.md) and the
[implementation plan](docs/implementation-plan.md) before changing scope.
