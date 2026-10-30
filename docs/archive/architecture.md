# Modular Architecture

> Archived reference (30 October 2026). Results, commands and status describe the recorded revision, not necessarily the current checkout. Use the [README](../../README.md) for current status and next steps, the [recording guide](../recording-setup.md) for capture, and the [tactic guide](../tactic-definitions.md) for labels.

## Boundaries

The package owns implementation. Earlier notebooks were optional experiment interfaces.
There is no runtime notebook parsing, `exec` loader, wildcard
import or duplicated notebook implementation. Normal Python imports connect modules.

| Modules | Responsibility |
|---|---|
| `config`, `context`, `contracts` | Configuration, per-run state, `Round` and shape contracts |
| `video`, `pilot` | Video discovery, crop detection, extraction and pilot caches |
| `segmentation`, `preprocessing`, `detector` | Colour baseline with badge shape exclusion, RGB tensors and CNN heatmaps |
| `synthetic` | Mirage geometry, backgrounds and rendered synthetic rounds for detector/tracker data |
| `datasets`, `training`, `evaluation` | Icon samples, detector optimisation and detector metrics |
| `tracking` | Fixed-slot tracking of detections into 10 player positions |
| `annotations` | Icon review and video round labelling (start/plant/end in video seconds) |
| `visualisation` | Diagnostic figures for crops, colour masks and detector output |
| `storage`, `workflow`, `__main__` | Protected experiment outputs, explicit stages and CLI |
| `ingest/` | Separate configuration, video shrinking, demo parsing, mapping, labels and export |
| `media`, `video_profiles`, `video_preparation` | FFmpeg/probe helpers, half-open crops, aligned native-resolution exports |
| `fps_comparison` | Independent visual/timing/storage reports from prepared videos |
| `recording_batch` | Inbox scan, shared crop, safe reruns and automatic preparation/comparison |
| `bomb`, `rounds`, `live_preview` | Colour-then-shape C4 badge detection, unreviewed dropped-bomb candidates, video-only round estimates and an interactive detection preview ([bomb detection](bomb-detection.md)) |
| `icon_split`, `count_eval` | Per-team masks, death-marker removal, disc-fit splitting with a 5-per-team cap, and label-free count checks ([overlapping icons](icon-split.md)) |

The [migration inventory](module-inventory.json) maps the original functions and
classes still present to their current modules and original zero-based notebook cell
indices. The original notebooks are available at Git commit `00e024f`.

**Tactic classification is not in this package.** The `vision-only` branch removed the
old classifiers (`models`, `features`, `pipeline`, `reports`, `SequenceDataset`,
`train_classifier`, `evaluate`, the round-to-clip dataset builders and their workflow
stages). They stretched every round to 32 steps between start and plant, so trajectory
timing in seconds was lost. A new time-aware tactic model will be built separately on
top of the tracker's positions. The removed code is available at commit `1a63107`.

The target is now one of `A_DIRECT`, `B_DIRECT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`
per reviewed tactical segment, as specified in [tactic definitions](../tactic-definitions.md).
An attack and a subsequent retake in one round are separate examples. Region polygons
and approach history support direct-versus-mid-split classification; ramp + palace
alone is not an A split. Tempo is supporting evidence, not a separate class.

`config.py`, `contracts.Round` and `annotations.RoundLabeller` retain legacy round/
pre/post-plant contracts. `vision-only` does not expose `label-rounds`; that command
on the separate branch is also legacy. A versioned segment format, review-state support,
region configuration and corresponding dataset/model integration remain unimplemented.
Migrate by reviewing source clips, not by remapping old class strings automatically.

Preparation commands bypass the prototype workflow. They use JSON profiles/manifests
and explicit arguments, not `RunContext`, the 32-frame sampler or tracker.
`process-recordings.ps1` launches the batch CLI using a local project environment.
The batch calls the existing preparation/comparison APIs; it does not duplicate
their encoding logic. Preparation/comparison depend on `media`. Capture is manual and has no Python OBS
dependency. See [recording setup](../recording-setup.md) for
formats, paths and recovery semantics.

Preparation manifests now use schema 2 with `playback_speed` and explicit timeline
mapping. Trim bounds remain source-video seconds. Normal-time output duration is
retained source duration multiplied by playback speed. Comparison accepts older
schema-1 bundles as speed 1. The batch prompts for speed/start/end for new recordings;
answers are recorded in the output manifest, with no per-video settings sidecars.
Completed bundles reuse their recorded settings without prompting. An explicit
non-interactive mode processes whole files with a supplied/default speed and crop.
The batch default is 2× for current 1440p/60 fps captures. The standalone
`prepare-video` API remains explicit: pass `--playback-speed 2` for accelerated input.
No capture-specific profile ships with the repository; first use selects the crop.
Speed correction precedes canonical sampling; crop profiles remain independent.
Batch folder names use cleaned source filenames. Manifest source hashes still drive
reuse, including legacy hash-only folders and renamed sources. Name collisions get
a hash suffix; display names do not replace match identities used for dataset splits.

## Explicit state

Functions that previously read notebook globals now accept keyword-only `context`.
Stateful classes retain the context supplied to their constructors. Pure utilities
remain ordinary functions without a context argument.

```python
from stratscout.config import Config
from stratscout.context import RunContext
from stratscout.segmentation import find_icons_teams
from stratscout.video import extract_clip

context = RunContext(Config(), device="cpu")
frames = extract_clip("normal-speed.mp4", 60, 65, context.config.crop_box, n_frames=32)
points = find_icons_teams(frames[0], context=context)  # RGB, normalised x/y/team
```

Set `context.config.crop_box` to reviewed bounds for the normal-speed input before
this example. Validate the crop for each source. `RunContext.config` replaces `CFG`; `context.device`
replaces `DEVICE`. Intermediate results use `context.state`, retaining familiar
names such as `VIDEOS`, `CROPS`, `detector` and `DET_METRICS`.
Each run owns its own dictionary and caches. Do not introduce module-level run state.

## Workflow order

`workflow.run_prototype(context)` handles the complete vision experiment. Individual
steps remain callable for notebook exploration:

1. Discover videos, locate crops, prepare pilot frames and the background.
2. Prepare map geometry, synthetic rounds, preprocessing, detector data and mask.
3. Train/evaluate the detector and check tracking error against true positions.
4. Save `detector.pt` and `summary.json` to a fresh directory.

For example, after detector preparation/training, the notebook can access
`context.state["detector"]` and pass it to `tracking.positions_from_video`.

Interactive tools are explicit too:

The example below launches the **legacy** round labeller, not the proposed five-class
segment tool. Do not treat its keyboard choices or exports as the new class contract.

```python
from stratscout.annotations import RoundLabeller

video = context.state["VIDEOS"][0]
RoundLabeller(str(video), context.state["CROPS"][video.name], context=context).run()
```

Run interactive notebook tools with an interactive Matplotlib backend. CLI runs use
the non-interactive backend unless `--plots` is supplied. Run from the repository root;
relative configuration paths are resolved against the working directory.

## Behaviour and verification

This migration preserves the numerical algorithms. Regression fixtures were produced
by executing selected definitions from the original committed notebook. Tests compare
trajectories, rendered frames, tracking, heatmaps and CNN outputs, check
imports and module dependency cycles, and test video decoding/export. The retained
notebook execution checks require the earlier thin notebook files to be present.

Intentional orchestration changes: CPU is the default device; experiment sizes are
configurable; checkpoint/summary writes refuse overwrites. Summary metadata
distinguishes real from synthetic detector data. No model-quality fix or real-footage
validation is implied by these checks.

The [baseline review](base-review-2026-10-12.md) remains the list of algorithm/data
faults to tackle next. In particular, review-tool saves and ingest exports retain
legacy semantics; output protection for experiments is not a general annotation fix.
