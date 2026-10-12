# Modular Architecture

## Boundaries

The package owns implementation. Notebooks own experiment order, explanatory text
and presentation. There is no runtime notebook parsing, `exec` loader, wildcard
import or duplicated notebook implementation. Normal Python imports connect modules.

| Modules | Responsibility |
|---|---|
| `config`, `context`, `contracts` | Configuration, per-run state, `Round` and shape contracts |
| `video`, `pilot` | Video discovery, crop detection, extraction and pilot caches |
| `segmentation`, `preprocessing`, `detector` | Colour baseline, RGB tensors and CNN heatmaps |
| `synthetic` | Mirage geometry, backgrounds and synthetic rounds |
| `datasets`, `models`, `training`, `evaluation` | Samples, four classifiers, optimisation and metrics |
| `tracking`, `features`, `pipeline` | Fixed slots, trajectory features and two-head inference |
| `annotations` | Icon review and video round labelling/export |
| `visualisation`, `reports` | Diagnostic figures, occupancy, trails and win-rate summaries |
| `storage`, `workflow`, `__main__` | Protected experiment outputs, explicit stages and CLI |
| `ingest/` | Separate configuration, video shrinking, demo parsing, mapping, labels and export |
| `media`, `video_profiles`, `video_preparation` | FFmpeg/probe helpers, half-open crops, aligned native-resolution exports |
| `fps_comparison` | Independent visual/timing/storage reports from prepared videos |
| `recording_batch` | Inbox scan, shared crop, safe reruns and automatic preparation/comparison |

The [migration inventory](module-inventory.json) maps all 125 original functions and
classes to their current modules and original zero-based notebook cell indices.
The original notebooks are available at Git commit `00e024f`.

Preparation commands bypass the prototype workflow. They use JSON profiles/manifests
and explicit arguments, not `RunContext`, the 32-frame sampler or tracker.
`process-recordings.ps1` launches the batch CLI using a local project environment.
The batch calls the existing preparation/comparison APIs; it does not duplicate
their encoding logic. Preparation/comparison depend on `media`. Capture is manual and has no Python OBS
dependency. See [recording setup](recording-setup.md) for
formats, paths and recovery semantics.

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
frames = extract_clip("recording.mp4", 60, 65, (16, 12, 508, 510), n_frames=32)
points = find_icons_teams(frames[0], context=context)  # RGB, normalised x/y/team
```

The crop above is an example from one recording, not a universal default. Validate
the crop for each source. `RunContext.config` replaces `CFG`; `context.device`
replaces `DEVICE`. Intermediate results use `context.state`, retaining familiar
names such as `VIDEOS`, `CROPS`, `detector`, `DATA`, `RESULTS` and `pipeline`.
Each run owns its own dictionary and caches. Do not introduce module-level run state.

## Workflow order

`workflow.run_prototype(context)` handles the complete experiment. Individual steps
remain callable for notebook exploration:

1. Discover videos, locate crops, prepare pilot frames and the background.
2. Prepare map geometry, synthetic rounds, preprocessing, detector data and mask.
3. Train/evaluate the detector, check tracking/features and train both classifier heads.
4. Compare oracle/vision inputs, construct the inference pipeline and produce reports.
5. Save checkpoints and the summary to a fresh directory.

For example, after detector preparation/training, the notebook can access
`context.state["detector"]`; after `prepare_pipeline`, use
`context.state["pipeline"].predict_window(frames, phase="pre")`.

Interactive tools are explicit too:

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
trajectories, rendered frames, features, tracking, heatmaps and CNN outputs, check
imports and module dependency cycles, test video decoding/export and execute both thin
notebooks with small synthetic inputs in a temporary directory.

Intentional orchestration changes: CPU is the default device; experiment sizes are
configurable; checkpoint/summary writes refuse overwrites; the B-prime experiment is
disabled by default because it uses test rounds for model selection. Summary metadata
distinguishes real detector data from synthetic classifier data. No model-quality fix
or real-footage validation is implied by these checks.

The [baseline review](base-review-2026-10-12.md) remains the list of algorithm/data
faults to tackle next. In particular, review-tool saves and ingest exports retain
legacy semantics; output protection for experiments is not a general annotation fix.
