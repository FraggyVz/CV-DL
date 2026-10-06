# Implementation Plan

## Current objective

Accept a constant-frame-rate Mirage demo recording, detect visible T/CT minimap
centres and export overlays plus timestamped positions. Inference uses pixels only.
The standalone `stratscout` package is the maintained implementation. The superseded
notebook, teammate pilot and checkpoints have been retired to Git history.

Implementation and generated-data tests are present. Real-footage calibration,
manual review and performance validation are **pending the first recording**.

## Milestones and dependencies

| Order | Work and dependency | Completion criterion |
| --- | --- | --- |
| 1 | Protect data and outputs before adding annotation proposals | Existing profiles and run/evaluation outputs are refused; reviewed labels remain input-only; future proposals must use separate files |
| 2 | Establish frame/coordinate contract | BGR uint8 frames are cropped once at native resolution; centres are normalised relative to that crop; offset/non-square crops pass tests |
| 3 | Two-team detection, depends on suitable recording and explicit colour mapping | Reusable profile, separate T/CT lists, overlay, provenance and real-footage validation below |
| 4 | Real datasets, depends on validated capture and labels | Whole-match recordings and demos have IDs; samples retain timestamps, side and review status; train/validation/test are separated by match |
| 5 | Tracking/classification, depends on synchronised ground truth and annotation rules | Missing/dead players are explicit; real Mirage geometry replaces toy coordinates; compare feature MLP and GRU, then oracle versus vision inputs with the same classifier weights |
| 6 | Application/deployment, depends on validated real-video inference | A teammate reproduces a clip prediction from the README; a minimal interface shows evidence, tactic and confidence; local deployment precedes optional cloud deployment |

Do not invent CT observations or fill empty slots with map-centre coordinates.
A later adapter must explicitly connect observations to tracking; no such
integration is claimed now.

## Capture and interfaces

Record at least three Mirage rounds showing both teams in two colours. Keep the
radar fixed and fully visible at constant recording frame rate and normal replay
speed. Record match/demo identity and rendering settings. Retain whole-match video
for repeated extraction rather than recording each example individually.

Select the crop, sample each team explicitly and tune filters on calibration
frames. Save a new profile version for changes. Matching dimensions alone do not
prove the radar/HUD configuration is unchanged.

- `calibrate --video --profile [--frame]`: crop/colour setup; saves a new version-1
  JSON profile with resolution, crop and per-team settings.
- `detect --video --profile --output`: streams frames and exports `annotated.mp4`,
  `detections.jsonl` and `run.json` into a new directory.
- `evaluate --run --annotations --output`: consumes reviewed point labels and
  writes metrics. See [annotation-guidelines.md](annotation-guidelines.md).
- Frame records contain source index, recording timestamp and both T/CT lists.
  Observations contain normalised `x, y`; no identities or calibrated confidence.

Incompatible resolutions, overlapping colour ranges and existing output destinations
are rejected. Failed-run diagnostics are retained after decoding/encoding errors.

## Acceptance evidence

1. Run `python -m unittest discover -s tests -v` in the pinned environment.
   Checks cover calibration, colour/crop contracts, output preservation,
   point matching, streaming export and error paths using generated inputs.
2. Review at least 30 real frames spanning at least three rounds. Reserve 10 for
   calibration and at least 20 from separate rounds for evaluation before tuning.
3. Label all scorable visible icons and empty frames. Check deaths, objective
   markers and close/overlapping icons; retain an audit of unscorable cases.
4. Use one-to-one matching within 2% of the crop diagonal. Require at least 90%
   precision and recall for each team, with ground-truth examples of both teams.
   Save profile, input SHA-256, annotations and metrics.
5. Inspect the overlay and record qualitative failures. Resolve material errors
   or report the milestone incomplete. Generated footage alone cannot satisfy this gate.

Performance is recording-specific until evaluated on other matches. If held-out
errors guide recalibration, reserve fresh evaluation rounds for the next run.

## Later work and exclusions

Keep `A_DEFAULT`, `B_DEFAULT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`. Finalise operational
route/timing rules before large-scale tactical annotation. Synchronise `.dem` ground
truth with video without making telemetry an inference dependency. Treat missing
visibility separately from actual deaths.

Use colour detection as a measured baseline and optional annotation proposer for
CNN training. Add tracking with validity masks before tactical features. Report
per-class metrics and the oracle-versus-vision gap on real data.

Automatic replay recording, live capture, additional maps, raw-video tactical models
and a dashboard are outside this milestone. Automate whole-match capture when its
cost becomes a demonstrated bottleneck. Do not add model complexity before validating data.
