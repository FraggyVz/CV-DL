# Development Direction

## Implemented base

Both notebooks call the modular Python package; see [architecture](architecture.md).
Reusable algorithms and checkpoints remain intact. The [baseline review](base-review-2026-10-12.md)
still applies: tracking validity, annotation safety, schema alignment and real-match
evaluation remain unresolved. Synthetic scores do not establish model quality.

## Recording and processing milestone

Use [the recording guide](recording-setup.md) and `process-recordings` for new footage.
The legacy `ingest --stage radar` interface remains available. Record each match once,
then derive **5, 10 and 30 fps** variants. Recording is manual in OBS; future masters
use 2560×1440 output at 60 fps with constant 2× demo playback. The batch speed prompt
defaults to 2. Speed cannot be inferred from the FPS header. On 14 October the user
requested removal of old recordings, exports, capture profiles and unlabelled caches.
The replacement recording must be calibrated and validated after OBS stops.

Implemented: crop profiles, single-decode FFmpeg branches, source hashes/match IDs,
timestamp offsets, overwrite refusal, complete decoding checks, side-by-side previews,
stills and HTML/JSON storage/timing reports. The OBS controller, command, config and
dependency have been removed at the user's request. Processing starts from a saved
video and does not require WebSocket access. The inbox command now chains crop
selection, preparation and automatic comparison windows, with safe skipping of
completed sources and recovery of comparisons that have not started. Speed correction
now happens before sampling; terminal prompts collect speed and trim boundaries
in raw-video seconds. Output manifests retain the answers and match/source identity;
no per-recording settings files are needed. Reports map normal-speed output
back to the accelerated source. Raw recordings are never overwritten.

Tests cover sampling, duration/colour/crop preservation, invalid inputs, interruption
and Matroska timestamp quantisation. Current acceptance is described
in [processing validation](processing-validation.md). No automated capture test is
required. A short manual capture/crop check is still needed when the layout changes.
Inspect game rendering FPS as well as OBS output FPS: a 60 fps file can contain
repeated rendered frames, so the header alone cannot establish observation rate.

## Spectator-radar detection milestone (`bomb-state`)

Implemented on three processed matches: C4 badge detection by colour then shape;
carried/dropped from the carrier's rim and arrow at the badge edge; dropped-bomb
candidates; video-only round estimates; per-team icon masks with ✕ removal, disc-fit
splitting and a five-per-team cap; `icon-count-eval` (fit on Furia/Spirit, test on NaVi);
and the `bomb-preview` window. Details and results: [bomb detection](bomb-detection.md),
[overlapping icons](icon-split.md).

Remaining before claiming accuracy: hand-labelled live-play frames for per-team
precision/recall (including carriers, overlaps and ✕ markers), a planted state, tracking
for spawn stacks and fully hidden players, and demo files to confirm bomb events.
This feeds step 4 below.

## Ordered work and completion criteria

1. **Check new manual captures:** verify the live CS2 preview, record 30 seconds,
   inspect the master and calibrate its crop. Run `prepare-video` and inspect the
   exports before a whole-match recording. No scripted OBS start/stop is needed.
2. **Protect and align labels:** preserve reviewed labels, separate proposals/review,
   retain empty frames and reconcile ingest/main pre/post-plant schemas. Finish this
   before substantial labelling or automatic proposal generation.
3. **Create match-separated data:** retain masters/demos and match IDs; add round
   manifests, timestamps and explicit demo/video synchronisation. All FPS variants
   of a match must stay in one split. Cropped videos alone are not labelled data.
4. **Measure detection and rate trade-offs:** use reviewed centres/visibility,
   held-out rounds/matches, per-team precision/recall and explicit bomb/death/overlap
   cases. Compare 5/10/30 fps; do not select a winner from file size alone.
5. **Repair tracking and features together:** propagate observed/missing/inferred
   states, remove invented players, and compute motion using actual elapsed time.
   The old 32-frame sequence is not a constant-FPS sampling contract.
6. **Train/evaluate real tactics:** connect reviewed real pre/post-plant rounds, fix
   leakage, compare colour/CNN baselines, then evaluate the classifier heads.
7. **Application/deployment:** build around measured model behaviour and clear failure
   states. CS2 launch, playback, camera and match-ending automation are separate work,
   justified when manual capture becomes the bottleneck.

Lower rates reduce frame counts, not storage in a guaranteed ratio. Preserve native
spatial resolution and full-screen context for round/objective annotation. This
milestone measures visual quality, timing, storage and processing cost only.
