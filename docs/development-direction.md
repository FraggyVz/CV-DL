# Modularisation and Recording Plan

## Status and agreed direction

The lecturer accepts notebooks or individual source files. The reusable code from
both of Ryan's notebooks now lives in `stratscout`, with thin notebooks for experiments,
plots and presentation. See [architecture](architecture.md) for the implemented module
layout. Existing recordings, labels and checkpoints are preserved. The recording-rate
comparison and expanded capture automation below remain planned and unvalidated.

Suggested responsibilities:

- `config.py` and `data.py`: configuration, label schemas, match identities and splits.
- `video.py`: decoding, crop validation, timestamps and derived recordings.
- `detector.py`, `tracking.py`, `features.py`: the existing vision pipeline.
- `training.py` and `evaluation.py`: explicit experiment entry points and metrics.
- `__main__.py`: command-line orchestration; `tests/`: focused regression checks.

Keep imports one-way: shared contracts first, processing modules next, orchestration
last. Importing a module must not launch training, open windows or write data.
Move and verify one component at a time; avoid maintaining duplicate implementations
in the package and notebook. Agree module ownership before each alternating turn.

## Frame rate and storage proposal

Test **10 fps radar-only working videos** before making them the capture standard.
This gives one observation every 100 ms. It is a plausible starting point for
minimap movement, but adequacy for close-player tracking and short events is unproven.
Lower frame rates do not solve bomb occlusion.

The companion notebook currently requests 8 fps radar exports. The main notebook
declares `fps_sampled=4.0`, but round extraction selects 32 evenly spaced frames per
window; this is not a constant 4 fps dataset. Separate recording rate, processing
rate and classifier sequence length. Preserve timestamps and round duration, and
compute speed using elapsed time rather than displacement per sample.

Keep the existing full-screen recording as a reference. Derive compressed,
native-resolution radar crops for routine work; do not shrink already-small icons
to save space. Avoid exporting every decoded frame into a large uncompressed array.
Extract only selected annotation/training frames with their source identities.

Reducing 60 fps to 10 fps gives six times fewer frames to process, not a guaranteed
sixfold reduction in encoded file size. At a fixed bitrate, similar-duration files
remain similar in size. Crop and tune quality-based encoding, checking colour edges
and numbered icons after compression. Keep full-screen context available for round,
plant and winner annotation until an alternative has been validated.

## Automation order

1. Extract video utilities and add a batch preparation command: validate input/crop,
   crop once, resample, encode and write metadata. Refuse overwrites. Metadata should
   retain match ID, source identity, crop, resolution, timestamps and encoding settings.
2. Produce comparable short segments at 10 and 20 fps from the existing 60 fps source.
   Review detections at common timestamps and tracking across each sequence. Include
   deaths, bomb carriers and crossing icons. Record accuracy, runtime and file size.
3. Automate OBS recording start/stop and naming after preprocessing is reliable.
   Loading CS2 demos, applying settings and positioning the spectator camera remain
   separate playback-automation work. A `.dem` still needs rendering by the game.

Record each map once and derive clips afterwards. Do not re-record individual rounds.
Resolve the schema and annotation-safety issues in the [base review](base-review-2026-10-12.md)
alongside extraction, before collecting substantial new labels.

## References

- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html): crop and frame-rate conversion.
- [OBS remote control](https://obsproject.com/kb/remote-control-guide): built-in WebSocket support.
- [OBS WebSocket protocol](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md): recording control requests.
