# Recording-Preparation Validation — 13 October 2026

## Real source and reproducible exports

Input: `data/raw/falcons-vs-furia/Falcons vs Furia Major Final.mp4`.
User-supplied match identity: `falcons-vs-furia-major-final`. The original
2,618,207,526-byte, 2560×1440, nominal 60 fps recording was preserved. Its SHA-256
was checked again after processing and matches the pre-processing value:

```text
55df9095664c1e3b1b2cc052f5719e221912cc73cee9b770a70a5cf7b97a647b
```

Profile: `configs/falcons-furia-video.json`, half-open crop `[16,12,508,510]`, native
492×498. All branches were encoded together from one source decode/crop with libx264
CRF 18, medium, yuv420p, no audio and fast-start. Each output passed a complete FFmpeg
decode and count/rate/dimension/duration validation.

Final local bundle: `data/processed/falcons-furia-fps-002/`; open its
`comparison/report.html`. Generated media are ignored by Git. Reproduction commands
with a fresh destination are in [recording setup](recording-setup.md).

| Rate | Decoded frames | Duration (s) | Bytes | Decode wall time (s) |
|---|---:|---:|---:|---:|
| 5 fps | 13,624 | 2,724.800 | 22,801,451 | 1.252 |
| 10 fps | 27,247 | 2,724.700 | 28,892,909 | 1.652 |
| 30 fps | 81,739 | 2,724.633 | 40,936,245 | 3.044 |

Shared conversion took **102.670 seconds**; encoding plus output validation took
**110.649 seconds**, excluding preflight/hashing and report generation. Timings are
one local run, affected by cache, load, threading and process start-up. They do not
measure detector inference. FFmpeg/ffprobe build: `2026-02-26-git-6695528af6-full_build`
from the installed Gyan build. Python 3.13.14; OpenCV 4.12.0; NumPy 2.2.6.

The earlier `fps-001` run is retained. The final run uses nearest timestamp rounding,
also tested against OBS-like millisecond-quantised MKV timestamps. Both runs produced
the same frame counts and file sizes on this MP4 source.

Source header duration is 2,724.650 seconds. End rounding differs by output frame
interval; the 30 fps export ends about 0.017 seconds earlier. Earlier OpenCV checks
could not read the source's final advertised frame. FFmpeg reported no fatal decode
error in this conversion; do not infer exact demo-tick alignment from header counts.

## Development comparisons

Isolated pilot bundles (`runs/fps-pilot-55`, `-115`, `-190`) and full-bundle comparisons
cover source windows `55:10`, `115:10`, `190:25`. Previews are labelled 30 fps side-by-
side videos with held lower-rate observations, timestamped PNGs and HTML/JSON reports.
The previews themselves passed full decoding checks.

Inspected stills at 60, 120 and 202.5 seconds show the native crop, team colours,
player numbers, death markers, spectator cone and bomb graphics. At 202.5 s the
5 fps panel holds 202.4 s while 10/30 fps show 202.5 s, making temporal loss visible.
Bomb graphics still obscure players; encoding does not resolve occlusion. These
are development frames, not a held-out labelled accuracy evaluation.

## Inbox workflow and 1080p sample

The new `process-recordings` command and Windows launcher were exercised on
`2026-10-13 02-08-54.mkv`: 1920×1080, 30 fps, 65.866 seconds, 11,171,872 bytes.
A copy was placed in `data/raw/inbox/`; the original recording remains in the
user's Raw Demos folder. Both copies had SHA-256
`a6f58a05660b0da6fe431d8dc773e7311c22c39f9f70d3c73d85f99a4ece2d40`.

The crop `[8,12,380,380]` was selected after inspecting the source at 25 seconds,
producing a native 372×368 minimap. The local, Git-ignored profile is
`configs/recording-video.json`; it applies only to this capture layout.
Bundle: `data/processed/recordings/a6f58a05660b0da6/`.

| Rate | Decoded frames | Duration (s) | Bytes |
|---|---:|---:|---:|
| 5 fps | 330 | 66.000 | 531,028 |
| 10 fps | 659 | 65.900 | 690,691 |
| 30 fps | 1,976 | 65.866667 | 955,000 |

All exports and three automatic comparison previews passed full decode checks.
The middle comparison still was inspected for crop coverage, colours and time labels.
Rerunning the actual PowerShell launcher skipped the completed source successfully.
This verifies preparation of the 1080p sample, not detector accuracy or equivalence
to a 1440p master. The sample still contains player numbers and bomb occlusion.

## Checks and remaining acceptance

Tests exercise lossless numbered frames with exact timestamps, 5/10/30 selection,
shared observations, RGB/crop preservation, fractional duration/offsets, preview
holding, invalid profiles/rates/tools/videos, overwrite refusal, interruption and
Matroska timestamp quantisation. Existing notebook/modular regression and import-safety
tests pass. The legacy prototype smoke CLI completed in
`runs/recording-workflow-smoke-001/`.

After adding inbox processing, **35 tests passed**, including first-use profile
creation, real batch encoding/comparison, renamed duplicate skipping, interrupted
processing, missing-comparison recovery, damaged output refusal, profile/resolution
changes and continuing after a corrupt input. Ruff lint/format and `git diff --check`
passed. The prototype smoke also passed in `runs/inbox-workflow-smoke-001/`.

The legacy ingest regression emits its existing missing-OpenH264 warning and succeeds
through its old MP4V fallback. The new processor uses the installed FFmpeg libx264
encoder directly and does not depend on that OpenCV encoder.

The OBS controller and its mock tests were removed when the user chose manual
recording. Automated capture acceptance is no longer part of this scope. For a new
layout, manually record a short clip and inspect its master and minimap crop before
a full match. Processing is validated on the supplied recording; there is still no
measured detector accuracy or selected training frame rate here.
