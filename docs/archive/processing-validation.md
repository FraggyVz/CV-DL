# Processing Validation

> Archived reference (30 October 2026). Results, commands and status describe the recorded revision, not necessarily the current checkout. Use the [README](../../README.md) for current status and next steps, the [recording guide](../recording-setup.md) for capture, and the [tactic guide](../tactic-definitions.md) for labels.

## Current status — 14 October 2026

The user has reset the recording dataset for a new **1440p/60 fps, 2×** capture.
Old recordings, derived videos, crop profiles, contact sheets and capture-specific
reports/scripts were removed. Earlier full-match timings and screenshots are not
acceptance evidence for the replacement recording.

The reusable processing implementation and synthetic regression fixtures remain.
See [recording setup](../recording-setup.md) for the current prompt-based workflow.

The replacement FURIA–Falcons and Spirit–Falcons recordings have since been processed.
Both manifests report complete 5/10/30 fps exports using the same native 494×500 crop.
Their readable output folders are `furia-falcons-mirage-cologne-major-2026` and
`spirit-falcons-mirage-cologne-major` under `data/processed/recordings/`.
Renaming preserved all 70 files byte-for-byte, and a batch rerun skipped both sources.
The share archive `data/processed/stratscout-processed-matches-2026-10-14.zip`
contains those files and an extraction guide (191,399,489 bytes). ZIP integrity and
each archived file's SHA-256 were checked against the originals.

## Automated coverage

After the capture reset, **47 tests passed**. Ruff lint/format, documentation-link
checks and the synthetic prototype smoke at `runs/capture-reset-smoke-001/` passed.
The launcher correctly reported an empty inbox without relying on deleted assets.
No new recording was processed during these checks.

After fixing odd-sized interactive crop selections, **49 tests passed**, including
two-video crop reuse. Ruff lint/format and the synthetic smoke at
`runs/crop-selection-smoke-001/` also passed.

Readable bundle naming passed **51 tests**, including renamed-source lookup, legacy
hash-folder recovery, name collisions and incomplete-destination refusal. Ruff and
the synthetic smoke at `runs/readable-bundles-smoke-001/` also passed.

Tests cover numbered-frame sampling, common observations at 5/10/30 fps, RGB/crop
preservation, 2× timestamp correction, source-time trimming, fractional duration,
Matroska timestamp quantisation and comparison frame holding. They also cover
invalid profiles/rates/tools/videos, overwrite refusal, interrupted encoding,
prompt validation, cancellation, first-use crops and completed-bundle reuse.
Interactive crop checks cover one-pixel adjustment at image boundaries, unchanged
preview pixels, confirmation before saving and reuse across two different videos.
GUI actions are mocked in these checks; confirm the actual crop visually when running.

Notebook/modular regressions and import-safety checks remain. A synthetic prototype
smoke run checks wiring only. The legacy ingest test can emit an OpenH264 warning
before succeeding through its MP4V fallback; the new processor uses FFmpeg libx264.

## Acceptance for the replacement recording

1. Wait for OBS to finish. Check saved dimensions are 2560×1440 at nominal 60 fps
   and confirm constant 2× playback; inspect motion for repeated/stalled frames.
2. Run the batch, select the useful raw-time interval and review the new minimap crop.
3. Confirm all three exports decode, retain native crop dimensions and play at
   normal speed. Expected duration is retained raw duration multiplied by two.
4. Inspect opening, middle and ending footage plus deaths, close icons, numbers,
   objective graphics and bomb-carrier overlap.
5. Review comparison previews, stills, sizes and timings. Rerun to verify completed
   outputs are skipped and the master remains unchanged.

Automated export validation is recorded in the new manifests; a complete manual
review of the footage above remains pending. Choosing a training
frame rate still requires labelled, match-separated detection/tracking evaluation.
No current detector-accuracy result or winning frame rate is claimed.
