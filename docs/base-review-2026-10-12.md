# Shared Base Review — 12 October 2026

> This review records the pre-refactor state at `00e024f`. The code has since moved
> into Python modules; see [architecture](architecture.md). Dependencies and regression
> tests now exist. B-prime is disabled by default and experiment checkpoints refuse
> overwrites. The algorithm/data faults listed below remain unless explicitly noted.

## Baseline and working agreement

Use Ryan's prototype as the base for incremental, alternating improvements.
Reviewed `ryan-to-present` at `00e024f`; the remote branch points to the same commit.
It and `gin` diverged from `6755650`. The CLI, profiles, tests and documentation on
`gin` are still available there, but are not part of this branch. Reuse individual
pieces where useful; do not merge the entire previous approach over this prototype.

## Implemented in code

`stratscout_prototype.ipynb` contains automatic radar cropping, background/colour
segmentation, icon and round labelling tools, a two-channel CNN heatmap detector,
fixed-slot tracking, four classifier architectures, oracle/vision comparisons,
occupancy maps, route plots and win-rate aggregation.

The current main notebook asks two separate questions:

- Before plant: `A_RUSH`, `B_RUSH`, `A_SPLIT`, `B_SPLIT`, `DEFAULT`.
- After plant: `RETAKE`, `TRICKLE`, `SAVE`.

`stratscout_dem_ingest.ipynb` adds radar-video extraction, demo parsing, coordinate
mapping and demo round labelling/export. Demo/video synchronisation is still absent.
Preserve the main notebook's video-only inference direction.

## Issues to address

1. **Incompatible label/export contracts.** Ingest still uses the older five-class
   list ending in `RETAKE` and excludes defaults. It exports whole-round positions,
   not the main notebook's pre/post-plant representation. Align these before labelling.
2. **Label safety.** Main §12 `auto_label_icons` overwrites its destination. Neither
   proposal/review separation nor per-frame review status is present. Both saving and
   loading drop empty icon-label frames. Isolated checks reproduced overwrite and loss.
3. **Invented and stale tracking positions.** Main §11 creates `(0.5, 0.5)` positions
   for never-seen slots, backfills late arrivals and retains disappeared players after
   the miss limit. Reproduced with empty inputs and a single disappearing player.
   Missing/observed/inferred state needs to propagate into features and plots.
4. **Missing-data features.** Main §13 uses ordinary means/differences on positions.
   One slot becoming NaN halfway through a 32-frame input produced 96 non-finite
   feature values. Demo deaths already produce NaNs, so the ingest adapter alone is
   insufficient for real oracle training.
5. **Evaluation leakage.** Main §15 selects the B-prime checkpoint using
   `test_rounds[:20]` then evaluates on all `test_rounds`. Main §10 splits real icon
   frames at 80% without enforcing match separation. Use explicit shared split manifests.
6. **Real training not connected.** Main §8 always generates synthetic classifier
   rounds. Real round loading exists later but does not replace that training path.
7. **Stale evidence.** `runs/prototype/summary.json` declares synthetic mode and
   reports perfect classifier scores. Its schema and classes predate the current
   two-head code. These outputs do not establish current real-footage performance.

## Local reproducibility

At the start of this review, README was only a title; no dependency file or tracked
tests are present. The existing
environment has NumPy/OpenCV but lacks Torch, Matplotlib, pandas and demoparser2.
It also emits a Python interpreter-location warning. Rebuild a documented environment.

The two recordings named in `CLAUDE.md` are absent locally. Our Falcons/FURIA source
is preserved under `data/raw/`; the main notebook only scans files directly in `data/`,
so default discovery finds no video. The tracked pilot array has shape
`(400, 198, 198, 3)` but no local source manifest or icon labels. Establish provenance
before using it. Restore ignore rules for generated runs, metadata and environments;
tracked caches/checkpoints need a separate, deliberate clean-up.

Both notebooks parse: 37 main and 8 ingest code cells. Neither has saved execution
outputs. This review used source inspection and isolated function checks, not an
end-to-end notebook run; no training, recording or existing labels were modified.

## Alternating work

1. First turn: document setup, align both schemas, protect reviewed labels, preserve
   empty frames and establish data identities/splits; add focused regression checks.
2. Next turn: create reviewed real icon labels and evaluate a detector on held-out matches.
3. Then: fix tracker validity and feature handling together, including bomb occlusion.
4. Connect real pre/post-plant rounds to training and rerun leakage-free comparisons.

Each turn should begin from the latest shared integration commit, make one bounded
change on a short-lived branch, and finish with checks, a commit and updated handoff
notes. Avoid simultaneous edits to the same notebook. Keep `CLAUDE.md` and README
consistent with the code and distinguish synthetic, development and held-out results.
