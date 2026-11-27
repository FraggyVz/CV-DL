# Detector check

The check compares the colour detector's player positions (the detector `bomb-preview`
draws) with the true positions recorded in the demo files. It answers one question: can a
later position-based tactic model trust these positions? It was first run on Furia and
Spirit, then on Aurora-Vitality and MOUZ-BetBoom. The test matches, NaVi and
Vitality-Falcons, are refused by name.

Demo positions are **evaluation evidence only**. They never become detector or model input.

## Inspect the submitted verification

The [four-match report](evidence/detector-check-2026-11-27/report.md) comes from a
fresh verification on 27 November 2026 using the detector at `3d83112`. The
[results notebook](../notebooks/detector-check-results.ipynb) reads the included
summary, calibration curves and player-frame measurements without account IDs. Run it top
to bottom in the project’s Python environment to reproduce the tables and figures.
The [input checksums](evidence/detector-check-2026-11-27/input-sha256.txt)
identify the four demos and four prepared videos used for verification.

## Repeat the check from footage

1. Put the prepared match folders in `data/processed/recordings/`; see the
   [labelling guide](labelling-guide.md). The check uses the whole `minimap-10fps.mp4`
   for each match.
2. Put the matching demos in `demos/`. Each file name must start with the match name,
   for example `furia-vs-falcons-m1-mirage.dem` and `spirit-vs-falcons-m2-mirage.dem`.

```powershell
python -m stratscout detector-check --matches furia spirit aurora mouz --output runs/detector-check-new
```

Use a fresh output folder. Open its `report.md` or set `RUN` in the results notebook
to that folder. A raw rerun also writes the per-detection table and player IDs. The
submitted notebook uses the team, x and y coordinates, speed, distance to the
matched detection and whether the player was found.

## What it does

| Step | What happens |
|---|---|
| Pair rounds | Video rounds (estimated) and demo rounds are paired in order; warm-up rounds are dropped. The counts must agree. |
| Line up time | A straight line from demo time to video time (the clock) is fitted on the start of every complete round; then a small offset is tuned on odd rounds. |
| Line up the map | Demo coordinates are scaled and shifted onto the radar crop. |
| Score | The detector runs on one frame per second; each team's detections are paired with its alive players. |

The clock uses every round's start time, from the demo and the video, not the detector.
**Odd rounds** fit only the small time offset and the map; **even rounds** are scored.
Calibration error and detector error are therefore reported separately.

## How to read the results

| Measure | Meaning | Pass mark |
|---|---|---|
| Found | Share of alive players with a detection within 12 px | at least 90% |
| Extra | Share of detections that match no alive player | at most 10% |
| Median / 90th pct | Distance between a player and its detection, in radar pixels | at most 6 / 12 px |

A player icon is about 18 px across. Errors use pairs within 36 px, so the 90th percentile
is not cut off at 12 px. Each team must pass all four measures. The verdict is
**INCONCLUSIVE** when the calibration itself is unreliable (see its flags in the report).

The report also splits results by moment, without pass/fail: spawn stacks (first 5 s),
overlapping icons, isolated players, the bomb carrier, and still or moving players. The
"floor" in the calibration table shows how well still, isolated players line up: it is the
error left over from the mapping, not from the detector.

## Limits

- It does **not** check the tracker (keeping player identity over time), which the README
  lists as unrepaired. A position-based model still needs that fixed or avoided.
- The CNN detector, the test matches (NaVi, Vitality-Falcons), freeze time and post-round
  frames are not scored.
- Hidden carriers inferred from the C4 badge count as T detections. In the 2026-11-11 run
  they were 21% of T detections; counting observed icons only, T found falls to about 76%.
  A position-based model must take the inferred carrier too, marked as inferred.
- In the same 11 November two-match run, at least one player was missing in 12% of team-frames, so
  a model reading a sequence of frames must tolerate missing players.
- The pass mark is pooled over the checked matches. Single rounds vary (in the 2026-11-11
  run, Furia round 2 had T found at 80.5%) and that run scored only 20 rounds.
- The time offset and map fit use detector centroids on odd rounds, so a detector that was
  off in the same way everywhere would partly be absorbed by the fit. The clock line does
  not use the detector.
