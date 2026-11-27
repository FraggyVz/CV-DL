# Detector check: colour detector vs demo positions

**Verdict: PASS** (even rounds of furia-falcons-mirage-cologne-major-2026, spirit-falcons-mirage-cologne-major 2026, aurora-vitality-mirage-esl-pro-league-season-24, mouz-betboom-mirage-iem-bejing-2026; test matches not read).

Pass mark per team: median <= 6 px, 90th percentile <= 12 px, found >= 90%, extra <= 10%. A player is found by a detection within 12 px; errors use pairs within 36 px.

## Result per team (all checked matches, scored rounds)

| Team | Players | Found | Extra | Median px | 90th pct px | Pass |
|---|---:|---:|---:|---:|---:|---|
| T | 13871 | 95.5% | 2.2% | 0.8 | 2.2 | yes |
| CT | 13853 | 98.3% | 0.5% | 0.7 | 1.9 | yes |

## Crowded and difficult moments (not pass/fail)

| Team | Moment | Players | Found | Extra | Median px | 90th pct px |
|---|---|---:|---:|---:|---:|---:|
| T | spawn stack | 865 | 88.7% | 1.7% | 1.6 | 3.6 |
| T | after spawn | 13006 | 95.9% | 2.2% | 0.8 | 2.1 |
| T | overlapping | 2816 | 86.0% | n/a | 1.4 | 3.1 |
| T | isolated | 10190 | 98.7% | n/a | 0.7 | 1.9 |
| T | bomb carrier | 2233 | 97.4% | n/a | 1.6 | 2.4 |
| T | still | 3111 | 95.3% | n/a | 0.5 | 1.7 |
| T | moving | 4097 | 94.3% | n/a | 1.4 | 2.9 |
| CT | spawn stack | 865 | 90.2% | 0.0% | 1.7 | 3.5 |
| CT | after spawn | 12988 | 98.9% | 0.5% | 0.7 | 1.7 |
| CT | overlapping | 1300 | 93.8% | n/a | 1.3 | 2.8 |
| CT | isolated | 11688 | 99.4% | n/a | 0.7 | 1.6 |
| CT | still | 3357 | 97.7% | n/a | 0.5 | 0.8 |
| CT | moving | 3982 | 97.3% | n/a | 1.3 | 2.8 |

## Calibration error (kept separate from detector error)

The clock line uses the start of every complete round; the time offset and radar
mapping are fitted on odd rounds only. The floor is still, isolated players after
spawn: it bounds how much of any error comes from the mapping and centroids, not the
detector.

| Match | Anchor max residual s | Offset s | Speed ratio | Pairs | Calibration median px | Floor median px | Floor 90th px |
|---|---:|---:|---:|---:|---:|---:|---:|
| furia-falcons-mirage-cologne-major-2026 | 0.55 | +1.05 | 0.9999 | 7348 | 0.8 | 0.4 | 0.8 |
| spirit-falcons-mirage-cologne-major 2026 | 0.70 | +0.95 | 0.9999 | 6355 | 0.7 | 0.4 | 0.9 |
| aurora-vitality-mirage-esl-pro-league-season-24 | 0.42 | +1.00 | 1.0000 | 6654 | 0.7 | 0.5 | 0.9 |
| mouz-betboom-mirage-iem-bejing-2026 | 0.71 | +1.05 | 0.9999 | 5393 | 0.7 | 0.4 | 0.8 |

Calibration flags: none.

## Not covered

- The tracker (player identity over time), listed as unrepaired in the README.
- The held-out test matches and the CNN detector.
- Freeze time and post-round frames; death-marker timing beyond counting extras.
- Inferred hidden carriers are counted as T detections (2516 here).
- The offset and radar mapping use detector centroids on odd rounds; demo positions are
  evaluation evidence only.
