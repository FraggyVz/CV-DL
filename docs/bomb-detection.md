# Bomb Detection

## What the radar shows

On the 494×500 spectator-radar crops from the 14 October matches, T players are yellow
numbered circles (OpenCV hue ≈ 27), CT players are cyan circles (≈ 94) and the bomb is an
orange rectangular "C4" badge of about 16×11 px (≈ 16). **Carried and dropped badges are
identical**: when carried, the badge sits beside the carrier and can cover its icon.

The badge hue is within the colour baseline's `hue_tol` (22) of T yellow, so colour alone
would count it as a player once `icon_area` is raised for this resolution (T icons here
are 220–270 px², above the old pilot-frame limit of 250).

## Shape rule

`segmentation.blob_shape` fills each blob's holes and measures aspect (width/height),
extent (fill of the bounding box) and corner fill (a rectangle fills its corners; a
circle does not). `looks_like_bomb` applies the `Config.bomb_*` thresholds. The colour
baseline now skips badge-shaped blobs, and `bomb.find_bomb` returns them.

Measured on 300 frames from each match, using hue only to name blobs for measurement:

| Feature | Badge median | Icon median | Alone |
|---|---|---|---|
| Aspect ≥ 1.15 | 1.33 | 1.00 | 90% badges, 12% icons |
| Extent ≥ 0.82 | 0.87 | 0.77 | 74% badges, 22% icons |
| Corner fill ≥ 0.30 | 0.45 | 0.14 | 81% badges, 11% icons |
| Polygon vertices | 4 | 4 | no separation at this size |
| Circularity | 0.70–0.77 | 0.60–0.82 | overlapping ranges |

The combined default (aspect ≥ 1.15, extent ≥ 0.80, corner fill ≥ 0.30) found ~72% of
badge blobs with ~0.5% of icon blobs misclassified. Most misses are badges merged with an
adjacent player icon. These are blob statistics, not reviewed detection accuracy.

## Dropped candidates

Carried versus dropped is inferred over time: a badge that stays within 3 px for at least
2 s with almost no T-coloured pixels within 40 px is a dropped candidate.

```powershell
python -m stratscout bomb-scan --video data/processed/recordings/<match>/minimap-5fps.mp4 --output runs/bomb-scan/<match>
```

The output folder must be new. It contains `track.csv` (per-frame badge position and
nearby T pixels), `candidates.csv`, an enlarged crop per candidate and `summary.json`.
Times are normal-game seconds from the trim start, as in the preparation manifest.

First 5 fps scans: Furia–Falcons 3,455/13,516 frames with a badge and 10 candidates;
Spirit–Falcons 4,597/14,304 and 21. Visual inspection found most candidates were lone,
still badges, including several in T spawn. Known false candidates are a still carrier
hidden under the badge with only a sliver of yellow showing. Some lone badges show a
white T-shaped mark above them; its meaning is unconfirmed.

## Limitations and next steps

- Candidates are unreviewed. Confirm states against demo `bomb_dropped`,
  `bomb_pickup` and `bomb_planted` events before treating them as labels.
- Planted bombs are not yet separated from dropped ones.
- Thresholds were measured on two matches with one radar layout; re-measure for others.
- A CNN detector class for the badge should be compared against this rule.
