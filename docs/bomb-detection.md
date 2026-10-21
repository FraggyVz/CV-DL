# Bomb Detection

## What the radar shows

On the 494×500 spectator-radar crops from the 14 October matches, T players are yellow
numbered circles (OpenCV hue ≈ 27), CT players are cyan circles (≈ 94) and the bomb is an
orange rectangular "C4" badge of about 16×11 px (≈ 16). **Carried and dropped badges are
identical.** A carried badge is drawn **on top of the carrier's icon**: only a thin yellow
rim and the carrier's white direction arrow show at its edge. Yellow icons beside the badge
are teammates, not the carrier.

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
those blobs with ~0.5% of icon blobs misclassified. **That figure was biased:** it only
counted blobs whose median hue was already orange, which excluded badges merged with
their carrier.

## Colour split before shape

Over a whole match the badge is on screen in ~99% of frames, but shape alone found it in
only 26% (Furia–Falcons). Diagnosing every miss: 58% of frames failed because the badge
blob had merged with an adjacent icon (usually the carrier), 16% on a single shape
threshold, and ~1% had no badge on screen.

`find_bomb` therefore first keeps only badge-orange pixels (`Config.bomb_hue` 10–21,
saturation and value ≥ `bomb_sv_min` 120), then applies a 3×3 morphological opening to
strip thin anti-aliased fringes such as a fading yellow ✕ death marker. T yellow (~27)
falls outside the band, so a badge touching its carrier remains a separate rectangle; the
shape rule then confirms it.
`bomb-scan --shape-only` keeps the old rule for comparison.

| 5 fps scan | Shape only | Colour then shape | + opening |
|---|---|---|---|
| Furia–Falcons frames with badge | 3,455 / 13,516 (26%) | 11,584 (86%) | 12,769 (94%) |
| Spirit–Falcons frames with badge | 4,597 / 14,304 (32%) | 12,666 (89%) | 13,700 (96%) |

Visual checks of 24 random colour-split detections, and of 24 detections found only after
adding the opening, found all 48 on the badge. Neither
figure is labelled recall or precision; both need reviewed frames.

The colour baseline's own badge exclusion (`find_icons_teams`) still uses shape on the
saturation mask, so a badge merged with its carrier can still distort T detections there.

## Carried or dropped

`carrier_rim` counts T-yellow and white pixels in a 4 px band around the badge rectangle.
At least 4 of either means the carrier is underneath (`carrier_visible`). On Furia–Falcons,
99% of frames where the badge was moving (certainly carried) passed; 93% of frames in the
earlier proximity-rule candidates had no rim pixels at all.

This replaced a first rule that counted T yellow within 40 px. That rule misread a still
carrier hidden under the badge as dropped and a teammate beside a dropped bomb as carried.

## Dropped candidates

A badge that stays within 3 px for at least 2 s with no carrier at its rim is a dropped
candidate.

```powershell
python -m stratscout bomb-scan --video data/processed/recordings/<match>/minimap-5fps.mp4 --output runs/bomb-scan/<match>
```

The output folder must be new. It contains `track.csv` (per-frame badge position, rim
pixel counts and the carried flag), `candidates.csv`, an enlarged crop per candidate and `summary.json`.
Times are normal-game seconds from the trim start, as in the preparation manifest.

With the rim rule, 5 fps scans gave 23 candidates (Furia–Falcons) and 22 (Spirit–Falcons).
Visual inspection found all 45 to be lone badges with no carrier showing, including drops
with a T teammate standing nearby. The earlier proximity rule's hidden-carrier errors were
gone. Some lone badges show a white T-shaped mark above them; its meaning is unconfirmed.

## Live preview

```powershell
python -m stratscout bomb-preview                       # pick a match from a list
python -m stratscout bomb-preview --match spirit --round 7
python -m stratscout bomb-preview --video data/processed/recordings/<match>/minimap-10fps.mp4 --start 330
```

Plays a prepared video (10 fps by default) with live T/CT colour-baseline detections and
the bomb state. When the bomb is carried, the hidden carrier is drawn as an inferred T at the
badge centre ("+ 1 carrier under badge") unless the colour baseline already found a T there.
Overlapping icons of the same team still merge and are missed. The bomb state comes from
`BombStateTracker`, the frame-by-frame form of the candidate rule.
Space plays/pauses at normal speed; → fast-forwards and ← rewinds (press again for 2×, 4×, 8×,
16×); K/L jump 5 s back/forward; N/B step one frame; `[`/`]` jump to the previous/next
round; Q/Esc quits. Fast-forward runs detection on shown frames only but counts skipped
frames for the still timer; rewinding pauses the bomb state, which is rebuilt on resume.
The window is resizable. A median-frame background removes static map art such as the
yellow A/B site letters, which otherwise read as T players.

## Round estimates

`rounds.py` estimates round starts from the video alone: during freeze time (~20 s) nearly
all T colour sits in the T spawn box and all CT colour in the CT spawn box. Freeze runs
of at least 3 s start a round; runs within 40 s merge (timeouts). Footage before the first
freeze is a partial round 1. Results are cached in `runs/round-cache/`.

Both matches gave 21 rounds, with the ~270 s halftime pause at the start of round 13 —
where MR12 halftime belongs — and timeouts of 50–95 s. Both recordings start during
round 1. These are estimates; demo round events are the reference.

Yellow ✕ death markers last about 2 s and can merge with nearby T icons, hiding them.

## Limitations and next steps

- Candidates are unreviewed. Confirm states against demo `bomb_dropped`,
  `bomb_pickup` and `bomb_planted` events before treating them as labels.
- Planted bombs are not yet separated from dropped ones. Badges on a bombsite with a red
  glowing outline and red rings look like the planted state; confirm with demo events.
- Thresholds were measured on two matches with one radar layout; re-measure for others.
- A CNN detector class for the badge should be compared against this rule.
- Bomb-carrier player recall is the main open problem; measure it separately.
