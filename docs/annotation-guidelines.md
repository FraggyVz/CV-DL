# Annotation Guidelines

## Detection labels

Annotate the centre of every **visibly identifiable live-player icon**, assigning
it to **T or CT** using the spectator display. Side is not roster identity, icon
number or the observed player's teammate colour. Verify the mapping from the recording.

- Use the coloured icon core rather than its facing cone, text or numeral.
- Exclude bomb/objective icons, map/site labels, utility and recognisable death
  markers. Returning these is a detector false positive.
- Include overlapping icons when separate centres remain identifiable. Do not
  invent centres for an inseparable blob or infer hidden players from `.dem` data.
- Do not silently omit hard examples. Evaluate scorable difficult cases and retain
  an audit list of unscorable cases with reasons. The scorer has no ignored-region
  matching; keep fully ambiguous frames in the qualitative audit.
- An empty team has `[]`. Missing entries mean missing work, not zero players.
  Counts need not equal five. Every accepted frame has both T and CT lists.

## Coordinates, timestamps and review

Use the profile crop `[x0, y0, x1, y1]`, with exclusive right/bottom edges. For a
centre at source pixel `(px, py)`:

```text
x = (px - x0) / (x1 - x0)
y = (py - y0) / (y1 - y0)
```

Use native pixels before resizing. Coordinates increase right/down. Do not crop
already cropped frames again. Annotate raw frames, not detection overlays that
obscure the evidence. Measure centres with a pixel-coordinate image viewer/editor
and convert them using the formula above.

Record zero-based source frame indices. For constant-frame-rate video,
`timestamp_s = frame_index / source_fps`. Video time starts with the recording,
not round start or demo tick zero. Store replay/video offsets separately.

Use `pending`, `reviewed` or `ambiguous` status in working annotations. Only
`reviewed` frames enter quantitative evaluation. A reviewer must check every icon
and both teams before marking a frame reviewed. Retain proposal source and reviewer
in optional metadata. Never copy model predictions into ground truth unchecked.

Historical teammate-only labels are not T/CT ground truth. Do not import them from
Git history by simply renaming their team channel. New annotations must follow the
two-team schema below and record explicit review status.

## Split before tuning

Select at least 30 frames across at least three rounds: 10 calibration frames and
at least 20 evaluation frames from **different rounds**. Include both teams,
movement, empty/low-count frames, close players and objective/death indicators.
Record the chosen frame/round list before tuning. Do not adjust thresholds using
evaluation frames.

This is a within-recording pilot. Future training, validation and test datasets
must be split by match; frame shuffling or an adjacent time split is insufficient.
If tuning uses evaluation errors, choose fresh evaluation rounds.

## Evaluation file and command

Create `data/annotations/mirage-review.json`. Copy `video_sha256`,
`source_resolution` and `crop` from the detection run metadata. Use this structure
(illustrative and too small to meet the acceptance protocol):

```json
{
  "schema_version": 1,
  "video_sha256": "replace with run.json input.sha256",
  "source_resolution": [1920, 1080],
  "crop": [20, 30, 320, 330],
  "reviewer": "reviewer name",
  "frames": [
    {
      "frame_index": 120,
      "round_id": "match_001_round_01",
      "split": "calibration",
      "review_status": "reviewed",
      "T": [{"x": 0.25, "y": 0.60}],
      "CT": []
    },
    {
      "frame_index": 6000,
      "round_id": "match_001_round_03",
      "split": "evaluation",
      "review_status": "reviewed",
      "T": [],
      "CT": [{"x": 0.70, "y": 0.30}]
    }
  ]
}
```

```powershell
python -m stratscout evaluate --run runs/two-team-001 --annotations data/annotations/mirage-review.json --output runs/two-team-001/evaluation.json
```

Use the project's virtual-environment interpreter. The evaluator rejects mismatched
provenance, duplicate indices, missing teams, non-reviewed frames and round overlap.
Only evaluation frames contribute to metrics.

Matching is one-to-one within `0.02 * sqrt(crop_width^2 + crop_height^2)` native
pixels. Duplicate detections cannot claim the same player twice. Per team:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

Require at least 90% for both metrics for both teams, the required sample size and
ground-truth examples of each team. Undefined metrics are zero and cannot pass.
Inspect hard-case overlays as well. Do not claim generalisation beyond this recording.

## Tactical labels: later milestone

Preserve the five classes from the project README v2:

| Class | Intended definition |
| --- | --- |
| `A_DEFAULT` | Methodical attack committing to A without satisfying the coordinated-split definition |
| `B_DEFAULT` | Methodical attack committing to B without satisfying the coordinated-split definition |
| `A_SPLIT` | Coordinated attack approaching A through at least two distinct routes |
| `B_SPLIT` | Coordinated attack approaching B through at least two distinct routes |
| `RETAKE` | Surviving defenders regroup and return towards a site after attackers take control or plant |

These meanings are fixed; operational annotation rules are **not finalised**.
Before large-scale tactical labelling, agree and version: Mirage route/region
boundaries, minimum participants, coordination time tolerance, commitment/site-control
criteria, clip start/end rules, and treatment of ambiguous defaults, splits and
rotations. Do not invent these thresholds during implementation.

Tactical windows should record match ID, round, map, acting team/side, recording
start/end times, class, reviewer and guideline version. A round may contain several
behaviours: do not automatically assign one whole-round tactic. Resolve overlapping
windows and ambiguity under the agreed rules; do not add a sixth tactic to hide
uncertainty. Check agreement between two annotators on a subset. `.dem` files
supply positions/events, not semantic tactic labels.
