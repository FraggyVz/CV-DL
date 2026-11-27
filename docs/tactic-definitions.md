# Tactical Clip Definitions and Labelling

## Scope and implementation status

The agreed target is five classes on Mirage: `A_DIRECT`, `B_DIRECT`, `A_SPLIT`,
`B_SPLIT` and `RETAKE`. This document is the source of truth for their meaning.
Timing and participation rules below are working annotation conventions: trial them
on 15–20 clips before freezing a version and labelling at scale.

**Tool status.** `label-segments`, `review-segments` and `export-segments` implement
this specification for the pilot; the [labelling guide](labelling-guide.md) explains
how to use them. Not yet implemented: participating approaches, approach-entry times
and notes in the record, and reviewer comparison. Draft map regions for model features
(not for labelling) are in `stratscout/mirage_regions.json`; see
[shared clip data](clip-data.md#map-regions). The legacy
`RoundLabeller` and the old `config.py` round labels keep the attacker/defender schema;
do not use them for new labels or as the five-class mapping.

## Keyboard labelling workflow

Navigate round by round, but save
individual attack and retake segments within each round. Map-region polygons support
overlays and route features; they do not supply tactical labels and need not block the
initial labelling pilot. Do not draw areas or mark every player on every frame for this
task; player-position annotation is separate detector/tracker validation work.

The workflow is:

1. **Watch the round with context.** Use playback, seeking and frame stepping to identify
   attacks and retakes. Check estimated round boundaries against the recording.
2. **Mark a segment's start and end** using the rules below. Keep the surrounding footage
   available for checking approaches; selecting a segment does not delete the rest.
3. **Choose a class.** Keys are `1` = `A_DIRECT`, `2` = `B_DIRECT`, `3` = `A_SPLIT`,
   `4` = `B_SPLIT`, `5` = `RETAKE`. New annotations start as `pending` until checked.
4. **Record exclusions and uncertainty.** Use `excluded` with a reason for clear
   out-of-scope or unusable candidate segments. Leave uncertain cases `pending` for
   another reviewer; do not force the nearest class. Routine freeze time, waiting and
   rotation footage can remain unlabelled without creating an exclusion for every gap.
5. **Save the segment and continue within the round.** A round can contain multiple
   segments. Record that the round has been checked separately from segment approval;
   finishing the viewing pass does not resolve its pending annotations.
6. **Review and approve.** Resolve class and boundary disagreements, then mark agreed
   segments `reviewed`. Export only reviewed segments with one of the five classes to
   training/evaluation datasets. Follow the pilot and match-split rules below.

`label-segments` autosaves, supports deletion and resume, and records match identity,
timestamps and definition version automatically.

## Five class definitions

| Code | Display name | Observable attack pattern |
|---|---|---|
| `A_DIRECT` | Direct A attack | Ts commit towards A through ramp and/or palace, without a coordinated attack through mid → connector. Ramp + palace still counts as direct. |
| `B_DIRECT` | Direct B attack | Ts commit towards B through apartments, without a coordinated attack through mid → short. |
| `A_SPLIT` | A split attack | An attack through ramp and/or palace combines with an attack from mid through connector during the same engagement. |
| `B_SPLIT` | B split attack | An attack through apartments combines with an attack from mid through short during the same engagement. |
| `RETAKE` | CT retake | After a confirmed plant, one or more CTs actively attempt to regain the planted site and enable a defuse. Success is not required. |

`A_DIRECT` and `B_DIRECT` replace the proposed names `A_DEFAULT` and `B_DEFAULT`.
An actual default is an earlier map-control phase and can precede either a direct
attack or a split. It is not a sixth class. Rushes, slow attacks, fakes, saves and
trickles are not additional model classes in this scope. Speed alone never determines
one of the five labels; a quick apartments attack can be `B_DIRECT`.

## Map regions and participation

When adding region features, draw and version the regions once for the fixed radar
layout, using normalised crop coordinates. Keep A ramp, palace, A site, mid, connector,
short, B apartments and B site separate. Add spawn/rotation regions as context.
Define site-approach boundaries so
reviewers use the same positions to judge entry attempts. A first draft for model
features exists (`mirage-spectator-v1`, rectangles in `stratscout/mirage_regions.json`,
documented in [shared clip data](clip-data.md#map-regions)). Polygon tooling and saved
region definitions remain implementation work; these names are not saved polygons.

A split requires active participation from both specified approaches. The second group
must advance towards the site during the attack, not merely occupy mid or watch a flank.
One participating player can constitute the secondary group; do not require five living
players or exactly two players per route. Record participation counts where observable.

For the pilot, use final-approach starts within approximately **10 normal-game seconds**
as a coordination check, alongside the movement sequence. This is a provisional rule,
not a universal CS definition or an automatic label generator. Record the times and
send borderline cases for review. If this tolerance changes, recheck pilot labels.

A player going from A site into connector after entry is not a mid-to-A attacker.
Likewise, a late flank does not retroactively turn a direct attack into a split.
Judge the approach history, not the final positions. Record unusual routes such as
mid-only attacks or window/jungle flanks as pending review rather than guessing.

Region occupancy shows presence, not proven map control, intent or utility use.
Do not infer missing players from the team cap or a fully obscured icon.

## Annotation unit and timestamps

Label a **tactical segment**, not necessarily an entire round. One round may provide
an `A_SPLIT` attack segment and a separate `RETAKE` segment. Earlier map-control play
is context, not a competing attack label.

Use elapsed seconds in the prepared, normal-speed video. Store the source match
identity from `manifest.json`, the round reference and video timestamps. Do not apply
2× correction again. Record freeze-time ending separately as the live-round start;
first observed player movement is only an estimate if the actual start is unavailable.

For the pilot, mark:

- **Attack start:** the first clear final advance towards the site by a participating
  group. Review the preceding footage to identify the route; do not include the whole
  default/setup phase. Apply the same rule to direct and split attacks.
- **Attack end:** confirmed completion of the bomb plant, clear abandonment of the
  attack, or actual round end, whichever happens first. Do not end the segment when
  the first player exits ramp/palace/apartments or crosses onto the site: entry and
  the following site fight belong to the same attack. Planting must finish; starting
  the plant animation is not the endpoint. Apply this rule to all four attack classes.
  A recognisable attack stopped at an entrance can still receive its route class;
  success or actual site entry is not required.
- **Retake start:** the first clear CT advance to contest the planted site after the
  plant. If CTs are already contesting at the plant, start there.
- **Retake end:** confirmed defuse, explosion, elimination of all CTs, or clear abandonment.
  A lone CT or staggered attempt can be `RETAKE`; staying away to save is excluded.

A clear A fake followed by a committed B apartments attack yields an excluded fake
segment (reason: outside current tactic scope) and a separate `B_DIRECT` segment.
Keep the fake and rotation footage as context; do not merge them into the B attack.
An unsuccessful A attack is not automatically a fake just because Ts later attack B:
keep a clearly identifiable committed A attack as its own labelled segment. Ambiguous
feints remain pending review. Do not label a segment solely from where the bomb eventually
gets planted. No-attack rounds and incomplete clips need not enter the five-class dataset.

Verify plants and round endings from the full recording's HUD/audio or aligned demo
evidence. A stationary badge does not prove a plant; the last T death does not always
end a round with a planted bomb. If the evidence is unavailable, mark the boundary
uncertain. Supporting demo truth is for annotation/evaluation, never inference input.

## Annotation record

Each record holds match ID, estimated round, start/end seconds, class, site, status,
boundary uncertainty, any exclusion reason, reviewer and definition version, plus
`reviewed_by`/`reviewed_at` once approved. Participating approaches, approach-entry
times and notes are still to be added. Keep class separate from review status:

| Status | Class field | Dataset use |
|---|---|---|
| `pending` | Proposed class, or unset if undecided | Await review; do not export |
| `reviewed` | One agreed class from the five | Eligible for training/evaluation under the match split |
| `excluded` | Unset for out-of-scope material; retain any previous proposal in the review history | Do not export; record a reason |

These are review states, not extra classifier outputs. Store uncertain boundaries
explicitly; resolve them before approval or exclude the unusable segment with a reason.
Track round viewing progress separately so an intentionally empty round is distinguishable
from one nobody has checked. Unlabelled footage is not automatically reviewed background.

Preserve original labels and create a versioned migration output. **Do not automatically
rename old labels:** old `A_SPLIT` examples may include ramp + palace and old `DEFAULT`
examples may lack a committed site. Existing rush/trickle/save labels need video review,
not a string substitution. Keep proposals separate from reviewed records.

## Worked examples

### Attack boundaries

Illustrative `A_DIRECT` segment, using elapsed prepared-video time:

| Time | Event | Annotation action |
|---|---|---|
| 00:40 | Ts begin their committed final advance through ramp/palace | Mark attack start |
| 00:45 | First T exits ramp | Continue the segment |
| 00:49 | Ts fight on A site | Continue the segment |
| 00:58 | Bomb plant completes | Mark attack end |

Save **00:40–00:58 as `A_DIRECT`**. Any subsequent retake is a separate segment.
If the attack is clearly abandoned or the round ends before the plant, use that earlier
event instead. A brief pause or one player's retreat does not by itself prove abandonment;
flag an uncertain boundary for review.

### Class decisions

| Observed sequence | Decision |
|---|---|
| Three Ts through ramp and two through palace, no mid-side group | `A_DIRECT` |
| Ramp/palace group attacks while teammates advance mid → connector | `A_SPLIT` |
| Four Ts attack through apartments while one holds mid | `B_DIRECT` |
| Apartments and mid → short groups join the same B attack | `B_SPLIT` |
| An apartments attack is stopped before entering B | `B_DIRECT`, if the commitment is clear |
| Ts take mid control, then regroup and attack only through apartments | `B_DIRECT` for the later attack segment |
| Clear A fake, rotation, then a committed B apartments attack | Exclude the fake; label only the later attack `B_DIRECT` |
| Committed A attack fails, Ts withdraw and later attack through B apartments | Separate A attack and `B_DIRECT` segments; failure alone does not make the A attack a fake |
| One CT attempts to regain the planted site but dies | `RETAKE` |
| CTs stay away to preserve weapons | Excluded: no retake attempt |
| The second group arrives much later or its route is obscured | Pending review |

## Review, evaluation and future scope

Two reviewers independently label the same 15–20 varied segments. Compare class and
boundary disagreements, clarify the rules, then freeze a definition version. Do not
use “pick the closest” as a substitute for review. Check class coverage before collecting
more footage; neighbouring frames do not count as independent tactical examples.

Keep every clip and FPS variant of a match in the same train/validation/test split.
Agree splits before tuning; retain the documented held-out match for detection work.
Compare a simple region/route baseline with a learned temporal model. Report per-class
precision/recall, macro-F1, confusion matrices and failure examples on unseen matches.
Manual boundaries constitute segmented-clip evaluation, not automatic discovery of
attacks in an entire match; evaluate boundary detection separately if it is later added.
Do not train excluded or unlabelled footage as an automatic sixth/background class.
A five-class model trained only on selected attacks/retakes cannot be assumed to reject
arbitrary footage. Whole-match scanning needs a separately designed and evaluated
background/rejection approach, with appropriate reviewed examples.

Preserve elapsed time when sampling model inputs. Rush tempo or an actual default
opening may be future attributes/phases, not additions needed to inflate the class count.
Other maps require their own region and route definitions. See the
[README next steps](../README.md#next-steps) for implementation gaps and the
[project structure](development.md#project-structure-and-contracts) for module responsibilities.
The [recording guide](recording-setup.md) covers capture and preparation.
