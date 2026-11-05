# Labelling Guide

## Get the videos (do this once)

The match videos are too big for GitHub, so they come as ZIP files instead.

1. Download the match ZIPs (Furia, Spirit, NaVi) from the shared folder.
2. Right-click each ZIP → **Extract All…** → choose the `data/processed/recordings/`
   folder inside the project. Create the folder if it does not exist.
3. Check the result. Each match should be **one folder**, with the videos directly inside:

   ```text
   data/processed/recordings/
       furia-falcons-mirage-cologne-major-2026/minimap-10fps.mp4
       spirit-falcons-mirage-cologne-major/minimap-10fps.mp4
       navi-falcons-mirage-cologne-major/minimap-10fps.mp4
   ```

**Common mistake:** a folder inside a folder with the same name, e.g.
`navi-falcons-.../navi-falcons-.../minimap-10fps.mp4`. Fix it by moving the inner
folder up one level and deleting the empty outer one. If the labeller says it
cannot find the match, this is almost always why.

## Open the window

From the repository root, with the project environment active:

```powershell
python -m stratscout label-segments --match furia --reviewer yourname
python -m stratscout label-segments --match spirit --reviewer yourname --round 7
```

- Use the **same short name every time**. Each reviewer gets their own file,
  `labels/<match folder>.<reviewer>.segments.json`, so two people can label the same
  clips independently without overwriting each other.
- **Click the window once** so it receives key presses. Everything uses the keyboard.
- **Only open one window per match and reviewer at a time**, or saves can overwrite
  each other.

## Quick start

1. Watch the **yellow** dots: setup first, then press **`I`** when they move again towards a site.
2. Press **`O`** when the plant finishes, the attack ends, or the last attacker dies.
3. Press **`1`–`4`** for the class, then **`S`** to save.
4. Bomb planted? Watch the **blue** dots: if they try to win the site back, `I`, `O`,
   **`5`**, then **`A`/`B`**, `S`. If they stay away to save: `I` (plant), `O` (explosion
   or last death), **`E` `E`**, `S`.
5. Press **`C`** when the round is done, then **`]`** for the next round.

```
MOVE     Space play/pause   ←/→ -/+2 s   , . one frame
JUMP     [ ] previous/next round start   N/P saved segments   M bookmark  J next bookmark
MARK     I start   O end
CLASS    1 A_DIRECT  2 B_DIRECT  3 A_SPLIT  4 B_SPLIT  5 RETAKE (+ A/B site)
         0 undecided   U uncertain
EXCLUDE  E exclude (repeat: next reason)   E E = save, after a plant with no retake
SAVE     S save   X clear the draft   C round checked   Q quit
DELETE   Z Z last saved   D D the segment marked >   (press twice)
```

**Fixing mistakes**

| Mistake | Fix |
|---|---|
| Pressed `I` or `O` at the wrong time (not saved yet) | Go to the right time and press `I` or `O` **again**. It replaces the old time |
| Wrong class or site (not saved yet) | Press the right key. The last key pressed counts |
| Want to start the segment over (not saved yet) | `X` clears the draft |
| Already saved with `S` | Go inside it (`N`/`P`, marked `>`), press `D` `D`, then `I`, `O`, class, `S` again. For the last one saved, `Z` `Z` also works |

Details: [using the labeller](#using-label-segments) and the class and boundary rules in the [tactic guide](tactic-definitions.md).

## Rules and tool status

The [tactic guide](tactic-definitions.md) defines the five classes, segment boundaries,
review states and dataset rules. This guide only explains how to use the tools; if it
ever seems to disagree with the tactic guide, the tactic guide wins.

**Tool status.** `label-segments`, `review-segments` and `export-segments` cover the
pilot: five-class segments, `pending`/`excluded`/`reviewed` status, autosave, deletion,
resume and a match-split clip export. They do not yet record participating approaches,
entry times or notes, or compare two reviewers' files. Do not use the legacy
`RoundLabeller` or the old `config.py` round labels for new labels.

## Using `label-segments`

This is a practical guide to the labelling window; [open the window](#open-the-window)
and the [quick start](#quick-start) are at the top of this document.

### Keys

| Group | Keys | What they do |
|---|---|---|
| **Move** | `Space` | Play / pause |
| | `←` / `→` | Back / forward 2 s, and pause |
| | `,` / `.` | Back / forward one frame |
| **Jump** | `[` / `]` | Previous / next round, at the moment its players can start moving |
| | `N` / `P` | Next / previous saved segment, to rewatch it (`>` marks it in the text) |
| | `M` / `J` | Add a bookmark here / jump to the next bookmark |
| **Mark** | `I` / `O` | Segment start / end at the time on screen. Press only at the exact moment |
| **Choose** | `1` `2` `3` `4` `5` | `A_DIRECT`, `B_DIRECT`, `A_SPLIT`, `B_SPLIT`, `RETAKE` |
| | `0` | Undecided: saved `pending` for another reviewer. Don't force the nearest class |
| | `A` / `B` | Planted site. Required for `RETAKE`; attack classes already imply their site |
| | `E` | Exclude; press again for the next reason: `E` fake (before the real attack), **`E` `E` save** (only after a plant, when the defenders don't try to retake), `E` `E` `E` unusable, `E` `E` `E` `E` other. Check the Draft line |
| | `U` | Mark the start or end as uncertain |
| **Save / fix** | `S` | **Save the segment to the file immediately** (`Enter` also works) |
| | `X` | Clear the segment being marked and start again |
| | `C` | Mark this round as checked (watched completely); press again to undo |
| | `Q` | Quit. Saved segments are already in the file; an unsaved draft is lost |
| **Delete** | `Z` `Z` | Delete the last saved segment. **Press twice**: the first press only asks |
| | `D` `D` | Delete the saved segment you are watching (marked `>`; use `N`/`P` to get there). **Press twice**, then label it again: `I`, `O`, class, `S` |

Deleting needs a second press of the same key; any other key cancels it, and a
`reviewed` segment is never deleted.

Choices (`1`–`5`, `0`, `E`, `U`, `A`/`B`) can be changed freely before `S`; the last key
pressed wins. There is no edit for a saved segment: go inside it (`N`/`P`), press `D` `D`
(or `Z` `Z` for the last one saved) and label it again.

### When to press `I`: setup versus final push

The attackers usually move twice. Only the second movement starts the segment:

```
[spawn] --move--> [stop and wait, often 20-60 s] --move again--> [site fight] [plant/end]
          setup: going to positions                ^ final push: press I here
```

- **Setup (do not mark):** players leave spawn and take positions such as palace,
  apartments or mid, then mostly hold still.
- **Final push (press `I`):** the attacking group leaves those positions and moves
  towards one site without turning back.
- **Easiest method:** play until the site fight or plant, step back with `←` to where
  the group was still waiting, then step forward with `→` or `.` to the moment it
  starts moving. A fast rush that never stops starts when the group leaves spawn.
- Use the yellow numbers to follow players. Watch the setup to know who comes from
  which route, but choose the class only from the final push: a player who stays in
  mid does not make a split; a player who pushes from mid towards the site does.

### One round, step by step

1. **`]`** jumps to the next round; **`Space`** plays.
2. Watch the **yellow** (T) dots through the setup. Press **`I`** at the final push.
3. Press **`O`** at whichever comes first: the plant **finishes**, the attack is clearly
   abandoned, or the round ends (for example the last T dies).
4. Choose the class from the routes used in the push: `1` ramp and/or palace only;
   `2` apartments only; `3` ramp/palace plus mid → connector; `4` apartments plus
   mid → short; `0` if you cannot tell.
5. Check the orange Draft line, then press **`S`**. A green "Saved #…" confirms it; a red
   message says what is missing.
6. **If the bomb was planted,** watch the **blue** (CT) dots: `I` at their first clear move
   to contest the site, `O` at the defuse, explosion, last CT death or clear
   abandonment, then `5`, `A`/`B`, `S`. If they stay away to save: `I` at the plant, `O`
   at the explosion or last death, `E` `E` ("no retake attempt (save)"), `S`. An
   excluded segment still needs `I` and `O`.
7. Press **`C`** when the round is fully watched, then `]` for the next one.

A stationary badge is not proof of a plant: a ✕ next to it means the carrier died and
dropped it. A planter walking away alive from a badge on site is a strong sign; if in
doubt press `U`. A round may have several segments (for example an excluded fake and
the real attack) or none (no attack).

| Situation | Keys |
|---|---|
| Push through ramp + palace, plant finishes | `I` … `O` `1` `S` |
| Apartments push plus one player mid → short at the same time | `I` … `O` `4` `S` |
| Then 3 CTs return together to B | `I` … `O` `5` `B` `S` |
| Fake A, rotate, real attack through apartments | `I` … `O` `E` `S`, then `I` … `O` `2` `S` |
| Bomb planted, defenders stay away and save | `I` (plant) … `O` (explosion) `E` `E` `S` |
| Cannot tell whether the mid player joined | `I` … `O` `0` `S` |

### The label file

Segments are written to the JSON file as soon as you press `S`; there is nothing to
push first. Open it in an editor to read it, but do not edit it by hand: delete with
`D` `D` (or `Z` `Z`) and label again. Each record stores the match ID from `manifest.json`, start and end in
seconds of the prepared normal-speed video, the estimated round, class, site, status,
reviewer, definition version and any exclusion reason or uncertainty flag. Opening a
file for a different match is refused. During the pilot, do not share files until both
reviewers have finished, so the comparison stays independent.

## After labelling: mark reviewed and export clips

Only `reviewed` segments are exported for training. Once the team agrees a match's
labels, mark them (undecided and excluded segments are refused, and reviewed segments
can no longer be deleted in the labeller):

```powershell
python -m stratscout review-segments --match furia --reviewer jx --by gin --all-labelled
python -m stratscout review-segments --match navi --reviewer jx --by gin --ids 3 5 9
```

`--all-labelled --except 7 12` leaves the listed segments pending. Then cut the clips:

```powershell
python -m stratscout export-segments --test navi --output data/processed/datasets/2026-11-01
```

The output folder holds `train/` and `test/`, each with its own `manifest.csv` (one row
per clip: class, match, round, times, source video) and `clips/<class>/*.mp4`, plus a
shared `export.json` (counts, skipped segments, label-file hashes). Every clip of a
`--test` match goes to `test/`; all others go to `train/`, so no match is shared between
splits and training code can be given `train/` alone. Choose validation folds from the
train matches only, by whole match. `--include-pending` exports unreviewed labels for trial
runs only; `--no-clips` writes just the manifest. The output folder must be new: use a
new dated folder for each export. To share the dataset, zip the whole output folder;
the README explains [where a received dataset goes](../README.md#where-files-go).
