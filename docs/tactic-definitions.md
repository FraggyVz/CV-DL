# Tactic Definitions (draft for team review)

Status: **proposal, not agreed**. Thresholds below are starting points to test on a few
real rounds, then fix before bulk labelling. Labels from different people are only
comparable if everyone applies the same rules.

Times are seconds after the freeze time ends. "Commit" means a player enters the site
area (not just the approach). Judge from the minimap only; use HLTV round history for
the winner.

## Attacker tactic (every round, before the plant)

| Key | Label | Rule (proposed) |
|---|---|---|
| 1 | `A_RUSH` | At least 4 attackers take **one** route and commit to A within 30 s. |
| 2 | `B_RUSH` | As `A_RUSH`, for B. |
| 3 | `A_SPLIT` | Attackers commit to A from **two or more** routes (e.g. ramp and palace, or ramp and connector) within about 10 s of each other, with at least 2 players on each route. |
| 4 | `B_SPLIT` | As `A_SPLIT`, for B (e.g. apartments and short/market). |
| 5 | `DEFAULT` | Attackers spread for map control and no site commit of 3+ players happens before 45 s, whatever happens later. |

Tie-breaks to agree:

- A fast hit through two routes: split (the route count decides, not the speed).
- Fakes: label the site of the **final** commit, not the fake.
- Rounds that end before any commit (all attackers die early, saves): `DEFAULT`.
- Eco and force-buy rounds: label the movement as normal; do not add a class.

## Defender response (only rounds with a plant)

| Key | Label | Rule (proposed) |
|---|---|---|
| 6 | `RETAKE` | At least 3 defenders enter the planted site within about 5 s of each other. |
| 7 | `TRICKLE` | Defenders contest the site but arrive more than about 5 s apart, or never 3 together. |
| 8 | `SAVE` | Surviving defenders do not contest the site and move away or hold elsewhere. |

Tie-breaks to agree:

- Only one or two defenders alive at the plant: `RETAKE` if they enter together,
  `TRICKLE` if separately, `SAVE` if they stay away.
- A retake abandoned half-way: label what the defenders did first, and note the round.

## Round boundaries and plant

- Start: end of freeze time (players start moving). End: the moment the round is
  decided (last kill, defuse or explosion), not the start of the next freeze time.
- Plant: the first frame the C4 badge stays still on a site. The current detector lists
  planted bombs as dropped, so check by eye.

## Uncertain rounds

The labeller has no notes field. Keep a list of `match, round, reason` for any round
that does not fit these rules; review the list together before changing a rule.
