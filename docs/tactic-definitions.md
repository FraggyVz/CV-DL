# How to Label Rounds

> **Status: draft rules, not yet agreed by the team.** The numbers below (30 s, 45 s,
> 5 s, …) are starting points. Try them on a few rounds, note anything that does not
> fit, and we will agree the final rules together before labelling lots of rounds.

This guide explains how to watch a match and record, for every round, **what the
attackers did** and **how the defenders reacted**. Everyone must follow the same rules,
otherwise the labels from different people cannot be mixed.

**Test the rules on Furia–Falcons and Spirit–Falcons only.** NaVi–Falcons is our held-out
test match: label it with the final agreed rules, but never use it to try out or change
the rules. Otherwise our test results would not be fair.

---

## 1. Before you start

**Open the labelling window** (from the project folder, with the virtual environment active):

```powershell
python -m stratscout label-rounds --match furia
```

- The terminal prints a list of rounds with a **"play from"** time for each one. That is
  roughly when players start moving. Keep it visible.
- **Click once on the window** so it receives your key presses. The mouse does nothing
  else; everything is done with the keyboard.
- Your labels are saved to `labels/<match name>.rounds.json`. If you close and reopen
  the tool, it continues where you stopped.

**On the minimap:**

- **Yellow** numbered dots = attackers (T). They plant the bomb.
- **Cyan / blue** numbered dots = defenders (CT). They protect the bomb sites.
- The **orange rectangle** = the bomb (C4). While someone carries it, it is drawn **on
  top of that attacker** and hides them: count it as one attacker.
- A **✕** in a team colour = a player who just died (it shows for about 2 seconds).
  It is not a living player.
- **B** = top-left, **A** = bottom-middle. The radar is rotated compared with the usual
  Mirage picture; the attackers start on the right.

**Only count players you can see.** If dots overlap or one is hidden, do not guess
where an unseen player is.

**Round 1:** the terminal may mark it "started before recording", but that is only an
estimate. Label it if the players are still in their spawns at the start; skip it only
if it really started in the middle.

---

## 2. The three kinds of keys

This is the most important thing to understand.

| Kind | Keys | What it does | When to press it |
|---|---|---|---|
| **Moving** | `→` / `←` (2 s), `Shift`+`→` / `Shift`+`←` (20 s) | Moves through the video | Any time |
| **Time markers** | `[` start, `]` end, `a` / `b` bomb planted | **Records the time shown on screen right now** | **Only at the exact moment** it happens |
| **Choices** | `1`–`5` attackers, `6`–`8` defenders, `t` / `c` winner | Records your decision | Any time before `Enter`. Press another key to change your mind |

Then:

| Key | What it does |
|---|---|
| `Enter` | Finishes this round and adds it to the list (the video stays where it is) |
| `s` | Saves the file. **Do this after every round or two** |
| `q` | Saves and closes. **Always close with `q`, never with the window's ✕**, or unsaved rounds are lost |
| `u` | Removes the last finished round (if you made a mistake) |
| `x` | Removes a wrong bomb-plant marker |

The top of the window always shows the current time (`t=…s`), the estimated game round,
and everything you have marked so far for this round. Check it before pressing `Enter`.

---

## 3. Labelling one round, step by step

1. **Go to the start.** Use `Shift`+`→` to jump near the "play from" time, then `→` until
   the players start moving. Press **`[`**.
   Note this time. All the rules below count seconds **from here**.

2. **Watch the attackers (yellow)** with `→`. Decide what they did using
   [section 4](#4-what-did-the-attackers-do-keys-1-5) and press **`1`–`5`**.

3. **Was the bomb planted?**
   - **Yes:** move to the first moment the bomb stops on a site and press **`a`** or **`b`**.
     Careful: a **dropped** bomb looks the same as a planted one. If the carrier just
     died there (a ✕ next to it), it was dropped, not planted. When unsure, check the
     round on HLTV (a planted round ends with a defuse, an explosion, or a "bomb planted"
     event). The plant time matters: it splits the round into the "before plant" part
     (used for the attackers' tactic) and the "after plant" part (used for the defenders).
     Then watch the defenders and press **`6`–`8`**
     ([section 5](#5-how-did-the-defenders-react-keys-6-8)).
   - **No:** skip this step.

4. **Go to the end.** Move to the moment the round is decided (the last kill, the defuse,
   or the bomb exploding) and press **`]`**.
   Do *not* use the next round's start; the round ends when it is decided.

5. **Who won?** Press **`t`** (attackers) or **`c`** (defenders). If you are not sure, check
   the match's round history on HLTV.

6. **Check the title, then press `Enter`.** The terminal confirms the round, for example:
   `label 1 (est. game round 2): A_SPLIT plant A defenders RETAKE winner CT`

7. Press **`s`** to save. Move on to the next round.

---

## 4. What did the attackers do? (keys 1–5)

Answer these questions in order. Count time from the start you marked with `[`.

**Question 1: Did 3 or more attackers go onto a bomb site before 45 seconds?**

- **No** → press **`5` DEFAULT** (slow round, spreading out for map control).
  This is also the answer if the attackers all die or give up before going onto a site.
- **Yes** → go to question 2.

**Question 2: Which site did they finally go onto, and by how many routes?**

| They went onto… | How | Key |
|---|---|---|
| **A** | **One route**, at least 4 players, within 30 s | **`1` A_RUSH** |
| **B** | **One route**, at least 4 players, within 30 s | **`2` B_RUSH** |
| **A** | **Two or more routes** at once (each with at least 2 players, arriving within about 10 s of each other), e.g. ramp + palace, or ramp + connector | **`3` A_SPLIT** |
| **B** | **Two or more routes** at once (same rule), e.g. apartments + short | **`4` B_SPLIT** |

**Tricky cases**

- **Fast, but from two routes?** → it is a **split**. The number of routes decides, not the speed.
- **They faked one site, then went to the other?** → label the site they **finally** went onto.
- **Eco or force-buy round?** → label what they did as normal.
- **"Went onto a site"** means actually entering the site area, not just walking towards it.
- **Fits none of these?** For example, one route but slower than 30 s, or only 3 players
  on one route. The draft rules don't cover this yet. Pick the closest key and write
  the round in your notes ([section 8](#8-rounds-that-dont-fit-the-rules)) so the team can decide.

---

## 5. How did the defenders react? (keys 6–8)

**Only for rounds where the bomb was planted.** Watch the defenders (blue) after the plant.

| What the defenders did | Key |
|---|---|
| **3 or more arrive on the site together** (within about 5 s of each other) | **`6` RETAKE** |
| **They come back, but one by one** (more than about 5 s apart), or never 3 together | **`7` TRICKLE** |
| **They don't go back to the site**; they stay away or leave | **`8` SAVE** |

**Tricky cases**

- **Only 1 or 2 defenders alive?** → `6` if they go in together, `7` if separately, `8` if they stay away.
- **They started a retake, then gave up?** → label what they did **first**, and write the round in your notes.

---

## 6. Examples

| What happened | Keys, in order |
|---|---|
| 4 attackers rush A through ramp at 20 s, plant, 3 defenders come back together, defenders win | `[` … `1` … `a` (at the plant) … `6` … `]` … `c` → `Enter` → `s` |
| Attackers hit B from apartments and short at 50 s, plant, defenders come back one by one, attackers win | `[` … `4` … `b` … `7` … `]` … `t` → `Enter` → `s` |
| Slow round, nobody goes onto a site, attackers all die at 70 s, no plant | `[` … `5` … `]` … `c` → `Enter` → `s` |
| Fake at A, then everyone goes to B through one route, plant, defenders stay away | `[` … `2` … `b` … `8` … `]` … `t` → `Enter` → `s` |

(`…` means "move through the video with the arrow keys".)

---

## 7. If something goes wrong

The tool will **not** accept the round when you press `Enter` if:

| Message / problem | Fix |
|---|---|
| Start, end or attacker tactic missing | Press `[`, `]` or `1`–`5` |
| End is before start | Move to the right time and press `]` again |
| The plant is not between start and end | Press `x`, move to the plant, press `a` / `b` again |
| Defender reaction chosen but no plant | If the bomb *was* planted, mark it with `a` / `b`. If it wasn't, press `x` to clear the reaction |

Other mistakes:

- **Wrong choice key?** Just press the right one; it replaces the old one.
- **Already pressed `Enter` on a wrong round?** Press `u` to remove it, then label it again.

---

## 8. Rounds that don't fit the rules

Don't spend long on one round. Pick the closest label, press `Enter`, and write one
line in a notes file, for example:

```
furia, game round 7, looked like a split but only 1 player on the second route
```

We will go through these notes together. They are how we improve the rules.

---

## Quick reference

```
MOVE        →/← 2 s     Shift+→/← 20 s
MARK TIME   [ start     ] end       a / b  bomb planted at A / B
ATTACKERS   1 A rush    2 B rush    3 A split    4 B split    5 Default
DEFENDERS   6 Retake    7 Trickle   8 Save        (only if planted)
WINNER      t attackers c defenders
FINISH      Enter       s save      q save & quit     u undo     x clear plant
```
