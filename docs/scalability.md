# Scalability

How StratScout can grow beyond the submitted Mirage classifier, what already supports
that, and what each direction would take. Current limits are in
[findings and limitations](challenges-and-findings.md#limitations); the near-term plan
is in the README's [next steps](../README.md#next-steps).

## What is built to grow

- **More data needs no code changes.** New matches go through the same steps:
  `process-recordings`, `label-segments`, `export-segments` (split by whole match),
  `build-clip-data` (about 5 s per clip) and `train-tactics`. Each dataset export is
  dated, and the submitted one is released with per-clip checksums.
- **Map knowledge sits in one file.** The map regions are a versioned JSON file
  (`stratscout/mirage_regions.json`). The models read detector counts per named region,
  and prediction refuses a regions file whose version differs from the one the model was
  trained with.
- **The detector needs no training data.** It is colour and shape rules for the
  spectator radar's icons, so new matches do not need icon labels.
- **One analysis path.** The local and cloud apps call the same `web_analysis` code, so
  a change to the pipeline or the models reaches both.
- **Packaged deployment.** The local app is one Docker image. The model checkpoints are
  1.7 MB, and the models take about 0.1 s per clip
  ([performance](web-app.md#how-the-deployed-models-are-kept-fast)).

## New maps, tactics and game mechanics

The pipeline separates seeing the radar (the detector) from understanding it (map
regions, clip data and the strategy models). That split is what lets the project widen
its scope.

### More maps

The player and bomb icons look the same on every map's spectator radar, so the
detector's colour rules should carry over, but they have only been checked on Mirage.
Each new map needs:
- its radar image, a detector check on that map, and a background check for the new
  layout;
- a regions file naming its routes and sites, like `mirage_regions.json`;
- class rules for its routes in the [tactic guide](tactic-definitions.md), which says
  each map needs its own region and route definitions;
- recorded and labelled matches, then training.

A model per map is the simplest start. A shared model across maps would need region
names that mean the same thing on every map (for example "A main", "B main", "mid
route to A"), and enough matches on each map.

### More tactics

The five classes cover direct and split attacks on each site, and retakes. The labels
already show where more classes are needed:
- **Mid-to-A attacks.** Furia #16 and Spirit #17 fit none of the five classes and are
  left pending in the [review notes](../labels/review-notes.md).
- **Changes of plan and fakes.** Furia round 14 is an abandoned B push followed by a
  push to A. A fake on one site before hitting the other is a common pattern of the
  same kind.
- **Tempo.** The tactic guide lists rush tempo and default openings as possible future
  attributes, alongside the class rather than as new classes.
- **CT setups.** The current classes describe the T side. The same region counts for CT
  players (`ct_side` is already a region) could describe defensive setups and
  rotations.

Each new class needs its own rule in the tactic guide, enough reviewed clips (the
smallest class today, B_SPLIT, has only nine training clips), and retraining. The tactic
guide warns against adding a catch-all "background" class without a separately
evaluated design.

### Other game mechanics

The radar shows more than tactics, and some of it is already extracted:
- **The bomb.** The detector already tracks the C4 badge: carried, dropped and an
  estimated plant with its site and time. Plant timing, drops and post-plant positions
  could be analysed on their own. The plant estimate would first need checking against
  the demo's bomb events.
- **Deaths and trades.** The detector finds death markers and currently drops them.
  Kept and timed, they would show where and when each side loses players.
- **Rotations.** Counts per region over time already show how fast CTs move between
  sites after a T commitment.
- **Round outcomes.** Round winners are kept as evaluation evidence only. Relating a
  predicted tactic to whether it won would turn the classifier into analysis.

Each of these is a new label or target. It needs checking against demo data, as the
[detector check](detector-check.md) did for player positions, before results are
reported.

## What scaling further would take

| Direction | What it needs | Status |
|---|---|---|
| More teams and matches on Mirage | Record, label, export and retrain. Prioritise split attacks: there are only nine B_SPLIT training clips. | Pipeline ready; no new data yet |
| Another map | A regions file, map image, route rules, labels and training, as above | Not started |
| More tactic classes | Class rules, reviewed clips per class, retraining | Candidate classes identified in the labels |
| Other game mechanics | A new target per mechanic, checked against demo data | Bomb states and death markers already detected |
| Full matches instead of clips | Round starts are already found from video alone. Finding each tactic's start and end within a round is not implemented. | Partly done |
| Other recording setups | A crop profile per machine. A 1920×1080 recording gave a 382 × 382 radar instead of the 494 × 500 the models use, so resizing would need validating. | Only the 1440p setup is checked |
| More users at once | The local app stores files on one machine's disk. The cloud app keeps one match per browser session and analyses on a shared CPU, about 14 s per clip. Many users would need a job queue, shared storage and possibly a GPU for ResNet-18. | Not needed for the demo |
