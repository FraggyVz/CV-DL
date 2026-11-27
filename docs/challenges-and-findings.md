# Findings, challenges, edge cases and limitations

This page collects what the project found, what went wrong on the way, where the system
breaks, and what it cannot do yet. Each point links to the run or guide that holds its
evidence. The scores and dataset are described in the
[README](../README.md#reproduce-the-submitted-results); this page does not repeat them in
full.

Tactic scores use macro-F1 unless another metric is named. The 2026-11-16 dataset
provided 82 training clips from four matches and 45 test clips from two matches.
"Nested" means leave-one-match-out cross-validation with the epoch chosen from the
training matches only.

## Findings

### The combined radar and image features performed best

With the same mean-pooling approach, nested macro-F1 rose from 0.518 on ResNet-18
image features alone to 0.796 on the combined detector and image features. The Compact
pooled GRU gained the same way, from 0.600 to 0.792. The TCN using the combined features
achieved the highest score overall
([model comparison](results.md#macro-f1-for-every-model)).

| Input | Best nested macro-F1 | Model |
|---|---:|---|
| ResNet images | 0.620 ± 0.015 | Temporal average pooling (3 segments) |
| Detector positions on an 8×8 grid | 0.595 | Mean pooling |
| Clip data (72 detector channels in total + 32 ResNet PCA components) | **0.849 ± 0.021** | TCN |

ResNet-18 was trained on everyday photos, and resizing to 224 × 224 shrinks the radar
icons from about 18 px to about 8 px. Its features describe the whole map's appearance,
not the players ([clip-data limits](clip-data.md#limits)). Naming the map areas that the
class rules use ("ramp and/or palace plus mid → connector") gave the models the
information the labels are defined by.

### Frame order helps, but only modestly

- On ResNet images, temporal average pooling scored 0.620. The random-segment control
  takes the same three segments with the frames shuffled first, and it scored 0.526
  ([details](tactic-temporal.md#the-random-segment-control)).
- On clip data, the TCN scored 0.849 and its pooled control 0.809. The pooled control
  reads the same rows averaged over the clip, so step order cannot matter.

The TCN's gain over its own order-blind control is 0.04, about two seed SDs. Seeds rerun
the same four matches, so this is a rough guide, not a significance test
([temporal limits](tactic-temporal.md#limits)).

### TCN and GRU on the held-out test set

The test matches were never used for training or for choosing a model (see the
[test notebook](../notebooks/tactic-final-test.ipynb) for details). The TCN was selected by
nested cross-validation before the GRU was scored.

| Model | Nested CV (mean ± seed SD) | Test, mean over seeds | Test, seed ensemble | Test accuracy (ensemble) |
|---|---:|---:|---:|---:|
| TCN | 0.849 ± 0.021 | 0.897 ± 0.009 | 0.927 | 0.933 |
| Compact pooled GRU | 0.792 ± 0.033 | 0.890 ± 0.030 | 0.963 | 0.956 |

The GRU's ensemble scored higher on the test set, but its cross-validation score was
lower and its seed spread wider. With 5–6 clips in some test classes, one clip moves a
class's F1 a lot. The TCN stays the selected model; the GRU is the second opinion
([test notebook](../notebooks/tactic-final-test.ipynb),
[GRU test report](../runs/tactic-gru-clipdata-test-20261121/report.md)).

### Playback speed did not change the reported confidence

The supplied 1× and 2× comparison reports identical TCN and GRU confidence for all
24 clips. In this run, playback speed made no difference to the reported confidence.
This does not establish that speed cannot affect predictions in other clips or setups.

Across the 24 clips, the GRU has a slightly higher mean confidence than the TCN
(90.2% against 87.8%) and a higher median (97% against 91%). The GRU reaches at least
90% confidence on 17 clips, compared with 14 for the TCN. The TCN's scores range from
61% to 97%; the GRU ranges from 46% to 100%. The GRU is more confident on average in
this sample, but this alone does not show that it is more reliable.

| Class | Clips | TCN mean confidence (range) | GRU mean confidence (range) |
|---|---:|---:|---:|
| A_DIRECT | 10 | 89.4% (61–97%) | 90.8% (74–99%) |
| A_SPLIT | 1 | 86% | 73% |
| B_DIRECT | 3 | 87.0% (78–92%) | 88.0% (65–100%) |
| B_SPLIT | 3 | 90.3% (87–95%) | 77.7% (46–98%) |
| RETAKE | 7 | 84.9% (61–92%) | 95.3% (83–100%) |

The models' confidence differs by 9.5 percentage points on average (absolute gap),
with gaps of at least 20 points on three clips. GRU confidence varies most on B_SPLIT:
it is 46% on clip 009 while the TCN reports 89%. The models also differ on RETAKE clip
019 (61% TCN, 92% GRU) and A_DIRECT clip 007 (61%, 90%). These are useful clips to
review. Only clip 009 falls below the app's 0.6 review threshold, with the GRU at 46%;
no TCN confidence is below it.
Confidence describes the model's certainty, not whether its prediction is correct.
The comparison does not record whether each prediction matches a reviewed label, so it
cannot establish accuracy or whether confidence is calibrated. One A_SPLIT clip is too
few to support a class trend; the class averages are descriptive only (manual 1×/2×
confidence comparison, `testing results.txt`).

### Where the TCN is wrong

In nested cross-validation, the TCN ensemble misclassified 10 of 82 clips. Four
errors confused direct and split attacks on the same site. Two A_DIRECT clips were
called A_SPLIT, one A_SPLIT was called A_DIRECT, and one B_DIRECT was called
B_SPLIT. The ensemble also classified three retakes as attacks
([nested confusion matrix](../runs/tactic-tcn-clipdata-nested-20261120/summary.json)).

On the test set the ensemble missed 3 of 45 clips: one A_DIRECT called B_SPLIT, one
B_DIRECT called B_SPLIT, and one RETAKE called A_DIRECT
([TCN report](../runs/tactic-tcn-clipdata-final/report.md)).

### Training choices that mattered

- **The GRU's readout.** A GRU that classifies from its last hidden state scored 0.345
  on ResNet images. Pooling the mean and maximum over all steps (the Compact pooled GRU)
  scored 0.600 ([details](tactic-temporal.md#the-gru)).
- **Class weights.** Without class weights, temporal average pooling fell from 0.620 to
  0.564, and A_SPLIT's F1 fell by 0.119
  ([details](tactic-temporal.md#class-weights)).
- **Epoch selection.** Scoring each match at the epoch that was best on the other folds,
  whose models were trained on it, gave most models higher scores than nested selection.
  Nested selection is reported as the main score.

### Overall player detection

Across the scored rounds of the four training matches, the detector found 95.5% of T
and 98.3% of CT player positions within 12 radar pixels of their demo positions.
Median error was under one pixel for both teams
([detector check](evidence/detector-check-2026-11-27/report.md)). The
[edge-case table](#detection) breaks down the results for isolated players, overlapping
icons and spawn stacks.

### Why the detector also finds the bomb

Yongheng added C4 badge detection to Gin's colour baseline for three reasons
([bomb-detection experiments](archive/bomb-detection.md)):

- **The badge looked like a T player.** The orange badge (OpenCV hue ≈ 16) lies within
  the colour baseline's hue tolerance of T yellow (≈ 27) and is player-sized (about
  16 × 11 px), so the baseline would count it as a T. A shape rule (aspect, box fill and
  corner fill) separates the rectangular badge from round icons, and player detection
  skips it.
- **The badge hides its carrier.** It is drawn on top of the carrier's icon, so one T
  disappears whenever the bomb is carried. The detector infers that carrier at the
  badge. In the four-match check, bomb carriers were found 97.4% of the time
  ([detector check](evidence/detector-check-2026-11-27/report.md)). In the 2026-11-11
  run, inferred carriers were 21% of T detections; without them, T found fell to about 76%
  ([detector limits](detector-check.md#limits)).
- **The bomb marks the post-plant phase.** Clip data gives the models the badge
  position, its state (carried, alone, dropped or missing), an estimated plant and the
  badge's distance to each site
  ([detector channels](clip-data.md#detector-channels-72-per-row)). An estimated plant
  appears in 75% of training RETAKE clips, against 0–20% of attack clips
  ([data checks](data-processing.md)).

In the bomb-badge ablation on 8×8 position features, planned before the scores were seen,
removing the badge lowered macro-F1 by only 0.004–0.025. RETAKE F1 fell by about 0.1 for
every model ([position results](tactic-temporal.md#results-with-position-features-2026-11-16)).
The badge matters for the post-plant class, not for telling the attacks apart.

### Overlapping icons are counted by disc fitting

Icons of one team that overlap merge into a single blob, which the colour baseline
counted as one player. Yongheng's splitter (`icon_split.find_icons_split`) gives each team
its own colour mask, so a T touching a CT stays separate. For each blob it places k equal
discs (k = 1–5, radius from the median icon area of 246 px²) at k-means centres and keeps
the k whose union best matches the blob (IoU). Each team is capped at five, or four while
a carrier is hidden under the badge ([overlapping icons](archive/icon-split.md)).

Label-free checks on NaVi–Falcons, with thresholds fitted on Furia–Falcons and
Spirit–Falcons:

| | Old baseline | Area rule + cap | Disc fit + markers + cap |
|---|---:|---:|---:|
| All-alive frames showing five per team | 8.9% | 96.6% | 53.4% |
| Whole-match frames with a team above five | 0.03% | 0% | 0% |

The area rule scored higher on all-alive frames, which are mostly tight spawn stacks, but
it over-counted small groups and death markers later in rounds. On a reviewed late-round
frame showing 1 T and 3 CT, it reported 5 T and 5 CT. The disc fit matched every reviewed
live-play frame, so the live preview and clip data use it. In the four-match demo check,
overlapping icons were found 86.0% (T) and 93.8% (CT) of the time.

### Round starts from video alone

`rounds.py` estimates round starts from freeze time, when nearly all T colour sits in T
spawn and all CT colour in CT spawn. Each recorded match gave 20–21 rounds, with the
halftime pause before round 13. The [detector check](detector-check.md#what-it-does) pairs
these video rounds with demo rounds, and the counts must agree. No calibration flags were
raised on the four matches.

## Challenges

These are problems the team hit, and what was done about each.

### Recording and storage

- **Full matches took too long to record.** Gin recorded each replay once in OBS at
  2560×1440 and 60 fps with playback fixed at 2×. For Aurora–Vitality, 24 minutes
  of recording covered about 48 minutes of gameplay
  ([recording setup](recording-setup.md#current-capture-setup)).
- **Full-screen recordings were too large to share.** Gin’s `process-recordings` script
  crops the minimap, restores 1× game time and exports aligned 5, 10 and 30 fps videos
  in one pass. For the same Aurora–Vitality footage, the files were 22.7, 29.2 and
  40.9 MB respectively. The 10 fps version used for labelling and modelling was 29%
  smaller than the 30 fps version ([sampling implementation](../stratscout/video_preparation.py)).

### Player and bomb detection

- **The badge merged with its carrier.** Shape alone found the badge in only 26% of
  Furia–Falcons frames. 58% of the misses were a badge merged with an adjacent icon,
  usually its carrier. The detector now keeps badge-orange pixels first (hue 10–21), then
  applies a 3×3 opening to strip thin fringes, and only then checks the shape:

  | 5 fps scan | Shape only | Colour then shape | + opening |
  |---|---:|---:|---:|
  | Furia–Falcons frames with a badge | 26% | 86% | 94% |
  | Spirit–Falcons frames with a badge | 32% | 89% | 96% |

  All 48 detections checked by eye were on the badge. This is not labelled recall.
- **Carried and dropped badges look identical.** A first rule called the bomb carried
  when T yellow was within 40 px. It misread a still carrier hidden under the badge as
  dropped, and a teammate beside a dropped bomb as carried. The current rule reads the
  badge's rim: four or more T-yellow or white arrow pixels in a 4 px band mean the
  carrier is underneath. On Furia–Falcons, 99.7% of moving badges (certainly carried)
  were judged carried. A still badge with no carrier for 2 s becomes a dropped candidate.
- **Look-alikes at the rim.** The yellow A/B site letters, where a planted bomb often
  sits, and the white arrow of a CT beside or defusing the bomb read as a carrier. Rim
  yellow that matches the empty-map background, and arrows attached to cyan icons, are
  now ignored.
- **Death markers looked like players.** A ✕ marker appears in the team colour for about
  2 s. Every living icon has a white direction arrow and a ✕ does not, so blobs touching
  fewer than 8 white pixels are dropped. On the training matches, single living icons
  touched at least 21–32 white pixels (1st–5th percentile).
- **No labels for player counts.** No frames had hand-labelled players, so detector
  versions were first compared with label-free checks: all-alive frames (freeze time and
  the first 8 s after it) should show five per team, and a team never has more than five.
  The demo-based [detector check](detector-check.md) later supplied true positions.

### Modelling

- **The first `short` region was in the wrong place.** The radar is rotated compared
  with the usual Mirage overview, and the first map-region draft put `short` on the CT
  side of B. B_SPLIT clips reached it no more often than B_DIRECT (22% against 17%).
  Region v1 moved `short` onto the route from mid, found on training clips only. With
  it, B_SPLIT reaches short in 89% of clips against 8% for B_DIRECT
  ([map regions](clip-data.md#map-regions)). This was a design choice informed by
  training labels, so the clip-data scores may be optimistic.
- **Little data.** There are 82 training clips and only nine B_SPLIT clips. Fold scores
  vary widely: the TCN scored 0.955 with Aurora–Vitality held out and 0.700 with
  Spirit–Falcons held out. The first (Last-state) GRU had about 280 weights per training
  clip and memorised its folds; the Compact pooled GRU has about 29 (2,349 weights).
- **The GRU was weak on ResNet images.** Even with the pooled readout
  ([training choices](#training-choices-that-mattered)), the Compact pooled GRU reached
  only 0.600 on ResNet-18 features, which describe the whole map rather than the players.
  It was retrained on the shared clip features, whose region counts give the routes that
  the class rules are defined by, and reached 0.792, close to the pooled classifier
  (0.809). The gain also includes the change in sampling (16 frames per clip against 5
  rows per second), so it cannot be credited to the features alone
  ([model comparison](../notebooks/tactic-model-comparison.ipynb)).
- **The first epoch rule was optimistic.** The TCN's original rule scored each held-out
  match at the epoch that was best on the other folds, whose models had been trained on
  that match. Nested selection chooses the epoch from the training matches only. Rerun
  that way, the old rule had scored most image models higher, by up to 0.081, but made
  little difference on clip data (−0.015 to +0.017). Nested selection is now the main
  score. It is slower: the nested TCN run took about an hour, and the detector-features
  TCN has no nested result yet
  ([nested against other-folds](../notebooks/tactic-model-comparison.ipynb)).
- **Keeping the test set clean.** The test matches were never used for training or for
  choosing a model (see the [test notebook](../notebooks/tactic-final-test.ipynb) for
  details). Changing settings after seeing a test score would turn the test matches
  into validation data.

### Deployment

- **Finding a free public host for the Docker app.** The local website is a React +
  FastAPI app run with Docker Compose. The team wanted a cloud deployment that anyone
  could open, without requiring each visitor to run Docker or sign in. The Docker-based
  hosting options investigated (Hugging Face Docker and Gradio Spaces) required a paid
  plan for the needed deployment when checked in November 2026, so they did not meet the
  free-hosting requirement. The team chose Streamlit Community Cloud for the public
  StratScout demo instead. Its app runs on CPU and uses the same analysis code as the
  local website, with a Streamlit page around the results view
  ([deployment guide](../deploy/streamlit/README.md)).
- **Reusing the local app in the cloud.** The cloud page calls the same
  `stratscout.web_analysis` code as the local app. It embeds the local app's React
  results view as a Streamlit component. Streamlit Community Cloud has no Node, so the
  built component (`deploy/streamlit/results_component/`) is committed
  ([Streamlit deployment](../deploy/streamlit/README.md)).
- **Missing ffmpeg in the cloud.** Streamlit Community Cloud only reads `packages.txt`
  from the repository root. While it sat in the app folder, analysis failed with
  "Missing ffprobe".
- **Multi-file uploads failed.** Uploading several clips at once gave "AxiosError:
  Network Error" because the page reset the uploader while files were still uploading.
  The page now keeps the uploader and skips files it has already added.
- **Clips that browsers cannot play.** Some source clips are MPEG-4 Part 2, which
  browsers do not decode. Both apps make a separate H.264 playback copy and keep the
  original for analysis.

## Edge cases

### Detection

| Situation | T found | CT found |
|---|---:|---:|
| Isolated players | 98.7% | 99.4% |
| Overlapping icons | 86.0% | 93.8% |
| Spawn stack | 88.7% | 90.2% |

- **Hidden bomb carriers** are inferred from the C4 badge and counted as T detections.
  In the 2026-11-11 run they were 21% of T detections. Counting observed icons only, T
  found falls to about 76% ([detector limits](detector-check.md#limits)).
- **Missing players.** In the same run, at least one player was missing in 12% of
  team-frames. Models that read a sequence of frames must tolerate gaps.
- **Single rounds vary.** Furia round 2 had 80.5% of T players found, against a pooled
  95.5%.
- **Connector picks up some direct A attacks.** 60% of A_DIRECT clips have a T in
  connector at some point, against 100% of A_SPLIT clips. Occupying connector alone
  cannot distinguish the two ([data checks](data-processing.md)).
- **Planted or dropped.** The badge alone cannot tell them apart, so a bomb dropped on a
  site counts as planted. A planted badge flickers in and out of detection and resets the
  2 s stillness timer, so the bomb-state tracker reached "dropped" in only 20% of training
  RETAKE clips. The plant estimate counts badge frames in a 3 s window instead
  ([clip-data notes](clip-data.md#detector-channels-72-per-row)).
- **Drops by death are reported late.** A yellow ✕ beside the badge reads as a carrier
  for the ~2 s it is visible, so the drop appears about 2 s after the death.
- **Spawn stacks of five** fit best as four discs, so they are under-counted. Tracking
  players seen separately just before they stack, or a trained detector, could recover
  them.
- **A ✕ merged with living icons** keeps their arrows, so it is not removed until it fades.

### Labels and tactics

These rounds are open in the [review notes](../labels/review-notes.md):

- **Mid-to-A attacks** (Furia #16, Spirit #17) do not fit any of the five classes. They
  are left pending and are not in the dataset.
- **Changes of plan.** Furia round 14 is an abandoned B push followed by a mid-only push
  to A. It is pending.
- **Uncertain boundaries.** Spirit #5 is reviewed as A_SPLIT, but its boundary is marked
  uncertain. It is excluded from modelling.
- **Disputed labels.** NaVi #25 is labelled RETAKE, but Gin suggested leaving round 16
  unlabelled. It still needs a decision.

### Input to the apps

- **One clip is one tactic.** The apps analyse each uploaded clip from first frame to
  last as one tactic. A full-match recording gives one prediction for the whole match.
- **Clip length.** Clips are padded and masked, never stretched. The TCN's blocks see
  about 25 s, and pooling covers the whole clip.
- **Low confidence.** Predictions below 0.6 are flagged for review, with the runner-up
  class.
- **Empty-map background.** The detector subtracts a per-match background built from
  that match's clips. Give at least about 5 clips per match, or a saved background
  (`save-background`, `--background-from`).
- **Full-screen recordings at other resolutions.** The crop profile in
  `configs/recording-video.json` is per machine. On one 1920×1080 full-match recording,
  `find_radar_box` found a 382 × 382 radar box, not the 494 × 500 crop the models were
  trained on, so that recording was not analysed. The capture standard is 1440p at 60
  fps ([recording guide](recording-setup.md)).
- **Replay speed.** Preparation assumes a constant 2× replay. Pauses, stalls or speed
  changes break that assumption, and a nominal frame rate does not prove that every
  frame is unique.

## Limitations

### Scope

- Mirage and the spectator radar layout only, on these recordings.
- A person picks each tactic's start and end. Full-match tactic segmentation is not
  implemented.
- The cloud app accepts prepared radar clips only (494 × 500 at 10 fps), and keeps one
  match per browser session.

### Evidence

- The test score is optimistic: the test teams also appear in training, and some classes
  have only 5–6 test clips. Report cross-validation as the estimate for new matches.
- The map regions (`mirage-spectator-v1`) are a draft and were adjusted after looking
  at training labels.
- The detector check covers the four training matches only. It does not score the test
  matches, the CNN detector, freeze time or post-round frames.
- The pipeline graphs in the app show the recorded inference steps. They are not a
  feature-attribution explanation of why a model chose a tactic.
- Bomb states, dropped candidates and plant estimates have not been checked against the
  demo's `bomb_dropped`, `bomb_pickup` and `bomb_planted` events. Badge detections and
  icon counts were reviewed by eye, not against hand labels.
- The disc fit was chosen over the area rule using label-free counts and a visual review
  of NaVi–Falcons video, which is now a test match. No tactic labels were involved, but
  that match's video informed the detector choice.

### Components

- **Tracking.** There is no reliable player identity over time. The tracker's fixed slots
  can invent or keep stale positions, and its gates and miss limits are per step rather
  than per second. The current models avoid it by reading per-region counts.
- **Detection.** Stacked or hidden icons are missed, and spawn stacks are under-counted.
  Planted-bomb separation and round boundaries are unverified estimates.
- **No time since round start.** The manifest gives each clip's start in the recording,
  but the recordings are not on every machine.
- **Spatial ResNet maps are not stored.** The 7 × 7 × 512 maps would take about 1 GB for
  this dataset, so only the pooled 512 numbers per row are kept.
- **Label safety.** Legacy proposal and review writes can overwrite labels, and empty
  icon frames are dropped. Explicit review status, uncertain centres and safe migration
  are not implemented. Never target reviewed labels with automatic proposals.
- **Legacy data formats.** The legacy ingest and round schemas differ from the segment
  format. World-to-radar mapping and demo/video alignment need calibration. Any split of
  real icon data must keep whole matches apart. Removed
  classifier results and historical synthetic scores are not current evidence.

## Next steps

The README's [next steps](../README.md#next-steps) list what follows from this page: new
teams and matches (especially split attacks), a team review of the map regions, decisions
on the open labels, and work beyond prepared Mirage clips.
