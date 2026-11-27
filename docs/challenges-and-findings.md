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
image features alone to 0.796 on the combined detector and image features. The TCN
using the combined features achieved the highest score overall
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

Both models were fixed before the test matches were scored. The TCN was selected by
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
  Spirit–Falcons held out. The GRU has about 280 weights per training clip, so it can
  memorise a fold.
- **Keeping the test set clean.** The test matches were scored once, after the settings
  were fixed. Changing settings after seeing a test score would turn the test matches
  into validation data.

### Deployment

- **Free hosting.** Hugging Face Docker and Gradio Spaces needed a paid plan when this
  was checked in November 2026. The cloud demo runs on Streamlit Community Cloud instead,
  on CPU.
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
