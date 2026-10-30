# Overlapping Player Icons

> Archived reference (30 October 2026). Results, commands and status describe the recorded revision, not necessarily the current checkout. Use the [README](../../README.md) for current status and next steps, the [recording guide](../recording-setup.md) for capture, and the [tactic guide](../tactic-definitions.md) for labels.

## Problem

The colour baseline (`segmentation.find_icons_teams`) treats each connected blob of
saturated colour as one player. Overlapping icons of one team merge into a single blob,
so a player standing behind a teammate disappears. Blobs above `icon_area` are dropped
entirely, which removes whole spawn clusters during freeze time. One mask is shared by
both teams, so a T touching a CT also merges and is assigned one team by median hue.

## Method (current: discs + markers + cap)

`icon_split.find_icons_split`:

1. **One colour mask per team** (hue ± 6, saturation and value ≥ 90, minus the empty-map
   median background); badge-shaped blobs are dropped.
2. **Death markers removed.** Every living icon has a white direction arrow attached; a ✕
   death marker does not. Blobs touching fewer than 8 white pixels (4 px band) are
   markers. On the training matches, single icons touched ≥ 21–32 white pixels (1st–5th
   percentile).
3. **Disc fitting.** For k = 1–5, k equal discs (radius from the median icon area, 246 px)
   are placed at k-means centres; the k whose union best matches the hole-filled blob
   (IoU) is the icon count. Unlike area alone, this separates two touching icons from
   three overlapping ones.
4. **Five per team.** If a team exceeds five (four with a hidden bomb carrier), the blob
   whose IoU drops least by losing an icon gives one up; then the worst-fitting single
   blob is dropped.

Earlier versions are kept for comparison: the area rule (`method="area"`: icons from blob
area, grid-searched `split_ratio`/`extra_area`) with and without the cap.

## Evaluation

```powershell
python -m stratscout icon-count-eval --train furia spirit --test navi --output runs/icon-count-eval
```

**All-alive check (label-free).** During freeze time and the first 8 s after it nobody has
died, so a correct detector shows 5 T and 5 CT. **Impossible counts:** a team above five is
always wrong. Area-rule parameters are fitted on Furia–Falcons and Spirit–Falcons; results
on held-out NaVi–Falcons:

| NaVi–Falcons (held out) | Old | Area split | Area split + cap | Discs + markers + cap |
|---|---|---|---|---|
| All-alive frames showing 5 (mean) | 8.9% | 63.4% | 96.6% | 53.4% |
| Freeze, CT | 0.0% | 95.9% | 100% | 0.6% |
| First 8 s, CT | 15.8% | 63.2% | 98.9% | 77.0% |
| First 8 s, T | 19.8% | 63.5% | 90.5% | 70.9% |
| Whole-match frames with a count > 5 | 0.03% | 23.6% | 0% | 0% |

**Visual review of live play.** The all-alive check is dominated by freeze-time spawn
stacks, where eager splitting happens to be right, and it cannot see over-counting when
fewer than five are alive. Reviewing NaVi frames by eye (`runs/compare-navi-v3`):

| Frame | Visible truth | Old | Area + cap | Discs + markers + cap |
|---|---|---|---|---|
| Round 6, late | 1 T, 3 CT | 1 T, 2 CT | 1 T, 5 CT | 1 T, 3 CT |
| Round 14, three ✕ markers | 1 T, 3 CT | 1 T, 3 CT | 5 T, 5 CT | 1 T, 3 CT |
| Round 5, A execute | 5 T, 5 CT | 4 T, 2 CT | 5 T, 5 CT | 5 T, 5 CT |
| Round 7, mid fights | 5 T, 4 CT | 2 T, 1 CT | 5 T, 5 CT | 5 T, 4 CT |

The disc fit is the most accurate in live play; its weakness is tightly stacked spawn
clusters, where five heavily overlapped icons fit best as four discs. The area rule with
the cap scores higher on the all-alive check because of those clusters, but over-counts
small groups and death markers later in rounds. The live preview uses the disc fit.

## Limitations and next steps

- Both checks are proxies; a small hand-labelled set of live-play frames is needed for
  real precision/recall and to settle the trade-off.
- Stacks of four or five in spawn are under-counted. Tracking (players seen separately
  just before stacking) or a trained detector should recover them.
- A ✕ merged with living icons keeps their arrows and is not removed (it fades in ~2 s).
- k-means centres of heavily overlapped discs are biased towards each other.
