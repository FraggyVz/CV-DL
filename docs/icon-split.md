# Overlapping Player Icons

## Problem

The colour baseline (`segmentation.find_icons_teams`) treats each connected blob of
saturated colour as one player. Overlapping icons of one team merge into a single blob,
so a player standing behind a teammate disappears. Blobs above `icon_area` are dropped
entirely, which removes whole spawn clusters during freeze time. One mask is shared by
both teams, so a T touching a CT also merges and is assigned one team by median hue.

## Method

`icon_split.find_icons_split` builds one colour mask per team (hue ± 6, saturation and
value ≥ 90, minus the empty-map median background) and drops badge-shaped blobs. Every
icon is a disc of the same size, so a blob of at least `split_ratio` × one icon's area
holds several: each extra icon adds about `extra_area` × that area, because overlapping
discs share pixels. Centres come from k-means on the blob's pixel coordinates.

## Evaluation without labels

During freeze time and the first 8 s after it, nobody has died, so a correct detector
shows 5 T and 5 CT. A hidden bomb carrier (badge rim check) counts for the T side.
`icon-count-eval` measures the share of such frames showing exactly 5 per team, fits
`single_area` (median early-play blob area) and grid-searches `split_ratio` and
`extra_area` on training matches, then scores held-out matches.

```powershell
python -m stratscout icon-count-eval --train furia spirit --test navi --output runs/icon-count-eval
```

Fitted on Furia–Falcons and Spirit–Falcons (5 fps): `single_area` 246, `split_ratio` 1.15,
`extra_area` 0.6. Share of all-alive frames showing exactly 5:

| NaVi–Falcons (held out) | Baseline | Split |
|---|---|---|
| Freeze, T | 0.0% | 30.9% |
| Freeze, CT | 0.0% | 95.9% |
| First 8 s, T | 19.8% | 63.5% |
| First 8 s, CT | 15.8% | 63.2% |
| Mean of the four | 8.9% | 63.4% |

The training matches gave 63.9% and 64.5%, so the fitted values transfer. Freeze-time T
counts stay low because the carrier's badge sits inside the spawn cluster and cuts a hole
in the T blob. The live preview now uses the split detector.

## Limitations

- A count proxy, not labelled precision/recall: it does not check positions, and right
  totals can hide a miss plus a false detection. Later-round frames with deaths are not
  covered.
- Completely hidden icons cannot be recovered from area; tracking or a trained detector
  is needed.
- k-means centres of heavily overlapped discs are biased towards each other.
