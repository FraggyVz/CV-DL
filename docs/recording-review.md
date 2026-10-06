# Falcons/FURIA Mirage: Initial Recording Review

Reviewed on 7 October 2026. This is development evidence, **not held-out accuracy**.
Match description comes from the supplied filename/user; tournament identity has
not been independently verified.

## Input and preserved profiles

- Video: `data/raw/falcons-vs-furia/Falcons vs Furia Major Final.mp4`.
- Resolution: 2560 × 1440; nominal frame rate: 60 fps.
- SHA-256: `55df9095664c1e3b1b2cc052f5719e221912cc73cee9b770a70a5cf7b97a647b`.
- User calibration: `configs/falcons-furia-mirage.json`, preserved unchanged.
- Revised calibration: `configs/falcons-furia-mirage-v2.json`.
- Both use native crop `[16, 12, 508, 510]`: 492 × 498 pixels.

The revision keeps the user's sampled hues. T minimum brightness increases from
90 to 190, minimum coloured area falls from 190 to 50 pixels, maximum area becomes
550, and maximum aspect ratio falls from 3 to 1.8. CT minimum brightness increases
from 90 to 170 and minimum coloured area falls from 151 to 50 pixels. Remaining
settings are unchanged. No detection algorithm or map exclusion region was added.

This separates bright icon pixels from darker site/death markers while accepting
smaller visible portions of numbered icons. Settings are recording-specific.

## Development samples and observations

Inspected source timestamps: 30, 45, 60, 90, 120, 180, 195, 210, 225 and 240 seconds.
The additional preview covers source frames `[7080, 7380)`, or 118–123 seconds.
All are calibration/development data and must be excluded from held-out evaluation,
along with the rest of their rounds. Exact round IDs and the user's earlier manual
calibration exposure must be recorded before selecting evaluation rounds.

| Source time | Saved profile | Revised profile | Visual finding |
| --- | --- | --- | --- |
| 01:00 | T=1, CT=1 | T=1, CT=1 | Old CT point is on a death cross; new point is on the visible live CT. Equal counts do not imply equal correctness |
| 02:00 | T=3, CT=5 | T=4, CT=5 | Recovers the yellow portion of a carrier partly covered by the bomb marker; centre may remain biased |
| 03:15 | T=4, CT=5 | T=5, CT=5 | Recovers a numbered T icon rejected by the old area threshold |
| 03:30 | T=4, CT=4 | T=4, CT=4 | Carrier is heavily obscured and nearby CT icons merge; unresolved |
| 04:00 | T=2, CT=4 | T=3, CT=4 | Recovers another small coloured component; occlusion and overlapping CTs remain |

No A/B labels or separate orange bomb markers were identified as detections in
the ten inspected revised-profile stills. This is a visual spot check, not a
precision/recall calculation. At 01:30, clustered icons are still severely
undercounted. Do not infer unseen centres from bomb markers or force five players.

## Reproduce the preview

Run from the repository root; choose a new output directory if it already exists:

```powershell
.\.venv\Scripts\python.exe -m stratscout detect --video "data/raw/falcons-vs-furia/Falcons vs Furia Major Final.mp4" --profile configs/falcons-furia-mirage-v2.json --output runs/falcons-furia-carrier-preview-v2 --start-frame 7080 --max-frames 300
```

The initial run exported and re-decoded 300 annotated frames, preserving source
indices 7080–7379 and source timestamps in JSONL. Local comparison images and the
development selection/provenance are in `runs/carrier-investigation/`. These outputs
and the recording remain ignored by Git. The automated suite passes 31 tests,
including preview timestamps, range limits, overwrite protection and early decode failure.

## Remaining work

1. Review the preview for flicker and positional bias, especially the carrier.
2. Label at least 20 untouched frames from separate rounds under the
   [annotation protocol](annotation-guidelines.md); retain difficult/unscorable cases
   in an audit. Initial tuning still needs formal calibration labels and round IDs.
3. Measure one-to-one precision and recall separately for T/CT. No 90% claim yet.
4. Investigate overlaps and occlusion using visible evidence. The colour baseline
   cannot recover a centre completely covered by another HUD symbol.
5. Check recording timing and full-file decoding before the whole-match export.
   OpenCV advertises 163479 frames, but a tail seek/decode stopped one frame short.
   Nominal 60 fps does not establish constant frame timing. Preserve the original
   while diagnosing this discrepancy; the short preview does not resolve it.
