# Clip data

Dataset: `data\processed\datasets\2026-11-16`. Regions: `mirage-spectator-v1`. ResNet features stored: yes. Built by `clip-data-1`.

## Region coverage

Share of player detections that fall in a named region (the rest are unplaced).

T detections placed, mean over clips: 100.0%.

## Does the data separate the classes? (training split only)

Share of clips where Ts reach a region, the median split gap when both approaches are used, and the share with an estimated plant.

| Class | Clips | reach connector | reach ramp/palace | reach short | reach apartments | A gap s (median) | B gap s (median) | plant est. |
|---|---|---|---|---|---|---|---|---|
| A_DIRECT | 15 | 60% | 100% | 7% | 13% | 8.6 | 8.6 | 20% |
| B_DIRECT | 24 | 0% | 8% | 8% | 100% | – | 6.4 | 0% |
| A_SPLIT | 15 | 100% | 100% | 7% | 13% | 1.0 | – | 7% |
| B_SPLIT | 9 | 11% | 11% | 89% | 100% | – | 1.3 | 11% |
| RETAKE | 20 | 25% | 25% | 5% | 50% | 0.2 | 17.0 | 75% |

Regions and the plant are estimates from an unvalidated colour detector and a draft region map; see docs/clip-data.md.
