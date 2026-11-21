# Final Compact pooled GRU on the test split

Test clips: 45 from 2 matches (navi-falcons-mirage-cologne-major: 24, vitality-falcons-mirage-esl-pro-league-season-24: 21). The GRU was run once, after model selection; nothing was trained or tuned. Backgrounds: one per match, from that match's test clips, as predict_clips builds them.

TCN check: predict_clips reproduced the saved TCN test predictions of `runs\tactic-tcn-clipdata-final\predictions.csv` (45/45 same class, largest probability difference 0.0000).

| GRU | Macro-F1 |
|---|---|
| Pooled over all clips, mean ± SD over seeds | 0.890 ± 0.030 |
| Seed ensemble (accuracy 0.956) | 0.963 |
| Seed ensemble, per match then averaged (for comparison with cross-validation) | 0.962 |
| Majority class (B_DIRECT) | 0.047 |

Per-seed and ensemble predictions: `predictions.csv`; the app's output for both models: `app-results.json`. Analysis: `notebooks/tactic-final-test.ipynb`.
