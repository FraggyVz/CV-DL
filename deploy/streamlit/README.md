# Streamlit deployment (cloud)

A small web page that classifies prepared radar clips with the final models. It calls
the same prediction code as `predict-tcn`, `predict-gru` and the web app
(`stratscout.gru_predict.predict_clips`). It runs the detector, then the ResNet-18
features, then the TCN, and optionally the Compact pooled GRU. Each model averages its
three seeds.

The full React + FastAPI site (`docker compose up`) is the local deployment. This page
is the cloud deployment, on Streamlit Community Cloud's free tier.

## Files

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The page |
| `requirements.txt` | Cloud packages: CPU-only PyTorch, OpenCV (headless), Streamlit |
| `backgrounds/*.png` | Empty-map images, one per training broadcast layout, made with `save-background` from the training clips only |
| `samples/*.mp4` | Three held-out **test** clips for the demo (never used in training) |

The models are read from `runs/tactic-tcn-clipdata-final/models` and
`runs/tactic-gru-clipdata-final-20261121/models`, which are already tracked in git.

## Run locally

From the repository root, in the project's virtual environment:

```powershell
pip install streamlit==1.67.0
streamlit run deploy/streamlit/streamlit_app.py
```

The page opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud (free)

1. Push the branch to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub and choose **Create app**.
3. Pick the repository and branch, and set the main file path to
   `deploy/streamlit/streamlit_app.py`.
4. Under **Advanced settings**, choose Python **3.13**.
5. Deploy. The first build installs PyTorch and takes a few minutes. The first
   classification also downloads the ResNet-18 weights (about 45 MB).

Streamlit uses `deploy/streamlit/requirements.txt` because it sits next to the app, not
the repository's full `requirements.txt`. The app sleeps when unused and wakes on the
next visit.

## What the page accepts

- Cropped spectator-radar clips at 10 fps, one tactic each, as `export-segments` makes
  them. Full matches are not supported.
- The detector needs an empty-map background of the same broadcast. Clips named like
  the dataset (`<match>-<event>.<reviewer>.s<nn>.mp4`) get their event's background
  automatically. For other names, pick the layout, or upload 5 or more clips of the same
  match and build the background from them.
- Layouts with a saved background: ESL Pro League Season 24, Cologne Major 2026, Cologne
  Major, IEM Beijing 2026.
