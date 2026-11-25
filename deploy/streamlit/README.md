# Streamlit deployment (cloud)

A cloud version of the local web app's classification flow. The local app is React +
FastAPI, run with `docker compose up` (Frontend branch). This page keeps the same
wording, result cards (TCN vs GRU, TCN only, GRU only), colours and model calls, so the
same clip gives the same answer on both.

It runs what `stratscout/web_analysis.py` runs. Each round clip builds its own
empty-map background. Then the detector and the ResNet-18 features run once, and both
saved ensembles classify the clip: the TCN and the Compact pooled GRU, three seeds each,
with a 0.6 confidence threshold for the UNCERTAIN badge.

## How it differs from the local app

| | Local app | Cloud demo |
|---|---|---|
| Input | Raw round recordings (cropped with `configs/recording-video.json`) or prepared clips | **Prepared clips only** (494x500 radar crop, 10 fps) |
| Saved state | Match list, rounds and analyses kept on disk | Nothing kept; results last for the browser session |
| Video player and clip boundaries | Yes | No |
| Sample clips | No | Three held-out test clips bundled |

## Files

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The page |
| `requirements.txt` | Cloud packages: CPU-only PyTorch, OpenCV (headless), Streamlit |
| `samples/*.mp4` | Three held-out **test** clips for the demo (never used in training) |
| `../../.streamlit/config.toml` | The dark theme with the local app's colours. It is read from the repository root. |

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

Streamlit needs admin rights on the GitHub repository. Use the repository owner's
account, or a fork or copy you own.

1. Push the branch to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub and choose **Create app**.
3. Pick the repository and branch, and set the main file path to
   `deploy/streamlit/streamlit_app.py`.
4. Under **Advanced settings**, choose Python **3.13**.
5. Deploy. The first build installs PyTorch and takes a few minutes. The first analysis
   also downloads the ResNet-18 weights (about 45 MB).
6. For a private repository, set the app to public in its **Share** settings so viewers
   don't need to sign in. The code stays private.

Streamlit uses `deploy/streamlit/requirements.txt` because it sits next to the app, not
the repository's full `requirements.txt`. The app sleeps when unused and wakes on the
next visit.
