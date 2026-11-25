# Streamlit deployment (cloud)

The cloud version of the local web app. The local app is React + FastAPI, run with
`docker compose up`. The cloud page follows the same sections: recording, analysis
flow, match rounds, video review and classification.

**It runs the local app's own code:**
- **Analysis:** each round goes through `stratscout.web_analysis.queue_analysis` and
  `run_analysis`, as the local analysis endpoint does. That covers the ffmpeg
  preparation, the detector, the ResNet-18 features, both saved ensembles (TCN and
  Compact pooled GRU, three seeds each) and the stage-by-stage pipeline record. The same
  clip gives the same numbers on both.
- **Classification section:** this is the local app's React results view itself
  (`frontend/src/viewer.tsx`, built into `results_component/`), embedded as a Streamlit
  component. The TCN vs GRU / TCN only / GRU only tabs, the round picker, the prediction
  cards, the walkthrough and the pipeline graphs are identical.

## How it differs from the local app

| | Local app | Cloud workspace |
|---|---|---|
| Input | Raw recordings (cropped with `configs/recording-video.json`) or prepared clips | **Prepared clips only** (494x500 radar crop, 10 fps); the prepared-clip box is always ticked |
| Matches | Several saved matches, kept on disk | One match per browser session; nothing is kept after the tab closes |
| Clip boundaries | Automatic for new rounds; older recordings can be edited | Automatic: the whole clip is one round |
| Sample clips | No | Three held-out test clips bundled |

## Files

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The page |
| `results_component/` | The built React results view. Rebuild it after changing the result cards in `App.tsx`, `ModelPipeline.tsx`, `viewer.tsx` or `app.css` (see below). |
| `requirements.txt` | Python packages: CPU-only PyTorch, OpenCV (headless), Streamlit |
| `../../packages.txt` | System packages: `ffmpeg`, which the local app's preparation step uses. It must sit at the repository root: Streamlit Cloud doesn't read it from the app's folder. |
| `samples/*.mp4` | Three held-out **test** clips for the demo (never used in training) |
| `../../.streamlit/config.toml` | The dark theme with the local app's colours, read from the repository root |

The models are read from `runs/tactic-tcn-clipdata-final/models` and
`runs/tactic-gru-clipdata-final-20261121/models`, which are already tracked in git.

## Rebuild the results view

From `frontend/`, with Node 22:

```powershell
npm ci
npm run build:streamlit
```

This writes `deploy/streamlit/results_component/`. Commit that folder, because Streamlit
Community Cloud has no Node. Without Node installed, run the same build in Docker:

```powershell
docker run --rm -v "${PWD}:/repo" -w /repo/frontend node:22-alpine sh -c "npm ci && npm run build:streamlit"
```

## Run locally

The analysis needs ffmpeg on PATH, as the local app does. The simplest way is the same
Linux setup the cloud uses, in Docker, from the repository root:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}:/repo:ro" python:3.13-slim sh -c "apt-get update -qq && apt-get install -y -qq ffmpeg && pip install -q -r /repo/deploy/streamlit/requirements.txt && cd /repo && streamlit run deploy/streamlit/streamlit_app.py --server.address 0.0.0.0"
```

Then open http://localhost:8501. With ffmpeg installed on Windows instead:
`pip install streamlit==1.67.0`, then `streamlit run deploy/streamlit/streamlit_app.py`.

## Deploy to Streamlit Community Cloud (free)

Streamlit needs admin rights on the GitHub repository. Use the repository owner's
account, or a fork or copy you own.

1. Push the branch to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub and choose **Create app**.
3. Pick the repository and branch, and set the main file path to
   `deploy/streamlit/streamlit_app.py`.
4. Under **Advanced settings**, choose Python **3.13**.
5. Deploy. The first build installs PyTorch and ffmpeg and takes a few minutes. The first
   analysis also downloads the ResNet-18 weights (about 45 MB).
6. For a private repository, set the app to public in its **Share** settings so viewers
   don't need to sign in. The code stays private.

Streamlit reads `requirements.txt` from the app's folder (`deploy/streamlit/`) but
`packages.txt` (ffmpeg) from the repository root. After changing `packages.txt`, reboot
the app from **Manage app** so the system packages are installed again. The app sleeps
when unused and wakes on the next visit.
