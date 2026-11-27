# Web application

The local web app (React + FastAPI, run in Docker) uploads one MP4 per round, analyses
it, and shows the TCN and GRU predictions with the pipeline behind them. The cloud
version runs the same analysis code; see the
[Streamlit deployment](../deploy/streamlit/README.md) for how it differs.

## Run it

Install Docker Desktop, then from the repository root run:

```powershell
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The service binds to localhost;
it is not exposed to the network. The first build downloads the pinned frontend and
Python web dependencies. Stop it with Ctrl+C, or run `docker compose down`.

## Use it

Create a match, choose whether the uploaded tactic clips are attacks or retakes, then
upload one MP4 per round (or select several to upload them sequentially). Each clip is
assigned the next round number and treated as one complete tactic interval. Select a
round and choose **Analyse Round** to prepare the entire clip (first frame to last), run
the player/bomb detector, and compare the saved TCN and GRU predictions. **Analyse all
rounds** starts the pending analyses together; the classification panel shows results for
every round in the selected match. Playback speed is shared, while each round's
prepared-clip setting is saved with that recording and used independently. Previous round
results stay in that match's round list when later clips are uploaded. Use **Remove round**
to delete a mistaken upload and its analysis; later rounds in that match are renumbered.
Use **New match** to start a separate sequence; existing matches and recordings remain
saved. Empty match workspaces and the current selection are also saved in that browser,
so a match remains selectable before its first round is uploaded. Predictions below 0.6
confidence are flagged for review; they are not ground truth.

### The pipeline view

In the **TCN only** and **GRU only** tabs, each prediction is paired with a walkthrough
of clip preparation, detector counts, feature extraction and model scoring. Below it, the
round's own values are drawn stage by stage:
- detected icons on the radar with the map regions, and icon counts over time
- region occupancy heatmaps
- the ResNet-18 PCA features and the scaled model input
- the TCN's five dilated residual blocks, or the GRU's projection and hidden state
- mean and max pooling
- each seed's probabilities and their average

The analysis saves these values in `stratscout/web_pipeline.py`'s format, and the app reads
them from `GET /api/videos/{id}/analysis/pipeline/{segment_id}`. Rounds analysed before
this view existed need to be analysed again. It explains the recorded inference pipeline
and class ranking; it is not a feature-attribution explanation of why the model chose a
tactic.

### Older recordings and reset

Older recordings uploaded before match rounds were introduced remain unassigned and
available under **Previous recordings**. Existing saved segments are preserved; an older
recording with no segments defaults to its full duration. **Reset workspace** deletes
all app uploads and analyses across every match, including these previous recordings,
after confirmation. It does not delete source footage, labels, or checkpoints. Reset is
blocked while an analysis is running. The app does not discover round boundaries in a
full-match recording; upload one clip that already represents one round/tactic.

## Prepared clips and full-screen recordings

New uploads start with the prepared-clip option ticked, for already-prepared minimap clips
(494×500 at 10 fps). Untick it in the round panel for a full-screen recording, and
create the per-machine crop profile at `configs/recording-video.json` using the
[recording guide](recording-setup.md) before running analysis. The app defaults to the
guide's 2× replay speed for full-screen recordings; choose 1× if that recording was
captured at normal speed. Prepared clips use their source speed. New-round clips are
analysed from start to finish.

## Storage and the Docker image

Videos that the browser cannot decode get a separate H.264/yuv420p playback copy; the
uploaded original is preserved. Prepared analysis clips, manifests and result status are
also kept under `data/web/uploads/analysis/` for provenance. Docker Compose mounts
the tracked TCN and GRU checkpoints from the checkout. The image includes the CPU
PyTorch/OpenCV inference stack, FFmpeg and the ImageNet ResNet weights. The first build
takes longer because it installs these packages and downloads the ResNet weights. Docker
is not required for the Python research commands in the README.

Uploads are limited to 20 GiB by default. Set `STRATSCOUT_MAX_UPLOAD_BYTES` before
starting Compose to change the byte limit. Keep sufficient free disk space: videos are
staged during upload, then stored in the ignored `data/web/uploads/` folder. A
browser-compatible copy and analysis outputs need additional space. Uploads and generated
files are not deleted when the container stops. Files manually extracted into nested
dataset folders are not automatically registered as recordings; use the upload control
for a match video.

## Frontend development

For frontend development outside Docker, use Python 3.13 and Node.js 22. In one
PowerShell window install the web dependencies and start the API:

```powershell
python -m pip install -r requirements-web.txt
python -m uvicorn stratscout.webapp:app --host 127.0.0.1 --port 8000
```

In another window, install the pinned frontend packages and start Vite:

```powershell
Set-Location frontend
npm ci
node .\node_modules\vite\bin\vite.js --host 127.0.0.1
```

Vite runs at `http://127.0.0.1:5173` and proxies API requests to the local Python
service. To build assets from PowerShell, run:

```powershell
node .\node_modules\typescript\bin\tsc --noEmit
if ($?) { node .\node_modules\vite\bin\vite.js build }
```

Frontend unit tests use Node's built-in test runner and need no extra packages:
`node --experimental-strip-types --test tests/*.test.ts` from `frontend/` (or `npm test`).

After changing the result cards (`App.tsx`, `ModelPipeline.tsx`, `viewer.tsx` or
`app.css`), rebuild the cloud app's copy as described in the
[Streamlit deployment](../deploy/streamlit/README.md#rebuild-the-results-view).
