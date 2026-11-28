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

## How the deployed models are kept fast

Both apps, local and cloud, run the same analysis on CPU. Per clip, measured on the
development laptop for clips of 19 to 33 s
([timings](gru-predict.md#time-per-clip-cpu)):

| Step | Time per clip | Share |
|---|---|---|
| Detection (colour rules, every frame) | about 11 s | most of it |
| ResNet-18 features (every 2nd frame) | about 3 s | most of the rest |
| TCN, 3 seeds | about 0.1 s | under 1% |
| GRU, 3 seeds | about 0.07 s | under 1% |

The choices that keep it at this level:

- **Small models.** The TCN has 35,365 trainable weights per seed and the GRU 2,349.
  All six checkpoints together, including each seed's saved scaling and ResNet
  compression, are 1.7 MB and ship in the repository, so neither app downloads a model.
- **Features computed once.** Detection and the ResNet features run once per clip, and
  both ensembles read the same features.
- **Small, fixed input.** Analysis runs on the 494 × 500 radar crop at 10 fps, never on
  the full-screen video. The detector is colour and shape rules, with no neural network.
- **Frozen ResNet-18 in inference mode.** The encoder runs in evaluation mode without
  gradient tracking. The Docker image downloads its ImageNet weights at build time, not
  on the first request.
- **Models loaded once.** The ensembles are cached in memory after the first load (about
  0.1 s); the cloud app loads them when it starts (`st.cache_resource`).
- **CPU-only PyTorch.** Both apps install the CPU wheels, which keeps the image small and
  lets the cloud app run on Streamlit Community Cloud, which has no GPU.
- **Work off the request.** Each analysis runs in a background task (FastAPI) or thread
  (Streamlit), so the page stays responsive and shows progress. A second analysis of a
  recording that is already being analysed is refused.

### Tried: int8 quantisation of ResNet-18

ResNet-18 is the only neural network in the analysis that takes noticeable time, so it
was converted to int8 with post-training static quantisation, calibrated on 8 training
clips. The [quantisation notebook](../notebooks/quantised-resnet.ipynb) has the charts;
`scripts/quantised_resnet_demo.py` runs the same check from the command line.

| | float32 (used) | int8 (tried) |
|---|---|---|
| ResNet-18, one batch of 64 frames | 1.06 s | 0.27 s (3.9× faster) |
| ResNet-18 per demo clip, including reading and resizing frames | | 2.1× faster |
| ResNet-18 weights | 45 MB | 11 MB |
| Detector + ResNet time per demo clip | | about 9% less |
| Features (cosine similarity to float32, 512 frames) | | median 0.9987, lowest 0.9979 |
| TCN and GRU predictions on the 3 demo clips | | same class in 6 of 6; largest confidence change 0.057 |

**The apps keep float32.** A user would save about 9% per clip, because the colour
detector still takes most of the time. The reported TCN and GRU results come from
float32 features, and three clips cannot show that predictions never change. Switching
would need the check on all 45 test clips, an int8 model file built once and committed
(the cloud app has no training clips to calibrate on), and a move to `torchao`, because
the `torch.ao.quantization` API used here is deprecated.

**Not done.** The TCN and GRU are not exported to TorchScript or ONNX, quantised or
served with TorchServe: they take under 1% of the time, so this would save little.
TorchServe would also need a separate model server, which Streamlit Community Cloud
cannot run. The larger target is detection (about 11 s per clip), but running the
detector on fewer frames would change the models' input and need the cross-validation
rerun.

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
