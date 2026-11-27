# Development: commands and code structure

The command-line tools, the preview, the legacy prototype, and how the code is laid
out. Contributor rules are in [AGENTS.md](../AGENTS.md); the checks to run before a
change are in the [README](../README.md#development-checks-and-historical-references).

## Commands

```powershell
python -m stratscout --help
python -m stratscout bomb-preview
python -m stratscout bomb-preview --match navi --round 7
python -m stratscout label-segments --match furia --reviewer <name>
python -m stratscout bomb-scan --video data/processed/recordings/<match>/minimap-5fps.mp4 --output runs/bomb-scan-001
python -m stratscout icon-count-eval --train furia spirit --test navi --output runs/icon-count-eval-001
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --output runs/tactic-tcn-<new> --exclude-uncertain --seeds 3
python -m stratscout train-tactics --dataset data/processed/datasets/2026-11-16 --clip-data runs/clip-data-2026-11-16 --output runs/tactic-tcn-clipdata-<new> --exclude-uncertain --seeds 3
python -m stratscout predict-tcn data/processed/datasets/2026-11-16/test/clips --output runs/predictions-001.csv
python -m stratscout predict-gru <clips> --with-tcn --background runs/backgrounds/<match>.png --output runs/predictions-gru-001.csv
python -m stratscout build-clip-data --dataset data/processed/datasets/2026-11-16 --output runs/clip-data-2026-11-16
python -m stratscout prototype --smoke --output runs/smoke-001
```

Replace `<match>` with a processed folder name. New experiment/scan destinations must
not already exist. The TCN commands are explained in [Tactic TCN](tactic-tcn.md), the
GRU in the [prediction guide](gru-predict.md) and `build-clip-data` in
[shared clip data](clip-data.md).

## The preview

The preview uses Space to play/pause, arrows to seek, K/L for ±5 s, N/B for one frame,
`[`/`]` for estimated rounds and Q to quit. It shows player/bomb estimates; the hidden
carrier is inferred, not directly observed. A stationary bomb without carrier evidence is
a dropped candidate; planted bombs are not separated.

`icon-count-eval` checks counts without reviewed labels; it does not establish detection
accuracy. Fit detector settings on Furia–Falcons and Spirit–Falcons and keep NaVi–Falcons
held out. Do not tune on the test match. Keep every FPS variant and tactical segment of
a match in one dataset split, and reserve validation data separately for model selection.

## The prototype and demo stages

The smoke command trains a tiny synthetic detector and checks tracking, saving
`detector.pt` and `summary.json`. It does not use real recordings or establish accuracy.
For an explicit vision experiment, pass `prototype --config configs/local.json --output
runs/experiment-001`; the JSON contains `Config` fields such as `data_dir`, `video_exts`,
`map_image` and `det_epochs`. Add `--device cuda` for a compatible GPU Torch installation,
or `--plots` for interactive figures. The prototype scans one directory directly and
can write pilot caches; it does not automatically use preparation manifests or correct
raw 2× timing. Establish prepared inputs and safe labels before training on real data.

Companion demo stages remain available via `ingest --stage radar|parse|map|export
--config configs/ingest.json` (choose one stage). Their legacy 8 fps exporter and label
schema are not the current video-preparation or five-class dataset route.

## Project structure and contracts

| Location/modules                                                                                                      | Responsibility                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `stratscout/media.py`, `video_profiles`, `video_preparation`, `recording_batch`, `fps_comparison`                     | Shared preparation and comparison APIs                                                                               |
| `segmentation`, `icon_split`, `bomb`, `rounds`, `live_preview`, `count_eval`                                          | Colour/shape detection, overlap handling, estimates and review                                                       |
| `segments`, `segment_labeller`                                                                                        | Five-class segment records, validation, safe saving and the keyboard labeller                                        |
| `segment_export`, `tactic_features`, `tactic_tcn`, `tcn_predict`                                                      | Clip export, per-frame clip features, the tactic TCN experiment and TCN prediction                                   |
| `gru_predict`                                                                                                         | Final Compact pooled GRU, and TCN/GRU prediction for the web app ([guide](gru-predict.md))                           |
| `clip_data`, `mirage_regions.json`                                                                                    | Shared model-ready clip data: detector channels per map region, bomb state and ResNet features                       |
| `webapp`, `web_analysis`, `web_pipeline`                                                                              | The local web app's API, round analysis and pipeline record ([web app](web-app.md))                                  |
| `preprocessing`, `detector`, `datasets`, `training`, `evaluation`, `synthetic`                                        | CNN icon detector and experiments                                                                                    |
| `tracking`, `annotations`, `config`, `contracts`, `context`, `workflow`, `storage`, `video`, `pilot`, `visualisation` | Tracking, legacy labels, explicit run state and orchestration                                                        |
| `stratscout/ingest/`                                                                                                  | Optional demo parsing, coordinate mapping and legacy exports                                                         |
| `frontend/`, `deploy/streamlit/`                                                                                      | The local app's React frontend and the cloud app ([Streamlit deployment](../deploy/streamlit/README.md))             |
| `tests/`                                                                                                              | Regression, import-safety, media and synthetic workflow checks                                                       |
| `configs/`, `data/`, `runs/`                                                                                          | Local settings, recordings, clip datasets and generated experiments; see [Where files go](../README.md#where-files-go) |
| `mirage_radar.png`                                                                                                    | Map asset; recordings, generated results and environments stay outside Git                                           |

Implementation lives in Python modules, not notebooks. Pass `RunContext` or
`IngestContext` explicitly; imports must not start processing, training, GUIs or writes.
OpenCV returns BGR; detector tensors use RGB. Preserve uint8 input where preprocessing
divides by 255, T/CT heatmap channel order (0/1) and positive centre peaks. Keep encoding
in the shared preparation API. Record observed/missing/inferred positions distinctly;
never turn a team-size cap into invented detections. Demo truth and round winners are
annotation/evaluation evidence, not vision-model inputs.
