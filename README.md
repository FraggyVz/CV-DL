# StratScout

StratScout is a CS2 Mirage research prototype: minimap pixels → player detection →
tracking → attacker tactics before the plant and defender responses afterwards.
Demo files are development evidence; the vision pipeline takes video pixels.

## Current status

Ryan's prototype has been moved into the `stratscout/` Python package. Both notebooks
are thin, optional experiment interfaces importing that implementation. Importing
modules does not extract data, train models, open windows or save results.

The colour baseline, CNN detector, tracker, four classifier architectures, annotation
tools, reports and companion demo-ingest utilities are retained. **This is an
experimental pipeline, not a validated real-match classifier.** Classifier training
still uses generated rounds; real icon labels can supply the detector separately.
Existing tracking, label-safety and ingest-schema problems are recorded in the
[baseline review](docs/base-review-2026-10-12.md).

## Windows setup

Use Python 3.13. The refactor was checked on Python 3.13.14 with CPU Torch.
Create a new environment if an existing one points to a moved interpreter:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The Python 3.13 launcher entry must exist; otherwise run its executable directly.
During this refactor, the working environment was created at `.venv-modular/`;
activate it with `.\.venv-modular\Scripts\Activate.ps1` to use it locally.
The requirements include the companion's pandas/demoparser2 dependencies.
For notebooks, select this environment in your editor; install `ipykernel` there
if your editor requests it. For linting, install `requirements-dev.txt`.

## Run and check

From the repository root:

```powershell
python -m stratscout --help
python -m stratscout prototype --smoke --output runs/smoke-001
python -m unittest discover -s tests
```

The smoke command runs a small synthetic detector/classifier experiment on CPU and
writes `detector.pt`, `stratscout.pt` and `summary.json`. It does not read your real
recordings or annotations. **Its scores are wiring checks, not accuracy evidence.**
Choose a new output directory for every run; existing destinations are refused.

For the full prototype, use a JSON object containing `Config` fields:

```json
{
  "data_dir": "data/raw/falcons-vs-furia",
  "map_image": "mirage_radar.png",
  "det_epochs": 25,
  "cls_epochs": 40
}
```

Save it as `configs/local.json`, then run:

```powershell
python -m stratscout prototype --config configs/local.json --output runs/experiment-001
```

Add `--plots` for interactive figures or `--device cuda` for a suitable GPU Torch
installation. The full workflow can extract/cache pilot frames and crop metadata;
read the annotation hazards below before using reviewed data. Video discovery scans
the configured directory directly, not recursively. One file represents one match.

## Notebooks and demo ingest

- `stratscout_prototype.ipynb`: inspect each vision/training/report step.
- `stratscout_dem_ingest.ipynb`: radar-video exports and optional demo parsing/mapping.
- `python -m stratscout ingest --stage radar --config configs/ingest.json`: explicit
  radar export using a `DemCfg` JSON object and existing `crops.json`.
- Other ingest stages are `parse`, `map` and `export`; export requires reviewed demo
  labels. **The companion's legacy label schema is incompatible with the main
  two-head schema.** Its outputs are not automatically wired into training.

The companion defaults to 8 fps; `radar_fps` is configurable. The proposed 10 fps
workflow remains unvalidated. OpenCV falls back from H.264 to `mp4v` when H.264 is
unavailable, so compression and file sizes depend on the local encoder.

## Development and limitations

See [module responsibilities and API examples](docs/architecture.md),
[the recording direction](docs/development-direction.md) and
[contributor rules](AGENTS.md).

- Tracker placeholders/coasting and feature handling of missing players are unchanged.
- Legacy annotation proposal/save behaviour still needs repair. Back up reviewed
  labels; never direct `auto_label_icons` at a reviewed annotation file.
- Real icon splitting still needs match-separated manifests.
- The leaking B-prime experiment is retained but disabled by default.
- Full CS2/OBS capture automation and a new safe batch encoder are not implemented.
- Real recordings, pilot data, existing annotations and checkpoints were preserved.
