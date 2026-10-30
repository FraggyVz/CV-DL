# Repository Guidelines

## Scope and Communication

Use UK English. Be direct about faults and distinguish synthetic/proxy results from
measured real-footage performance. Continue from the shared modular prototype.
Read [README](README.md) for current status, limitations and next steps. Use the
[recording guide](docs/recording-setup.md) for capture/preparation and the
[tactic guide](docs/tactic-definitions.md) for labelling. Archived notes are historical.

## Structure and Style

Implementation belongs in `stratscout/`; companion demo utilities in `stratscout/ingest/`.
Use four spaces, `snake_case` functions/modules and `PascalCase` classes. Pass
`RunContext` or `IngestContext` explicitly. Keep imports one-way and free of training,
extraction, GUI and writing side effects. Never import implementation from notebooks.
Keep inbox processing in `process-recordings` and encoding in the shared preparation API.

## Checks

Use Python 3.13, `requirements.txt` and `requirements-dev.txt`. From the root:

```powershell
python -m unittest discover -s tests
ruff check stratscout tests
ruff format --check stratscout tests
python -m stratscout prototype --smoke --output runs/new-smoke-run
```

Use a fresh output directory. Add focused regression checks for changed contracts.
For documentation-only work, check references and formatting; report existing test
blockers honestly. Synthetic checks do not establish detector accuracy.

## Data and Collaboration

Preserve masters, reviewed annotations and checkpoints. Never target reviewed labels
with automatic proposals or regenerate indexed frames without preserving provenance.
Capture is 1440p/60 fps at constant 2× replay; reuse a crop only after checking the layout.
Keep observed, missing and inferred positions distinct and demo truth out of inference.
Group all variants and clips by match when splitting datasets. Fit spectator-radar
settings on Furia/Spirit; keep NaVi held out. Count proxies are not accuracy metrics.

The tactic guide owns the five-class target; legacy annotation code still needs migration.
Do not silently rename labels or describe planned tools as implemented. Use short
imperative commit messages and bounded changes. Handoffs state changes, checks and
limitations. Update the relevant guide rather than duplicate it in another document.
