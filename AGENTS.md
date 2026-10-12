# Repository Guidelines

## Scope and Communication

Use UK English. Be direct about faults and distinguish synthetic results from measured
real-footage performance. Continue from the shared prototype; avoid wholesale rewrites.
Read [README](README.md), [architecture](docs/architecture.md) and the
[baseline review](docs/base-review-2026-10-12.md) before changing behaviour.

## Structure and Style

Implementation belongs in `stratscout/`; companion demo tools belong in
`stratscout/ingest/`. Keep the two root notebooks as thin experiment interfaces.
Use four-space indentation, `snake_case` functions/modules and `PascalCase` classes.
Pass `RunContext` or `IngestContext` explicitly where state is needed. Keep imports
one-way and free of training, extraction, GUI and file-writing side effects.
Do not import implementation from notebooks or duplicate it there.

## Checks

Use Python 3.13 and `requirements.txt`; development tools are in
`requirements-dev.txt`. From the root run:

```powershell
python -m unittest discover -s tests
ruff check stratscout tests
ruff format --check stratscout tests
python -m stratscout prototype --smoke --output runs/new-smoke-run
```

Choose a fresh output directory. Tests use synthetic or temporary inputs; they do
not establish detector accuracy. Add focused regression tests for changed contracts.

## Data and Collaboration

Preserve recordings, reviewed annotations and existing checkpoints. Do not regenerate
indexed pilot frames without preserving their labels and provenance. Legacy annotation
and ingest overwrite hazards remain; never target reviewed labels with proposals.
Group future dataset splits by match. Keep demo-derived truth out of video-only inference.

Use short imperative commit messages. Each handoff should describe changed modules,
checks and remaining limitations. Update documentation with behaviour changes and keep
each contribution bounded enough for the other contributor to review.
