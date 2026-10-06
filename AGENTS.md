# Repository Guidelines

## Scope and Documentation

Use UK English and direct explanations. StratScout ultimately recognises five
Mirage tactics: `A_DEFAULT`, `B_DEFAULT`, `A_SPLIT`, `B_SPLIT`, `RETAKE`. Preserve
video-only inference; `.dem` files support development ground truth only. Do not
change the taxonomy or architecture without explicit agreement.

Read [README.md](README.md) for setup and limitations,
[the implementation plan](docs/implementation-plan.md) for ordered milestones,
and [annotation guidelines](docs/annotation-guidelines.md) before collecting labels.
The current milestone is colour-based T/CT detection, not tactical classification.

## Project Structure

- `stratscout/`: standalone calibration, detection, streaming export and evaluation.
- `tests/`: `unittest` checks with generated images/videos; no real-footage accuracy claim.
- `docs/`: implementation and annotation protocols.
- `requirements.txt`: pinned NumPy and OpenCV dependencies.
- Raw recordings and generated runs are local, ignored assets; no legacy prototype
  or trained checkpoints are required.

## Development Commands

Run from the repository root using Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m stratscout --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dependencies are pinned for the detector. Do not add PyTorch, Matplotlib or Jupyter
merely to run colour detection.

## Coding and Validation

Use four-space indentation, `snake_case`, `PascalCase` classes, uppercase constants
and type hints. Document BGR/RGB order, array shapes, timestamp units and coordinates.
No formatter, linter or numeric coverage threshold is configured.

Add meaningful `tests/test_*.py` cases for changed behaviour. Prioritise crop/label
alignment, separate team outputs, missing observations, file preservation and video
I/O failures. Synthetic checks do not replace reviewed, held-out real footage.

## Data Safety and Contributions

Never overwrite reviewed labels with proposals. New profiles/runs refuse existing
destinations. Preserve new source recordings and reviewed labels. The retired
prototype is recoverable from Git history; do not reinterpret its synthetic metrics
as real-data evidence or reintroduce it as a runtime dependency.

History uses short descriptive commits without a formal convention. Prefer
action-led subjects such as `Fix crop coordinate conversion`. PRs should explain
the problem, change, validation and limitations, link relevant issues, and include
before/after overlays for visual changes. Avoid adding caches, raw recordings or
large model files; document required asset locations instead.
