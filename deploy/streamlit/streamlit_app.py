"""StratScout cloud demo: the local web app's classification flow on Streamlit.

It mirrors the local React + FastAPI app (``stratscout/webapp.py`` and
``stratscout/web_analysis.py`` on the Frontend branch): the same wording, the same result
cards and the same model calls. Each uploaded round clip builds its own empty-map
background, then the detector and ResNet features run once and both saved ensembles
(TCN and Compact pooled GRU, three seeds each) classify it. Only already-prepared clips
(494x500 radar crop at 10 fps) are accepted, because the cloud has no recording crop
profile.

Run locally from the repository root:

    streamlit run deploy/streamlit/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
import tempfile
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stratscout.config import TACTIC_CLASSES  # noqa: E402

SAMPLES = HERE / "samples"
MODEL_SIZE = (494, 500)  # web_analysis.MODEL_SIZE
MODEL_FPS = 10.0  # web_analysis.MODEL_FPS
MIN_CONFIDENCE = 0.6  # the web app's default min_confidence

CSS = """
<style>
:root { --muted:#8d958e; --accent:#c8f169; --panel:#171b18; --line:#292f2b; --red:#f27b6e; }
.stApp { background: radial-gradient(ellipse at 82% 0%, rgb(200 241 105 / 8%), transparent 36rem), #101311; }
.ss-mono { font-family: Consolas, "Courier New", monospace; letter-spacing: .06em; }
.ss-topbar { display:flex; align-items:center; justify-content:space-between; gap:20px;
  padding: 0 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 8px; flex-wrap: wrap; }
.ss-brand { display:flex; align-items:center; gap:10px; font-weight:600; font-size:16px; color:#e7e9e5; }
.ss-mark { width:28px; height:28px; display:grid; place-items:center; border-radius:7px;
  background: var(--accent); color:#101311; font-weight:800; }
.ss-status { display:flex; align-items:center; gap:8px; color:#a7afa8; font-size:12px; }
.ss-dot { width:8px; height:8px; border-radius:50%; background: var(--accent); }
.ss-badge { margin-left:6px; font-size:10px; color:#a7afa8; }
.ss-eyebrow { color: var(--accent); font-size: 11px; margin: 0 0 8px; }
.ss-hero h1 { font-size: 2.6rem; line-height: 1.1; margin: 28px 0 14px; padding:0; }
.ss-hero p.copy { color:#a7afa8; max-width: 640px; }
.ss-section { display:flex; justify-content:space-between; align-items:flex-end; gap:12px;
  margin: 28px 0 10px; flex-wrap: wrap; }
.ss-section h2 { margin:0; padding:0; font-size:1.35rem; }
.ss-quiet { color:#737d75; font-size: 10px; }
.ss-pipeline { list-style:none; padding:0; margin: 6px 0 8px; display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; }
.ss-pipeline li { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background: var(--panel);
  display:flex; flex-direction:column; gap:2px; }
.ss-pipeline li.complete { border-color: rgb(200 241 105 / 35%); }
.ss-num { font-family: Consolas, monospace; color: var(--accent); font-size: 12px; }
.ss-ptitle { font-weight:600; font-size: 14px; }
.ss-pstate { color: var(--muted); font-size: 12px; }
.ss-note { color: var(--muted); font-size: 13px; }
.ss-round { border:1px solid var(--line); border-radius:10px; background: var(--panel);
  padding: 14px 16px; margin: 0 0 12px; }
.ss-round-head { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.ss-round-num { color: var(--accent); font-size: 11px; margin:0; }
.ss-round h3 { font-size: 15px; margin: 2px 0 0; padding:0; word-break: break-all; }
.ss-round-status { color: var(--muted); font-size: 12px; }
.ss-seg-head { display:flex; align-items:center; gap:10px; margin: 12px 0 10px; color: var(--muted);
  font-size: 12px; flex-wrap: wrap; }
.ss-pill { padding: 3px 7px; border-radius: 4px; background:#263445; color:#a9c7ef;
  font-size: 10px; text-transform: uppercase; }
.ss-pill.retake { background:#402725; color:#f1a49b; }
.ss-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }
.ss-card { min-width:0; padding: 12px; border:1px solid #343d36; border-radius:7px; background:#101411; }
.ss-card-head { display:flex; justify-content:space-between; gap:8px; color:#aab3aa; font-size: 12px; }
.ss-uncertain { color:#f0bf77; font-size: 10px; font-weight:700; }
.ss-tactic { display:block; margin-top: 8px; color: var(--accent); font-family: Consolas, monospace; font-size: 20px; font-weight:700; }
.ss-conf { display:block; margin: 2px 0 10px; color:#89948b; font-size: 11px; }
.ss-probs { display:grid; gap:6px; }
.ss-prow { display:grid; grid-template-columns: 76px minmax(30px,1fr) 34px; align-items:center; gap:8px;
  color:#929b93; font-family: Consolas, monospace; font-size: 11px; }
.ss-prow > span:last-child { text-align:right; }
.ss-track { height:5px; border-radius:3px; background: var(--line); overflow:hidden; }
.ss-track > span { display:block; height:100%; background: var(--accent); }
.ss-warn { margin: 10px 0 0; padding-left: 18px; color:#e4b979; font-size: 12px; }
.ss-notice { display:flex; gap:11px; margin: 22px 0 0; padding: 14px 16px;
  border:1px solid rgb(200 241 105 / 16%); border-radius:9px; background: rgb(200 241 105 / 4%); }
.ss-notice-mark { width:18px; height:18px; flex:0 0 auto; display:grid; place-items:center;
  border:1px solid #879d56; border-radius:50%; color: var(--accent); font-family: Consolas, monospace; font-size: 11px; }
.ss-notice p { margin:0; color:#b7c0b3; font-size: 13px; line-height: 1.55; }
.ss-footer { display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; padding: 22px 0 8px;
  color:#737d75; font-size: 10px; text-transform: uppercase; }
@media (max-width: 640px) { .ss-pipeline { grid-template-columns: repeat(2, minmax(0,1fr)); } .ss-hero h1 { font-size: 2rem; } }
</style>
"""


def esc(value) -> str:
    return html.escape(str(value))


def section(eyebrow: str, title: str, quiet: str = "") -> None:
    st.markdown(
        f'<div class="ss-section"><div><p class="ss-eyebrow ss-mono">{esc(eyebrow)}</p>'
        f"<h2>{esc(title)}</h2></div>"
        f'<span class="ss-quiet ss-mono">{esc(quiet)}</span></div>',
        unsafe_allow_html=True,
    )


def prediction_card(model: str, p: dict) -> str:
    """The web app's PredictionCard."""
    rows = "".join(
        f'<div class="ss-prow"><span>{c}</span><span class="ss-track">'
        f'<span style="width:{min(p["probabilities"][c] * 100, 100):.1f}%"></span></span>'
        f"<span>{round(p['probabilities'][c] * 100)}%</span></div>"
        for c in TACTIC_CLASSES
    )
    badge = '<span class="ss-uncertain">UNCERTAIN</span>' if p["uncertain"] else ""
    return (
        f'<div class="ss-card"><div class="ss-card-head"><span>{esc(model)}</span>{badge}</div>'
        f'<strong class="ss-tactic">{p["tactic"]}</strong>'
        f'<span class="ss-conf">{round(p["confidence"] * 100)}% confidence · next {p["runner_up"]}</span>'
        f'<div class="ss-probs">{rows}</div></div>'
    )


def pipeline(stage: int, failed: bool = False) -> str:
    """The web app's four pipeline steps; ``stage`` is how many are complete."""
    steps = [
        ("Upload", "Ready", "Waiting"),
        ("Prepare & detect", "Complete", "Ready to run"),
        ("Assign round", "Automatic", "Automatic"),
        ("Classify", "Complete", "Waiting"),
    ]
    items = []
    for i, (title, done, waiting) in enumerate(steps, 1):
        complete = i <= stage
        state = done if complete else ("Failed" if failed and i == stage + 1 else waiting)
        items.append(
            f'<li class="{"complete" if complete else ""}"><span class="ss-num">'
            f'{"✓" if complete else i}</span><span class="ss-ptitle">{title}</span>'
            f'<span class="ss-pstate">{state}</span></li>'
        )
    return f'<ol class="ss-pipeline">{"".join(items)}</ol>'


def check_prepared(path: Path) -> None:
    """web_analysis._profile_for's check for already-prepared clips."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if size != MODEL_SIZE or abs(fps - MODEL_FPS) > 0.5:
        raise ValueError(
            f"Already-prepared clips must be 494x500 at 10 fps; "
            f"{path.name} is {size[0]}x{size[1]} at {fps:g} fps."
        )


@st.cache_resource(show_spinner="Loading the saved TCN and GRU ensembles...")
def predictor():
    from stratscout import gru_predict

    gru_predict.load_models("tcn", device="cpu")
    gru_predict.load_models("gru", device="cpu")
    return gru_predict


def detect_and_predict(paths: list[Path], progress) -> list[dict]:
    """web_analysis._detect_and_predict, for clips that are already prepared."""
    gp = predictor()
    tcn_members = gp.load_models("tcn", device="cpu")
    gru_members = gp.load_models("gru", device="cpu")
    steps = {tcn_members[0][2].step, int(gru_members[0][1]["settings"]["step"])}
    if len(steps) != 1:
        raise ValueError(f"TCN and GRU checkpoints use different frame steps: {sorted(steps)}")
    step = steps.pop()
    results = []
    for i, path in enumerate(paths):
        base = i / len(paths)

        def at(fraction, message):
            progress.progress(min(base + fraction / len(paths), 1.0), text=message)

        at(0.10, f"Detecting player and bomb icons in {path.name}.")
        detections, warnings = gp.detect([path], background_from=[path], log=lambda *a: None)
        at(0.45, "Extracting the model's detector and ResNet features.")
        features = gp.clip_features(
            [path], detections, step=step, device="cpu", log=lambda *a: None
        )
        at(0.80, "Running the saved TCN and GRU ensembles.")
        prediction = gp.predict_clips(
            [path],
            models=("tcn", "gru"),
            min_confidence=MIN_CONFIDENCE,
            device="cpu",
            features=features,
        )[0]
        detection = detections[path.stem]
        results.append(
            {
                "detection": {
                    "frames": int(detection["n_frames"]),
                    "player_icon_observations": int(len(detection["players"])),
                    "bomb_visible_frames": sum(bool(f[0]) for f in detection["bomb"]),
                },
                "warnings": list(dict.fromkeys(prediction["warnings"] + warnings[path.stem])),
                "predictions": prediction["predictions"],
            }
        )
    progress.progress(1.0, text="Detection and both model predictions are ready.")
    return results


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="StratScout", page_icon=":material/radar:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<header class="ss-topbar"><span class="ss-brand"><span class="ss-mark">S</span>StratScout</span>'
    '<span class="ss-status"><span class="ss-dot"></span>Models loaded on the cloud service'
    '<span class="ss-badge ss-mono">CLOUD DEMO</span></span></header>'
    '<section class="ss-hero"><p class="ss-eyebrow ss-mono">CS2 · MIRAGE · MATCH WORKSPACE</p>'
    "<h1>Review the round.<br>Read the strategy.</h1>"
    '<p class="copy">Upload prepared round clips, or try the bundled samples, and compare '
    "the TCN and GRU predictions.</p></section>",
    unsafe_allow_html=True,
)

# 01 / UPLOAD ----------------------------------------------------------------
section("01 / UPLOAD", "Build your match rounds", "MP4 · DELETED AFTER ANALYSIS")
samples = sorted(SAMPLES.glob("*.mp4")) if SAMPLES.is_dir() else []
col_source, col_phase = st.columns([2, 1])
with col_source:
    source = st.radio(
        "Clips",
        ["Sample round clips", "Upload my own"] if samples else ["Upload my own"],
        horizontal=True,
        key="source",
    )
with col_phase:
    phase = st.selectbox(
        "Clip type",
        ["attack", "retake"],
        key="phase",
        help="As in the local app: a label shown with the result. It doesn't change the prediction.",
    )

chosen: list[Path] = []
uploads = []
if source == "Sample round clips":
    picked = st.multiselect(
        "Held-out test clips (never seen in training)",
        [p.name for p in samples],
        default=[samples[0].name],
        key="samples",
    )
    chosen = [SAMPLES / n for n in picked]
else:
    uploads = (
        st.file_uploader(
            "Prepared round clips (.mp4, 494x500 radar crop at 10 fps)",
            type=["mp4"],
            accept_multiple_files=True,
            key="uploads",
        )
        or []
    )
names = [p.name for p in chosen] + [u.name for u in uploads]

# 02 / ANALYSIS FLOW ---------------------------------------------------------
section("02 / ANALYSIS FLOW", "From round clip to model output", "PIPELINE STATUS")
flow = st.empty()
done = st.session_state.get("results_for") == names and bool(names)
flow.markdown(pipeline(4 if done else (1 if names else 0)), unsafe_allow_html=True)
st.markdown(
    '<p class="ss-note">Add one MP4 per round, then analyse. Round numbers follow the order '
    "of the clips; full-match round detection is not supported.</p>",
    unsafe_allow_html=True,
)

if st.button("Analyse rounds", type="primary", disabled=not names, key="go"):
    progress = st.progress(0.0, text="Queued.")
    with tempfile.TemporaryDirectory() as tmp:
        paths = list(chosen)
        error = None
        for u in uploads:
            path = Path(tmp) / Path(u.name).name
            if path.exists() or path.name in {p.name for p in chosen}:
                error = f"Two clips are called {path.name}. Rename one and upload again."
                break
            path.write_bytes(u.getbuffer())
            paths.append(path)
        try:
            if error:
                raise ValueError(error)
            progress.progress(0.05, text="Checking the prepared clips.")
            for path in paths:
                check_prepared(path)
            flow.markdown(pipeline(1), unsafe_allow_html=True)
            results = detect_and_predict(paths, progress)
        except Exception as exc:  # Shown like the web app's failed state.
            flow.markdown(pipeline(1, failed=True), unsafe_allow_html=True)
            st.error(
                "Analysis failed. See the error below and correct the input before retrying.\n\n"
                f"{type(exc).__name__}: {exc}"
            )
            st.stop()
    st.session_state.update(results=results, results_for=names, results_phase=phase)
    flow.markdown(pipeline(4), unsafe_allow_html=True)
    progress.empty()

# 03 / CLASSIFICATION ---------------------------------------------------------
results = st.session_state.get("results") if st.session_state.get("results_for") == names else None
section(
    "03 / CLASSIFICATION",
    "All round classifications",
    f"{len(results) if results else 0} / {len(names)} COMPLETE",
)
tabs = st.tabs(["TCN vs GRU", "TCN only", "GRU only"])
views = [("tcn", "gru"), ("tcn",), ("gru",)]
labels = {"tcn": "TCN", "gru": "Compact pooled GRU"}
result_phase = st.session_state.get("results_phase", phase)
for tab, view in zip(tabs, views):
    with tab:
        if not names:
            st.markdown(
                '<p class="ss-note">Add a round clip to see its TCN and GRU output here.</p>',
                unsafe_allow_html=True,
            )
            continue
        for number, name in enumerate(names, 1):
            r = results[number - 1] if results else None
            body = ""
            if r:
                d = r["detection"]
                visible = [w for w in r["warnings"] if not w.startswith("background from only ")]
                cards = "".join(prediction_card(labels[k], r["predictions"][k]) for k in view)
                warns = (
                    '<ul class="ss-warn">'
                    + "".join(f"<li>{esc(w)}</li>" for w in visible)
                    + "</ul>"
                    if visible
                    else ""
                )
                body = (
                    f'<div class="ss-seg-head"><span class="ss-pill {result_phase}">{result_phase}</span>'
                    f"<span>{d['player_icon_observations']} player-icon observations · "
                    f"{d['bomb_visible_frames']}/{d['frames']} frames with bomb visible</span></div>"
                    f'<div class="ss-grid">{cards}</div>{warns}'
                )
            st.markdown(
                f'<article class="ss-round"><div class="ss-round-head"><div>'
                f'<p class="ss-round-num ss-mono">Round {number}</p><h3>{esc(name)}</h3></div>'
                f'<span class="ss-round-status">{"Analysis complete" if r else "Not analysed"}</span>'
                f"</div>{body}</article>",
                unsafe_allow_html=True,
            )

st.markdown(
    '<aside class="ss-notice"><span class="ss-notice-mark">i</span><p>Each clip is treated as '
    "one round, so use one round/tactic clip per upload. Automatic round detection inside a "
    "full match is not available, and raw recordings must be prepared first (the local app "
    "does this step). Model confidence and detector counts are review aids, not accuracy "
    "guarantees.</p></aside>"
    '<footer class="ss-footer ss-mono"><span>Streamlit Community Cloud · Uploads are not kept</span>'
    "<span>TCN and GRU output grouped by round</span></footer>",
    unsafe_allow_html=True,
)
