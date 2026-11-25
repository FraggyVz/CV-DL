"""StratScout cloud demo: the local web app's workflow on Streamlit Community Cloud.

It mirrors the local React + FastAPI app section by section (recording, analysis flow,
match rounds, video review, classification) and runs the local app's own analysis code,
``stratscout.web_analysis``: the same ffmpeg preparation, detector, ResNet features, TCN
and GRU ensembles and stage-by-stage pipeline record. The classification section is the
local app's React results view itself (``frontend/src/viewer.tsx``), built into
``results_component/`` and embedded as a Streamlit component, so the tabs, round picker,
prediction cards and pipeline graphs are identical.

Differences from the local app: the cloud has no recording crop profile, so only
already-prepared clips (494x500 radar at 10 fps) are accepted, and the match lives in the
browser session (nothing is kept after the tab closes).

Run locally from the repository root (needs ffmpeg on PATH):

    streamlit run deploy/streamlit/streamlit_app.py
"""

from __future__ import annotations

import html
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stratscout import web_analysis  # noqa: E402

SAMPLES = HERE / "samples"
MIN_CONFIDENCE = 0.6  # the local app's default
results_view = components.declare_component(
    "stratscout_results", path=str(HERE / "results_component")
)

CSS = """
<style>
html { font-size: 17.6px; } /* 10% above Streamlit's 16px; the title is fixed in px */
:root { --muted:#8d958e; --accent:#c8f169; --panel:#171b18; --line:#292f2b; --red:#f27b6e; }
.stApp { background: radial-gradient(ellipse at 82% 0%, rgb(200 241 105 / 8%), transparent 36rem), #101311; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 100% !important;
  padding-left: 16px !important; padding-right: 16px !important; padding-top: 2.5rem !important; }
.ss-mono { font-family: Consolas, "Courier New", monospace; letter-spacing: .06em; }
.ss-topbar { display:flex; align-items:center; justify-content:space-between; gap:20px;
  padding: 0 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 8px; flex-wrap: wrap; }
.ss-brand { display:flex; align-items:center; gap:10px; font-weight:600; font-size: 17.6px; color:#e7e9e5; }
.ss-mark { width:28px; height:28px; display:grid; place-items:center; border-radius:7px;
  background: var(--accent); color:#101311; font-weight:800; }
.ss-status { display:flex; align-items:center; gap:8px; color:#a7afa8; font-size: 13.2px; }
.ss-dot { width:8px; height:8px; border-radius:50%; background: var(--accent); }
.ss-badge { margin-left:6px; font-size: 11px; color:#a7afa8; }
.ss-eyebrow { color: var(--accent); font-size: 12.1px; margin: 0 0 8px; }
.ss-hero h1 { font-size: clamp(35px, 6vw, 64px); font-weight: 520; letter-spacing: -0.06em;
  line-height: 1; margin: 34px 0 16px; padding:0; }
.ss-hero p.copy { color:#a7afa8; max-width: 640px; }
.ss-section { display:flex; justify-content:space-between; align-items:flex-end; gap:12px;
  margin: 34px 0 12px; flex-wrap: wrap; }
.ss-section h2 { margin:0; padding:0; font-size:1.35rem; }
.ss-quiet { color:#737d75; font-size: 11px; }
.ss-pipeline { list-style:none; padding:0 !important; margin: 6px 0 8px !important; display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; }
.ss-pipeline li { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background: var(--panel);
  display:flex; flex-direction:column; gap:2px; }
.ss-pipeline li.complete { border-color: rgb(200 241 105 / 35%); }
.ss-pipeline li.current { border-color: #e9bd62; }
.ss-num { font-family: Consolas, monospace; color: var(--accent); font-size: 13.2px; }
.ss-ptitle { font-weight:600; font-size: 15.4px; }
.ss-pstate { color: var(--muted); font-size: 13.2px; }
.ss-note { color: var(--muted); font-size: 14.3px; margin: 4px 0; }
.ss-round-num { color: var(--accent); font-size: 12.1px; }
.ss-round-name { font-size: 14.3px; word-break: break-all; }
.ss-round-status { color: var(--muted); font-size: 13.2px; }
.ss-pill { padding: 2px 6px; border-radius: 4px; background:#263445; color:#a9c7ef; font-size: 11px;
  text-transform: uppercase; }
.ss-pill.retake { background:#402725; color:#f1a49b; }
.ss-notice { display:flex; gap:11px; margin: 22px 0 0; padding: 14px 16px;
  border:1px solid rgb(200 241 105 / 16%); border-radius:9px; background: rgb(200 241 105 / 4%); }
.ss-notice-mark { width:18px; height:18px; flex:0 0 auto; display:grid; place-items:center;
  border:1px solid #879d56; border-radius:50%; color: var(--accent); font-family: Consolas, monospace; font-size: 12.1px; }
.ss-notice p { margin:0; color:#b7c0b3; font-size: 14.3px; line-height: 1.55; }
.ss-footer { display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; padding: 22px 0 8px;
  color:#737d75; font-size: 11px; text-transform: uppercase; }
/* Buttons in the local app's style: dark, with a lime accent for the main action. */
.stButton button, .stPopover button, [data-testid="stBaseButton-secondary"] {
  background: #191e1a; border: 1px solid #424b43; color: #e7e9e5; border-radius: 6px; }
.stButton button:hover, .stPopover button:hover { border-color: #73835d; color: #e7e9e5; background: #1d231e; }
[data-testid="stBaseButton-primary"] {
  background: rgb(200 241 105 / 9%) !important; border: 1px solid #536348 !important;
  color: var(--accent) !important; font-weight: 600; }
[data-testid="stBaseButton-primary"]:hover { background: rgb(200 241 105 / 15%) !important; }
.stButton button:disabled { opacity: .5; }
.st-key-reset_open button, .st-key-reset_confirm button { background: #2a1b19 !important; border-color: #70403b !important; color: #f29a8f !important; }
.st-key-reset_open button:hover, .st-key-reset_confirm button:hover { background: #38201d !important; border-color: #a45b51 !important; }
.st-key-reset_row { align-items: flex-end !important; }
.st-key-reset_row .stMarkdown { text-align: right; }
.st-key-reset_row p.ss-quiet { font-size: 11px !important; }
[data-testid="stVideo"] video, video { width: 100%; max-height: 520px; object-fit: contain; border-radius: 8px;
  border: 1px solid var(--line); background: #000; }
@media (max-width: 640px) { .ss-pipeline { grid-template-columns: repeat(2, minmax(0,1fr)); } }
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


# --------------------------------------------------------------------------- state


def workspace() -> Path:
    """This browser session's upload folder, laid out as the local app's data/web/uploads."""
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="stratscout-")
        st.session_state.rounds = []
        st.session_state.selected = ""
        st.session_state.match_name = "Match 1"
    return Path(st.session_state.workdir)


def rounds() -> list[dict]:
    return st.session_state.rounds


def selected_round() -> dict | None:
    return next((r for r in rounds() if r["id"] == st.session_state.selected), None) or (
        rounds()[0] if rounds() else None
    )


def browser_video(video_id: str) -> Path:
    """An H.264 copy the browser can play and seek, made as the local app makes it."""
    source = workspace() / f"{video_id}.mp4"
    try:
        from stratscout.video_preparation import browser_compatible_video

        return Path(browser_compatible_video(source, workspace() / f"{video_id}.browser.mp4"))
    except Exception:  # Without ffmpeg the original still plays in most browsers.
        return source


def add_round(name: str, data: bytes, phase: str) -> None:
    video_id = str(uuid.uuid4())
    (workspace() / f"{video_id}.mp4").write_bytes(data)
    browser_video(video_id)
    rounds().append(
        {
            "id": video_id,
            "filename": Path(name).name,
            "round_number": len(rounds()) + 1,
            "phase": phase,
            "status": "idle",
            "error": None,
            "analysis_id": None,
            "results": [],
            "pipelines": {},
        }
    )
    st.session_state.selected = video_id


def remove_round(video_id: str) -> None:
    upload_dir = workspace()
    for path in upload_dir.glob(f"{video_id}*"):
        path.unlink(missing_ok=True)
    shutil.rmtree(upload_dir / "analysis" / video_id, ignore_errors=True)
    st.session_state.rounds = [r for r in rounds() if r["id"] != video_id]
    for number, r in enumerate(rounds(), 1):  # Later rounds are renumbered, as locally.
        r["round_number"] = number


def reset_workspace() -> None:
    """The local app's Reset workspace: remove every round and its analysis."""
    for r in list(rounds()):
        remove_round(r["id"])
    st.session_state.match_name = "Match 1"
    st.session_state.match_count = 1
    st.session_state.selected = ""
    st.session_state.confirm_reset = False
    clear_uploader()


def new_match() -> None:
    """Start the next match with an empty round list (one match per session)."""
    for r in list(rounds()):
        remove_round(r["id"])
    st.session_state.match_count = st.session_state.get("match_count", 1) + 1
    st.session_state.match_name = f"Match {st.session_state.match_count}"
    st.session_state.selected = ""
    clear_uploader()


def ask_reset(value: bool) -> None:
    st.session_state.confirm_reset = value


def add_uploads() -> None:
    """Each uploaded file becomes the next round as soon as it arrives, as locally.

    The uploader is never replaced here: several files may still be uploading into it,
    and swapping the widget mid-upload cuts them off ("AxiosError: Network Error").
    Files are matched by their upload id, so each one becomes exactly one round."""
    added = st.session_state.setdefault("added_uploads", set())
    key = f"uploader-{st.session_state.get('upload_generation', 0)}"
    for f in st.session_state.get(key) or []:
        if f.file_id not in added:
            added.add(f.file_id)
            add_round(f.name, f.getvalue(), st.session_state.upload_phase)


def clear_uploader() -> None:
    """A fresh, empty uploader; only called when no upload can be in progress."""
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    st.session_state.added_uploads = set()


def add_sample(name: str) -> None:
    add_round(name, (SAMPLES / name).read_bytes(), st.session_state.upload_phase)


# --------------------------------------------------------------------------- analysis


@st.cache_resource(show_spinner="Loading the saved TCN and GRU ensembles...")
def warm_models() -> bool:
    from stratscout import gru_predict

    gru_predict.load_models("tcn", device="cpu")
    gru_predict.load_models("gru", device="cpu")
    return True


def analyse(r: dict, progress, playback_speed: int = 2) -> None:
    """web_analysis.queue_analysis + run_analysis, as the local app's analysis endpoint
    runs them, with the whole clip as the round's one segment."""
    from stratscout.media import probe

    warm_models()
    upload_dir = workspace()
    video_id = r["id"]
    duration = probe(upload_dir / f"{video_id}.mp4")["duration_seconds"]
    segment = {
        "id": str(uuid.uuid4()),
        "start_s": 0.0,
        "end_s": float(duration),
        "round_number": r["round_number"],
        "phase": r["phase"],
    }
    state = web_analysis.queue_analysis(upload_dir, video_id)
    failure: list[BaseException] = []

    def job():
        try:
            web_analysis.run_analysis(
                upload_dir,
                video_id,
                state["analysis_id"],
                [segment],
                playback_speed=playback_speed,
                prepared_clip=True,
                min_confidence=MIN_CONFIDENCE,
            )
        except BaseException as exc:  # The state file already records the failure.
            failure.append(exc)

    worker = threading.Thread(target=job, daemon=True)
    worker.start()
    while worker.is_alive():
        current = web_analysis.get_analysis(upload_dir, video_id)
        progress.progress(
            min(int(current.get("progress", 0)), 100) / 100,
            text=f"Round {r['round_number']}: {current.get('message', '')}",
        )
        time.sleep(0.4)
    worker.join()
    final = web_analysis.get_analysis(upload_dir, video_id)
    r.update(
        status="complete" if final["status"] == "complete" else "failed",
        error=final.get("error"),
        analysis_id=final.get("analysis_id"),
        results=final.get("results", []),
        pipelines={},
    )
    for result in r["results"]:
        if result.get("pipeline"):
            r["pipelines"][result["segment_id"]] = web_analysis.get_pipeline(
                upload_dir, video_id, result["segment_id"]
            )


def pipeline_steps(r: dict | None) -> str:
    """The local app's four pipeline cards for the selected round."""
    status = r["status"] if r else None
    steps = [
        ("Upload", "Ready" if r else "Waiting", bool(r), not r),
        (
            "Prepare & detect",
            {"complete": "Complete", "failed": "Failed"}.get(
                status, "Ready to run" if r else "Waiting"
            ),
            status == "complete",
            False,
        ),
        ("Assign round", f"Round {r['round_number']}" if r else "Automatic", bool(r), False),
        (
            "Classify",
            {"complete": "Complete", "failed": "Failed"}.get(status, "Waiting"),
            status == "complete",
            False,
        ),
    ]
    items = []
    for i, (title, state, complete, current) in enumerate(steps, 1):
        cls = "complete" if complete else ("current" if current else "")
        items.append(
            f'<li class="{cls}"><span class="ss-num">{"✓" if complete else i}</span>'
            f'<span class="ss-ptitle">{title}</span><span class="ss-pstate">{esc(state)}</span></li>'
        )
    return f'<ol class="ss-pipeline">{"".join(items)}</ol>'


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="StratScout", page_icon=":material/radar:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
workspace()

st.markdown(
    '<header class="ss-topbar"><span class="ss-brand"><span class="ss-mark">S</span>StratScout</span>'
    '<span class="ss-status"><span class="ss-dot"></span>Cloud service ready'
    '<span class="ss-badge ss-mono">CLOUD WORKSPACE</span></span></header>'
    '<section class="ss-hero"><p class="ss-eyebrow ss-mono">CS2 · MIRAGE · MATCH WORKSPACE</p>'
    "<h1>Review the round.<br>Read the strategy.</h1>"
    '<p class="copy">Upload one round clip at a time, keep each analysis in your match list, '
    "and compare the TCN and GRU predictions.</p></section>",
    unsafe_allow_html=True,
)

# 01 / RECORDING ------------------------------------------------------------------
heading, header_right = st.columns([3, 1], vertical_alignment="bottom")
with heading:
    st.markdown(
        '<div class="ss-section"><div><p class="ss-eyebrow ss-mono">01 / RECORDING</p>'
        "<h2>Build your match rounds</h2></div></div>",
        unsafe_allow_html=True,
    )
with header_right, st.container(key="reset_row"):
    # Reset workspace sits above the section's quiet label, as an inline confirmation.
    if not st.session_state.get("confirm_reset"):
        with st.container(key="match_actions", horizontal=True, horizontal_alignment="right"):
            st.button(
                "New match",
                key="new_match",
                disabled=not rounds(),
                on_click=new_match,
                help="Start the next match (Match 2, Match 3…) with an empty round list.",
            )
            st.button(
                "Reset workspace",
                key="reset_open",
                disabled=not rounds(),
                on_click=ask_reset,
                args=(True,),
            )
    else:
        st.markdown(
            '<p class="ss-note">Remove every uploaded recording and saved analysis in this '
            "session?</p>",
            unsafe_allow_html=True,
        )
        with st.container(key="reset_confirm_box", horizontal=True, horizontal_alignment="right"):
            st.button("Cancel", key="reset_cancel", on_click=ask_reset, args=(False,))
            st.button("Remove everything", key="reset_confirm", on_click=reset_workspace)
    st.markdown(
        '<p class="ss-quiet ss-mono" style="margin:0 0 14px">MP4 · KEPT FOR THIS SESSION</p>',
        unsafe_allow_html=True,
    )
col_match, col_phase = st.columns([2, 1])
with col_match:
    st.text_input("Current match", key="match_name")
with col_phase:
    phase = st.selectbox(
        "Clip type", ["attack", "retake"], format_func=str.title, key="upload_phase"
    )

upload_tab, sample_tab = st.tabs(["Upload round clips", "Use sample clips"])
with upload_tab:
    st.file_uploader(
        "Prepared round clips (.mp4). Each file becomes the next round when it is added.",
        type=["mp4"],
        accept_multiple_files=True,
        key=f"uploader-{st.session_state.get('upload_generation', 0)}",
        on_change=add_uploads,
    )
with sample_tab:
    samples = sorted(SAMPLES.glob("*.mp4")) if SAMPLES.is_dir() else []
    st.markdown(
        '<p class="ss-note">Held-out test clips, never seen in training. Click one to add it '
        "as the next round.</p>",
        unsafe_allow_html=True,
    )
    for sample in samples:
        st.button(
            f"+ {sample.stem}",
            key=f"sample-{sample.name}",
            on_click=add_sample,
            args=(sample.name,),
        )

# 02 / ANALYSIS FLOW --------------------------------------------------------------
current = selected_round()
section("02 / ANALYSIS FLOW", "From round clip to model output", "PIPELINE STATUS")
st.markdown(pipeline_steps(current), unsafe_allow_html=True)
st.markdown(
    '<p class="ss-note">Upload one MP4 per round, then analyse it. Round numbers advance '
    "automatically within the match; full-match round detection is not supported.</p>",
    unsafe_allow_html=True,
)

# 03 / MATCH ROUNDS ---------------------------------------------------------------
section("03 / MATCH ROUNDS", f"{st.session_state.match_name} rounds", f"{len(rounds())} ROUNDS")
pending = [r for r in rounds() if r["status"] != "complete"]
analyse_all = st.button(
    "Analyse all rounds", type="primary", disabled=not pending, key="analyse_all"
)
all_progress = st.empty()
if analyse_all:
    bar = all_progress.progress(0.0, text="Queued.")
    for r in pending:
        analyse(r, bar, int(st.session_state.get("playback_speed", "2")))
    all_progress.empty()
    st.rerun()
if not rounds():
    st.markdown(
        '<p class="ss-note">No rounds yet. Upload a round clip or add a sample above.</p>',
        unsafe_allow_html=True,
    )
for r in rounds():
    with st.container(border=True):
        info, review, remove = st.columns([5, 1.3, 1], vertical_alignment="center")
        status = {"complete": "Analysis complete", "failed": "Needs attention"}.get(
            r["status"], "Not analysed"
        )
        info.markdown(
            f'<span class="ss-round-num ss-mono">ROUND {r["round_number"]}</span> '
            f'<span class="ss-pill {r["phase"]}">{r["phase"]}</span><br>'
            f'<span class="ss-round-name">{esc(r["filename"])}</span> · '
            f'<span class="ss-round-status">{status}</span>',
            unsafe_allow_html=True,
        )
        reviewing = current is not None and r["id"] == current["id"]
        if review.button(
            "Reviewing" if reviewing else "Review",
            key=f"review-{r['id']}",
            disabled=reviewing,
            help="Show this round in 04 / Video review",
        ):
            st.session_state.selected = r["id"]
            st.rerun()
        if remove.button(
            "Remove", key=f"remove-{r['id']}", help=f"Remove Round {r['round_number']}"
        ):
            remove_round(r["id"])
            st.rerun()
st.markdown(
    '<p class="ss-note">Analyse all runs every round that has not been analysed yet.</p>',
    unsafe_allow_html=True,
)

# 04 / VIDEO REVIEW ---------------------------------------------------------------
if current:
    section(
        "04 / VIDEO REVIEW",
        f"Round {current['round_number']} · {current['filename']}",
        current["phase"].upper(),
    )
    video_col, panel_col = st.columns([5, 6])
    with video_col:
        st.video(str(browser_video(current["id"])), loop=True)
    with panel_col:
        st.markdown(
            f'<p class="ss-eyebrow ss-mono">ROUND {current["round_number"]}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("**Clip boundaries are automatic**")
        st.markdown(
            '<p class="ss-note">New round clips are analysed from the first frame to the last.</p>',
            unsafe_allow_html=True,
        )
        st.selectbox(
            "Demo replay speed",
            ["2", "1"],
            format_func={"2": "2× (recording guide default)", "1": "1× (normal speed)"}.get,
            key="playback_speed",
            help="The speed the demo was replayed at when it was recorded. It is passed to "
            "the analysis as in the local app; prepared clips are already at normal speed, so "
            "it only changes the result for full-screen recordings.",
        )
        st.checkbox(
            "Already a prepared minimap clip (494×500, 10 fps)",
            value=True,
            disabled=True,
            key=f"prepared-{current['id']}",
            help="The cloud workspace accepts prepared clips only. Raw recordings need the "
            "local app's crop profile.",
        )
        label = (
            "Analyse this round again"
            if current["status"] == "complete"
            else f"Analyse Round {current['round_number']}"
        )
        run = st.button(label, key=f"analyse-{current['id']}", type="primary")
        round_progress = st.empty()
        if run:
            bar = round_progress.progress(0.0, text="Queued.")
            analyse(current, bar, int(st.session_state.get("playback_speed", "2")))
            round_progress.empty()
            st.rerun()
        if current["status"] == "failed" and current.get("error"):
            st.error(current["error"])

# 05 / CLASSIFICATION -------------------------------------------------------------
results_view(
    rounds=[
        {
            key: r[key]
            for key in (
                "id",
                "round_number",
                "filename",
                "status",
                "error",
                "analysis_id",
                "results",
                "pipelines",
            )
        }
        for r in rounds()
    ],
    key="results",
    default=None,
)

st.markdown(
    '<aside class="ss-notice"><span class="ss-notice-mark">i</span><p>Each new clip is '
    "treated as one round, so use one round/tactic clip per upload. Automatic round "
    "detection inside a full match is not available. Model confidence and detector counts "
    "are review aids, not accuracy guarantees.</p></aside>"
    '<footer class="ss-footer ss-mono"><span>Streamlit Community Cloud · Recordings are '
    "kept only for this session</span><span>TCN and GRU output grouped by round</span></footer>",
    unsafe_allow_html=True,
)
