"""StratScout tactic classifier: a Streamlit page for the cloud deployment.

Upload prepared spectator-radar clips (one tactic each) or pick a sample, and the page
runs the same prediction code as the CLI and the web app (``gru_predict.predict_clips``):
detector -> ResNet-18 features -> TCN (selected model) and optionally the Compact pooled
GRU, each averaging its three seeds.

Run locally from the repository root:

    streamlit run deploy/streamlit/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stratscout.config import TACTIC_CLASSES  # noqa: E402

BACKGROUNDS = HERE / "backgrounds"
SAMPLES = HERE / "samples"

# Broadcast layout -> (saved empty-map background, file-name hint used to pre-select it).
LAYOUTS = {
    "ESL Pro League Season 24": ("aurora.png", "esl-pro-league-season-24"),
    "Cologne Major 2026": ("furia.png", "cologne-major-2026"),
    "Cologne Major": ("spirit.png", "cologne-major"),
    "IEM Beijing 2026": ("mouz.png", "iem-bejing-2026"),
}
FROM_UPLOADS = "Build it from my uploaded clips (same match, 5+ clips)"

CLASS_NAMES = {
    "A_DIRECT": "Direct A attack",
    "B_DIRECT": "Direct B attack",
    "A_SPLIT": "A split attack",
    "B_SPLIT": "B split attack",
    "RETAKE": "CT retake",
}
MODEL_NAMES = {"tcn": "TCN (selected model)", "gru": "Compact pooled GRU"}


def guess_layout(name: str):
    """The layout whose hint appears in the file name, or None; the longest hint wins, so
    'cologne-major-2026' is not mistaken for 'cologne-major'."""
    hits = [(len(hint), layout) for layout, (_, hint) in LAYOUTS.items() if hint in name]
    return max(hits)[1] if hits else None


@st.cache_resource(show_spinner="Loading the models...")
def predictor():
    """Import once per server process; gru_predict also caches the loaded checkpoints."""
    from stratscout import gru_predict

    gru_predict.load_models("tcn")
    gru_predict.load_models("gru")
    return gru_predict


def run(paths, models, layout):
    """Each clip uses the background of the event its file name names; the chosen layout
    covers the rest. Results come back in the order of ``paths``."""
    gp = predictor()
    quiet = {"models": models, "log": lambda *a: None}
    if layout == FROM_UPLOADS:
        return gp.predict_clips(paths, background_from=paths, **quiet)
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(guess_layout(path.name) or layout, []).append(path)
    by_name = {}
    for group_layout, group in groups.items():
        background = BACKGROUNDS / LAYOUTS[group_layout][0]
        for r in gp.predict_clips(group, background=background, **quiet):
            by_name[r["clip"]] = {**r, "layout": group_layout}
    return [by_name[p.stem] for p in paths]


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="StratScout", page_icon=":material/radar:", layout="wide")
st.title("StratScout: CS2 Mirage tactic classifier")
st.write(
    "Classifies a spectator-radar clip of one tactic into one of five classes: "
    + ", ".join(f"**{CLASS_NAMES[c]}**" for c in TACTIC_CLASSES)
    + "."
)

with st.expander("How it works"):
    st.markdown(
        """
1. **Detector** (rule-based): finds T and CT icons and the bomb in every frame and turns
   them into 72 map-region numbers per 0.2 s.
2. **ResNet-18** (frozen, ImageNet): 512 image features per frame, compressed to 32 by PCA.
3. **TCN**: dilated 1D convolutions over the 104-number sequence (about 25 s receptive
   field), pooled over time, then five class probabilities. Three seeds averaged.
4. **GRU** (optional): the Compact pooled GRU on the same features, also three seeds.

Held-out test (45 clips, 2 unseen matches), macro-F1: **TCN 0.927**, GRU 0.963.
Nested cross-validation: **TCN 0.874**, GRU 0.856.
"""
    )

st.subheader("1. Choose clips")
st.caption(
    "Clips must be the cropped spectator radar (as `process-recordings` prepares it), "
    "10 fps, cut from the start to the end of one tactic. Full matches are not supported."
)
source = st.radio("Source", ["Sample clips", "Upload my own"], horizontal=True, key="source")

samples = sorted(SAMPLES.glob("*.mp4")) if SAMPLES.is_dir() else []
chosen: list[Path] = []
uploads = []
if source == "Sample clips":
    if not samples:
        st.info("No sample clips are bundled with this deployment. Upload your own instead.")
    else:
        picked = st.multiselect(
            "Held-out test clips (never seen in training)",
            [p.name for p in samples],
            default=[samples[0].name],
            key="samples",
        )
        chosen = [SAMPLES / n for n in picked]
else:
    uploads = st.file_uploader(
        "Prepared clips (.mp4)", type=["mp4"], accept_multiple_files=True, key="uploads"
    )

names = [p.name for p in chosen] + [u.name for u in uploads or []]

st.subheader("2. Settings")
col_a, col_b = st.columns(2)
with col_a:
    options = list(LAYOUTS) + ([FROM_UPLOADS] if source == "Upload my own" else [])
    unknown = [n for n in names if guess_layout(n) is None]
    layout = st.selectbox(
        "Broadcast layout for clips whose file name doesn't name the event",
        options,
        key="layout",
        disabled=bool(names) and not unknown,
        help="The detector compares each frame with an empty map of the same broadcast. "
        "Clips named like the training data (e.g. ...-cologne-major.jx.s029.mp4) are "
        "matched to their event automatically.",
    )
    if names and not unknown:
        st.caption("Every clip's event was recognised from its file name.")
with col_b:
    with_gru = st.checkbox("Also run the GRU for comparison", value=True, key="with_gru")

if not st.button("Classify", type="primary", disabled=not names, key="go"):
    st.stop()

models = ("tcn", "gru") if with_gru else ("tcn",)
with tempfile.TemporaryDirectory() as tmp:
    paths = list(chosen)
    for u in uploads or []:
        path = Path(tmp) / Path(u.name).name
        if path.exists() or path.name in {p.name for p in chosen}:
            st.error(f"Two clips are called {path.name}. Rename one and upload again.")
            st.stop()
        path.write_bytes(u.getbuffer())
        paths.append(path)
    with st.spinner(f"Classifying {len(paths)} clip(s). The first run takes about a minute..."):
        try:
            results = run(paths, models, layout)
        except Exception as error:  # Shown to the user rather than as a stack trace.
            st.error(f"Could not classify the clips: {error}")
            st.stop()

st.subheader("3. Results")

summary = []
for r in results:
    row = {"Clip": r["clip"], "Layout used": r.get("layout", "From uploads")}
    for kind in models:
        p = r["predictions"][kind]
        row[MODEL_NAMES[kind]] = CLASS_NAMES[p["tactic"]]
        row[f"{kind.upper()} confidence"] = f"{p['confidence']:.0%}" + (
            " ⚠" if p["uncertain"] else ""
        )
    summary.append(row)
st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")

if len(results) > 1:
    counts = pd.Series(
        [CLASS_NAMES[r["predictions"]["tcn"]["tactic"]] for r in results]
    ).value_counts()
    st.markdown("**Tactics found (TCN)**")
    st.bar_chart(counts.reindex([CLASS_NAMES[c] for c in TACTIC_CLASSES], fill_value=0))

for r in results:
    with st.container(border=True):
        st.markdown(f"**{r['clip']}**")
        cols = st.columns(len(models))
        for col, kind in zip(cols, models):
            p = r["predictions"][kind]
            with col:
                st.metric(
                    MODEL_NAMES[kind],
                    CLASS_NAMES[p["tactic"]],
                    f"{p['confidence']:.0%} confident",
                    delta_color="off",
                )
                if p["uncertain"]:
                    st.caption(f"Low confidence. Runner-up: {CLASS_NAMES[p['runner_up']]}.")
                probs = pd.DataFrame(
                    {"Probability": [p["probabilities"][c] for c in TACTIC_CLASSES]},
                    index=[CLASS_NAMES[c] for c in TACTIC_CLASSES],
                )
                st.bar_chart(probs, horizontal=True, height=220)
        for warning in r["warnings"]:
            st.warning(warning)
