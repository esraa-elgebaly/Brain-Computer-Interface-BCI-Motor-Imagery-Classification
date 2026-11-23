"""
Streamlit demo for the BCI motor-imagery pipeline (task, section 6).

Run from this folder:   streamlit run app.py

Pages (app_pages/):
  demo.py    - pick a trial, guess LEFT / RIGHT, "read the brain", see every model's vote
  models.py  - how close the three models are, and why
  data.py    - why the labels are the real problem

Shared code (loading, brain-rhythm math, animated brain) lives in app_core.py;
all preprocessing and inference go through bci_pipeline.py, the same code as training.
"""
import importlib

import streamlit as st

import app_core as core

# A server that was already running can keep an older copy of app_core in memory after the file
# changes. If it is missing what this version needs, load the current file again.
if not all(hasattr(core, a) for a in ("CSS", "compare_html", "brain_svg")):
    core = importlib.reload(core)

st.set_page_config(page_title="Motor imagery decoder", page_icon=":material/neurology:", layout="wide")

# Per-user state shared by the pages, set up in one place.
st.session_state.setdefault("rounds", [])          # one dict per revealed trial (the guess game)
st.session_state.setdefault("revealed", None)      # key of the trial whose result is shown
st.session_state.setdefault("example", None)

page = st.navigation(
    [
        st.Page("app_pages/demo.py", title="Live demo", icon=":material/neurology:", default=True),
        st.Page("app_pages/models.py", title="Model face-off", icon=":material/leaderboard:"),
        st.Page("app_pages/data.py", title="Why it's a coin flip", icon=":material/fact_check:"),
    ],
    position="top",
)
st.html(core.CSS)          # background glow + card look for every page (after the nav, keeps it on top)
page.run()
