"""Streamlit demo: one EEG trial (CSV or Excel) -> LEFT / RIGHT + confidence.

Run from this folder:  streamlit run app.py
The models and model card come from section 11 of main_pipeline.ipynb (models/deploy/).
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import bci_pipeline as bpl

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models", "deploy")
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "example_trials", "*.csv")))

st.set_page_config(page_title="BCI motor imagery demo", layout="wide")


@st.cache_resource
def get_bundle():
    return bpl.load_bundle(MODELS_DIR)


bundle = get_bundle()
card, cfg = bundle["card"], bundle["card"]["config"]
names = list(bundle["models"])

st.title("Motor imagery: LEFT or RIGHT hand?")
st.warning(card["warning"])

with st.sidebar:
    model_name = st.selectbox("Model", names, index=names.index(card["best_model"]),
                              help="Default: the model with the best mean CV AUC")
    st.caption(f"Pipeline version {card['pipeline_version']} · trained {card['created']} "
               f"on {card['training_data']['trials']} trials")
    st.subheader("Cross-validation (dev split)")
    st.dataframe(pd.DataFrame(card["cv"]).T.round(3))

st.file_uploader(f"Upload one trial: CSV or Excel with columns {', '.join(cfg['channels'])}, "
                 f"{cfg['fs'] * cfg['trial_seconds']} rows ({cfg['trial_seconds']} s at {cfg['fs']} Hz)",
                 type=["csv", "xlsx", "xls"], key="upload",
                 on_change=lambda: st.session_state.pop("example", None))
for col, path in zip(st.columns(max(len(EXAMPLES), 1)), EXAMPLES):
    if col.button(f"Example trial ({os.path.basename(path).split('_')[1].upper()})", key=path):
        st.session_state["example"] = path

source = st.session_state.get("example") or st.session_state.get("upload")
if source is None:
    st.info("Upload a trial or press an example button.")
    st.stop()
name = source if isinstance(source, str) else source.name

try:
    x = bpl.trial_to_array(bpl.read_trial(source, name), cfg)
except Exception as err:
    st.error(f"Could not read {os.path.basename(name)}: {err}")
    st.stop()

values, problems = bpl.quality_check(x, card)
if problems:
    st.error("Quality check failed: " + "; ".join(problems) +
             ". Training removed trials like this, so the prediction is not reliable.")
else:
    st.success("Quality check passed (same artifact limits as training).")

epoch = bpl.preprocess(x[None], cfg)
p_all = {n: float(bpl.predict_epochs(epoch, bundle, n)[0]) for n in names}
p_right = p_all[model_name]
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", bpl.CLASS_NAMES[int(p_right >= 0.5)])
c2.metric("Confidence", f"{max(p_right, 1 - p_right):.1%}")
c3.metric("P(RIGHT)", f"{p_right:.3f}")
st.caption(f"File: {os.path.basename(name)} · model: {model_name}")

left, right = st.columns([2, 1])
with left:
    st.subheader("Preprocessed signal (band-pass, notch, CAR, 1–9 s)")
    t = np.arange(epoch.shape[-1]) / cfg["fs"] + cfg["epoch_seconds"][0]
    fig, axes = plt.subplots(len(cfg["channels"]), 1, figsize=(10, 5), sharex=True)
    for ax, ch, sig in zip(axes, cfg["channels"], epoch[0]):
        ax.plot(t, sig, lw=0.6)
        ax.set_ylabel(ch, rotation=0, labelpad=15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Every model on this trial")
    st.dataframe(pd.DataFrame({"P(RIGHT)": p_all,
                               "prediction": {n: bpl.CLASS_NAMES[int(p >= 0.5)] for n, p in p_all.items()}}).round(3))
    st.subheader("Quality check")
    st.dataframe(pd.DataFrame({"value": values, "limit": card["artifact_limits"]}).round(3))
