"""Why it's a coin flip: the evidence that labels.csv is not linked to the right files."""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_core as core

bundle = core.get_bundle()
cfg = bundle["config"]
CH, FS = cfg["channels"], cfg["fs"]
dup = core.duplicates()
index = core.dataset_index()
ref = core.training_reference(FS, CH)
n_labels = len(pd.read_csv(os.path.join(core.HERE, cfg["data"]["labels_csv"])))
n_files = len(index) if index is not None else 0

st.title("Why every model is a coin flip")
st.caption("The models are fine. The labels are not linked to the right files, so there is nothing to learn.")

# ---------------------------------------------------------------- 3 pieces of evidence
same, total = int(dup.same_label.sum()), len(dup)
with st.container(horizontal=True):
    st.metric("Labels vs trial files", f"{n_labels:,} vs {n_files:,}", border=True,
              help="labels.csv has no file names, and 240 more rows than there are files.")
    st.metric("Identical copies with the same label", f"{same} of {total}", f"{same / total:.0%} (should be 100%)",
              delta_color="red" if same < total else "green", delta_arrow="off", border=True)
    if ref is not None:
        l, r = ref.loc[ref.label == "LEFT", "mu"], ref.loc[ref.label == "RIGHT", "mu"]
        d = (r.mean() - l.mean()) / np.sqrt((l.var() + r.var()) / 2)
        st.metric("LEFT vs RIGHT difference in mu (Cohen d)", f"{d:+.3f}", "0.2 = small effect",
                  delta_color="off", delta_arrow="off", border=True)

c1, c2 = st.columns([3, 2], gap="medium")

with c1, st.container(key="card_data_1", height="stretch"):
    st.markdown("**The brain signal does not follow the labels**")
    if ref is not None:
        b = st.segmented_control("Band", list(core.BANDS), default="mu", key="data_band") or "mu"
        lo, hi = ref[b].quantile([0.01, 0.99])
        dens = alt.Chart(ref).transform_filter((alt.datum[b] >= lo) & (alt.datum[b] <= hi)).transform_density(
            b, groupby=["label"], as_=["lateralization", "density"], extent=[lo, hi]
        ).mark_area(opacity=0.35, line={"strokeWidth": 2}).encode(
            x=alt.X("lateralization:Q", title=f"{b} lateralization (C4 − C3) / (C4 + C3)"),
            y=alt.Y("density:Q", title="Share of trials", axis=alt.Axis(labels=False), stack=None),
            color=alt.Color("label:N", title="Given label",
                            scale=alt.Scale(domain=list(core.CLASS_COLORS), range=list(core.CLASS_COLORS.values()))),
            tooltip=["label:N", alt.Tooltip("lateralization:Q", format="+.3f")],
        )
        st.altair_chart(dens.properties(height=280), width="stretch")
        st.caption(f"All {len(ref):,} trials kept after cleaning. Real LEFT / RIGHT imagery shifts these curves "
                   "to opposite sides; here they are the same curve.")

with c2, st.container(key="card_data_2", height="stretch"):
    st.markdown("**Same file, different label**")
    nums = dup.files.str.findall(r"(\d+)\.csv")
    show = pd.DataFrame({"Trial files": nums.map(lambda n: f"#{n[0]} = #{n[1]}"), "Labels": dup.labels,
                         "same_label": dup.same_label}).sort_values("same_label")
    st.dataframe(show, hide_index=True, height=280, width="stretch",
                 column_config={"same_label": st.column_config.CheckboxColumn("Same?", width="small")})
    st.caption("Byte-identical recordings can only have one true label. "
               f"{total - same} of {total} pairs disagree, about what random labels give.")

# ---------------------------------------------------------------- what we tried
with st.container(key="card_data_3"):
    st.markdown("**Can we re-link the labels ourselves?** We tried, and no.")
    st.markdown(
        "The `Time` column shows the files come in **240 blocks of 9 back-to-back 10 s trials**, while "
        "2400 labels = **240 × 10**. So one trial per block was probably lost. We scored every way of "
        "re-linking with a band-power classifier (folds split by block):"
    )
    tried = pd.DataFrame({
        "Way of linking label → file": ["Row i → file i (what the pipeline uses)",
                                        "Drop one label per block of 10 (each of the 10 positions)",
                                        "Order files by recording time",
                                        "Shuffled labels (reference)"],
        "Accuracy": ["0.467", "0.47 – 0.52", "0.48 – 0.52", "0.49 (max 0.52)"],
    })
    st.dataframe(tried, hide_index=True, width="stretch")
    st.caption("None beats shuffled labels, so the right pairing cannot be recovered from the data.")
