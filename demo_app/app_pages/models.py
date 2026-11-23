"""Model face-off: where each model ended up, how close they are, and why."""
import altair as alt
import pandas as pd
import streamlit as st

import app_core as core

bundle = core.get_bundle()
card = bundle["card"]
res = core.cv_results()
names = [n for n in res.index if n in core.MODEL_INFO]
USABLE = 0.70           # common rule of thumb for a usable 2-class BCI

st.title("Model face-off")
st.caption("Three very different models, trained and tested the same way: 5 folds split by recording session, "
           "so a model never sees a test session during training.")

# ---------------------------------------------------------------- headline
best = res.loc[names, "AUC"].idxmax()
spread = res.loc[names, "AUC"].max() - res.loc[names, "AUC"].min()
noise = res.loc[names, "AUC std"].mean()
with st.container(horizontal=True):
    st.metric("Best AUC", f"{res.loc[best, 'AUC']:.3f}", f"{res.loc[best, 'AUC'] - 0.5:+.3f} vs chance",
              delta_color="off", delta_arrow="off", border=True, help=best)
    st.metric("Gap between best and worst", f"{spread:.3f}", border=True)
    st.metric("Fold-to-fold noise (mean std)", f"±{noise:.3f}", border=True)
    st.metric("Needed for a usable BCI", f"≈ {USABLE:.2f}", border=True,
              help="A common rule of thumb for 2-class BCIs is about 70% accuracy.")

if spread < noise:
    st.info(f"The models differ by **{spread:.3f} AUC**, less than the **±{noise:.3f}** they move between folds. "
            "Statistically they are tied, and all of them are tied with a coin.", icon=":material/balance:")

# ---------------------------------------------------------------- AUC with error bars
df = res.loc[names].reset_index().rename(columns={"Model": "model"})
df["lo"], df["hi"] = df["AUC"] - df["AUC std"], df["AUC"] + df["AUC std"]
base = alt.Chart(df).encode(y=alt.Y("model:N", sort=names, title=None))
bars = base.mark_rule(strokeWidth=3, color="#8b7cf6").encode(
    x=alt.X("lo:Q", scale=alt.Scale(domain=[0.4, 0.8]), title="Cross-validated AUC (mean ± std over 5 folds)"),
    x2="hi:Q")
dots = base.mark_point(filled=True, size=140, color="#8b7cf6").encode(
    x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.4, 0.8])), tooltip=["model", alt.Tooltip("AUC:Q", format=".3f"), alt.Tooltip("AUC std:Q", format=".3f")])
refs = pd.DataFrame({"x": [0.5, USABLE], "what": ["chance (coin flip)", "usable BCI"]})
rules = alt.Chart(refs).mark_rule(strokeDash=[5, 4], color="#a9b0cc").encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[0.4, 0.8])), tooltip=["what"])
rule_txt = alt.Chart(refs).mark_text(align="left", dx=5, dy=-8, color="#a9b0cc").encode(
    x="x:Q", y=alt.value(0), text="what")
with st.container(key="card_models_1"):
    st.markdown("**How far each model got**")
    st.altair_chart((rules + rule_txt + bars + dots).properties(height=200), width="stretch")
    st.caption("Every error bar crosses the chance line, and none comes near the usable zone.")

# ---------------------------------------------------------------- one card per model
cols = st.columns(len(names), gap="medium")
for col, n in zip(cols, names):
    info, r = core.MODEL_INFO[n], res.loc[n]
    train, test = r["Train acc"], r["Accuracy"]
    if train < 0.55:
        why = (f"It cannot even fit its **training** trials (train accuracy {train:.0%}). "
               "There is no pattern that holds across trials, so there is nothing to learn.")
    else:
        why = (f"It does learn the training trials (train {train:.0%}) but that does not carry over to new "
               f"sessions (test {test:.0%}). It memorised noise, which is what happens when labels are random.")
    with col, st.container(key=f"card_model_{names.index(n)}", height="stretch"):
        st.subheader(f"{info['icon']} {n}", anchor=False)
        st.badge(info["kind"], color="violet")
        st.caption(info["how"])
        with st.container(horizontal=True):
            st.metric("AUC", f"{r['AUC']:.3f}")
            st.metric("Train acc", f"{train:.0%}")
            st.metric("Test acc", f"{test:.0%}")
        st.markdown("**Why it ended here**")
        st.markdown(why)

# ---------------------------------------------------------------- train vs test
long = df.melt("model", value_vars=["Train acc", "Accuracy"], var_name="set", value_name="accuracy")
long["set"] = long["set"].map({"Train acc": "Training trials", "Accuracy": "Unseen sessions"})
with st.container(key="card_models_2"):
    st.markdown("**Training vs unseen sessions**")
    pts = alt.Chart(long).mark_point(filled=True, size=130).encode(
        x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0.4, 0.7]), axis=alt.Axis(format="%"), title="Accuracy"),
        y=alt.Y("model:N", sort=names, title=None),
        color=alt.Color("set:N", title=None, scale=alt.Scale(range=["#8b7cf6", "#199e70"])),
        tooltip=["model", "set", alt.Tooltip("accuracy:Q", format=".1%")],
    )
    link = alt.Chart(df).mark_rule(color="#4a5683", strokeWidth=2).encode(
        x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0.4, 0.7])), x2="Train acc:Q", y=alt.Y("model:N", sort=names))
    chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[5, 4], color="#a9b0cc").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.4, 0.7])))
    st.altair_chart((chance + link + pts).properties(height=180), width="stretch")
    st.caption("With real labels both dots move right together. Here the test dot stays on the chance line "
               "whatever the model does with its training trials.")
