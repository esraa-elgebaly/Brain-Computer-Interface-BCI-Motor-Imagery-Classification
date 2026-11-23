"""Live demo: pick a trial -> guess -> read the brain -> every model's vote + the brain rhythms behind it.

Direct link to a trial (handy for sharing or slides):  ?trial=42&guess=LEFT&reveal=1
"""
import glob
import io
import os
import random
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_core as core
import bci_pipeline as bp

try:
    bundle = core.get_bundle()
except FileNotFoundError:
    st.error("No trained models found. Run BCI_MI_Pipeline.ipynb first.", icon=":material/error:")
    st.stop()

card, cfg = bundle["card"], bundle["config"]
CH, FS = cfg["channels"], cfg["fs"]
names = list(bundle["models"])
index = core.dataset_index()
examples = sorted(glob.glob(os.path.join(core.HERE, "example_trials", "*")))
ss = st.session_state

# ---------------------------------------------------------------- direct link (?trial=42&guess=LEFT&reveal=1)
if "query_applied" not in ss:
    ss.query_applied = True
    qp = st.query_params
    if index is not None and qp.get("trial", "").isdigit():
        ss.source_kind, ss.trial_no = "Dataset", int(qp["trial"])
        ss.query_guess = qp.get("guess", "").upper() or None
        ss.query_reveal = qp.get("reveal") == "1"

# ================================================================ header
head, how = st.columns([5, 1], vertical_alignment="bottom")
with head:
    st.html('<div class="bci-eyebrow">Motor imagery · 4-channel EEG</div>')
    st.title("Which hand are you imagining?", anchor=False)
    st.caption("Pick a 10-second EEG trial, make your own guess, then let three models read the brain.")
with how, st.container(horizontal_alignment="right"):
    with st.popover("How it works", icon=":material/help:"):
        st.markdown(
            "**The brain picture** is seen from above, nose up.\n\n"
            "- **C3** (left side) controls the **right hand**, **C4** (right side) the **left hand**.\n"
            "- Imagining a movement makes the mu rhythm (8–12 Hz) **quieter** over the opposite side. "
            "The electrode with the quieter rhythm glows more. This comes straight from the signal.\n"
            "- **The moving hand** is the models' majority vote. With no majority you see **?**.\n"
            "- **Fz** and **Cz** do not move a hand, but the models use them as a reference and to remove noise.\n\n"
            "A model with P(RIGHT) between 0.45 and 0.55 is **undecided**."
        )


def surprise():
    ss.trial_no = random.choice(index.number.tolist())


# ================================================================ controls
source = name = given = info = None
upload_note = None
with st.container(key="card_controls"):
    c1, c2, c3 = st.columns([5, 3, 3], gap="large")
    with c1:
        st.html(core.step_title(1, "Pick a trial"))
        kinds = (["Dataset"] if index is not None else []) + ["Examples", "Upload"]
        ss.setdefault("source_kind", kinds[0])
        kind = st.segmented_control("Trial source", kinds, key="source_kind",
                                    label_visibility="collapsed") or kinds[0]
        if kind == "Dataset":
            ss.setdefault("trial_no", 1)
            with st.container(horizontal=True, vertical_alignment="bottom"):
                st.number_input("Trial number", min_value=int(index.number.min()),
                                max_value=int(index.number.max()), step=1, key="trial_no")
                st.button("Surprise me", icon=":material/casino:", on_click=surprise)
            row = index[index.number == ss.trial_no]
            if len(row):
                info = row.iloc[0]
                name, source, given = row.index[0], info.path, info.label
        elif kind == "Examples":
            with st.container(horizontal=True):
                for path in examples:
                    side = "LEFT" if "_left_" in os.path.basename(path) else "RIGHT"
                    if st.button(f"Example ({side})", icon=":material/play_arrow:", key=path):
                        ss.example = path
            if ss.example:
                source = ss.example
                name = os.path.basename(source)
                given = "LEFT" if "_left_" in name else "RIGHT"
        else:
            upload = st.file_uploader("EEG trial (CSV or Excel)", type=["csv", "xlsx"],
                                      help=f"Columns: Time, {', '.join(CH)}. At least "
                                           f"{cfg['fs'] * cfg['trial_seconds']} rows ({cfg['trial_seconds']} s at {cfg['fs']} Hz).")
            if upload is not None:
                # Read the bytes every run: the same UploadedFile comes back on each rerun, and reading
                # it directly a second time would start at the end of the file and find nothing.
                data = upload.getvalue()
                source, name = io.BytesIO(data), upload.name
                upload_note = f"{upload.name} · {len(data) / 1024:,.0f} KB"
            elif examples:
                with open(examples[0], "rb") as f:
                    st.download_button("Download a sample file", f.read(), file_name="sample_trial.csv",
                                       mime="text/csv", icon=":material/download:", type="tertiary")

    trial_key = f"{kind}:{name}:{upload_note}" if name else None
    if ss.get("last_trial") != trial_key:          # a new trial clears the guess
        ss.last_trial = trial_key
        ss.guess = ss.pop("query_guess", None)

    with c2:
        st.html(core.step_title(2, "Your guess"))
        guess = st.segmented_control("Your guess", ["LEFT", "RIGHT"], key="guess",
                                     label_visibility="collapsed", disabled=source is None)
        st.caption("Trust your gut. You score against the given label.")
    with c3:
        st.html(core.step_title(3, "Read the brain"))
        if st.button("Read the brain", type="primary", icon=":material/bolt:", disabled=source is None,
                     width="stretch"):
            ss.revealed, ss.animate = trial_key, True
        if ss.pop("query_reveal", False) and source is not None:
            ss.revealed = trial_key
        st.caption(f"Runs the {len(names)} saved models with the training preprocessing.")

if info is not None:
    if isinstance(info.copy_of, str):
        st.warning(f"**{name}** is a byte-identical copy of **{info.copy_of}**, "
                   f"but here it is labelled **{info.label}**.", icon=":material/content_copy:")
    if isinstance(info.rejected, str):
        st.caption(f":material/report: This trial was rejected in training ({info.rejected}).")

# ================================================================ read the trial
x = None
if source is not None:
    try:
        x = bp.trial_to_array(bp.read_trial(source, name), cfg)
        if upload_note:
            st.caption(f":material/check_circle: Uploaded {upload_note} · read {x.shape[1]:,} samples × "
                       f"{x.shape[0]} channels ({', '.join(CH)}).")
    except ImportError:
        st.error("Reading Excel files needs the openpyxl package: pip install openpyxl", icon=":material/error:")
    except Exception as err:            # wrong columns, too few rows, not a table...
        st.error(f"Could not read **{name}**: {err}", icon=":material/error:")

revealed = x is not None and ss.revealed == trial_key
brain_col, result_col = st.columns([5, 7], gap="medium")

if not revealed:
    with brain_col, st.container(key="card_brain", horizontal_alignment="center"):
        st.image(core.brain_svg(), width=340)
        st.caption("Waiting for you to read the brain...", text_alignment="center")
    with result_col, st.container(key="card_models", height="stretch"):
        placeholder = "".join(
            f'<div class="bci-model"><div class="top"><span class="name">{n}</span></div>'
            f'<div class="kind">{core.MODEL_INFO[n]["kind"]}</div><div class="p">?</div>'
            f'<div class="bci-gauge"><div class="zone"></div><div class="mid"></div></div>'
            f'<div class="bci-scale"><span>LEFT</span><span>50/50</span><span>RIGHT</span></div></div>'
            for n in names)
        st.html('<div class="bci-verdict"><div class="label">The models say</div>'
                f'<div class="big" style="color:{core.NEUTRAL}">?</div></div>'
                f'<div class="bci-models">{placeholder}</div>'
                + core.compare_html() +
                '<div class="bci-note">Pick a trial, make your guess, then press <b>Read the brain</b>.</div>')
    st.stop()

# ================================================================ predict
if ss.pop("animate", False):
    with st.status("Reading the brain...", expanded=True) as status:
        for step in [f"Band-pass {cfg['bandpass_hz'][0]}–{cfg['bandpass_hz'][1]} Hz and {cfg['notch_hz']} Hz notch",
                     f"Cut the {cfg['epoch_seconds'][0]:g}–{cfg['epoch_seconds'][1]:g} s window"] \
                    + [f"{n} is thinking..." for n in names]:
            st.write(step)
            time.sleep(0.3)
        status.update(label="Done", state="complete", expanded=False)

values, problems = bp.quality_check(x, card)
probs = {n: bp.predict_trial(x, bundle, n) for n in names}
decisions = {n: core.decision(p) for n, p in probs.items()}
ep = bp.preprocess(x[np.newaxis], cfg)[0]
freqs, psd, powers = core.band_power(ep, FS)
a3, a4 = core.motor_activity(powers, CH)
verdict = core.vote(probs)
rhythm_side = "RIGHT" if a3 > a4 else "LEFT"

if given and trial_key not in [r["key"] for r in ss.rounds]:
    ss.rounds.append({"key": trial_key, "given": given, "You": guess, **decisions})

with brain_col, st.container(key="card_brain", horizontal_alignment="center"):
    st.image(core.brain_svg(a3, a4, side=verdict), width=340)
    st.caption(f"Signal: mu is quieter over **{'C3' if rhythm_side == 'RIGHT' else 'C4'}**, "
               f"so the brain pattern looks like **{rhythm_side}**. The moving hand is the models' vote.",
               text_alignment="center")

with result_col, st.container(key="card_models", height="stretch"):
    dev = max(abs(p - 0.5) for p in probs.values())
    n_und = sum(d == "Undecided" for d in decisions.values())
    st.html(core.verdict_html(verdict, n_und, len(names), given, guess)
            + f'<div class="bci-models">{"".join(core.model_card_html(n, probs[n]) for n in names)}</div>'
            + core.compare_html(rhythm_side, verdict, given)
            + f'<div class="bci-note">All {len(names)} models stay within <b>±{dev:.3f}</b> of 50/50 '
              f'(grey band = undecided). Cross-validated AUCs: '
            + ", ".join(f"{n} {card['cv'][n]['AUC']:.2f}" for n in names)
            + ", so a confident answer here would be luck.</div>")

# ================================================================ scoreboard
def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


rounds = pd.DataFrame(ss.rounds)
with st.container(key="card_score"):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="bci-step">Scoreboard <span style="color:#7f88ad;font-weight:500">'
                f'· {len(rounds)} round{"s" if len(rounds) != 1 else ""}</span></div>')
        st.space("stretch")
        if st.button("New game", icon=":material/restart_alt:", type="tertiary"):
            ss.rounds = []
            st.rerun()
    if len(rounds):
        items = []
        for player in ["You"] + names:
            calls = rounds[rounds[player].isin(["LEFT", "RIGHT"])]
            passes = len(rounds) - len(calls)
            if len(calls):
                acc = (calls[player] == calls["given"]).mean()
                items.append((player, f"{acc:.0%}", plural(len(calls), "call") + (f" · {plural(passes, 'pass', 'passes')}" if passes else "")))
            else:
                items.append((player, "—", f"no call in {plural(len(rounds), 'round')}"))
        items.append(("Coin flip", "50%", "for reference"))
        st.html(core.score_html(items))
        st.caption("Match rate against the given label, counting only real calls (undecided = pass). "
                   "After 20+ rounds everyone ends up near the coin.")

# ================================================================ details
with st.container(key="card_details"):
    rhythm_tab, spec_tab, sig_tab, q_tab = st.tabs([":material/neurology: Brain rhythms", ":material/equalizer: Spectrum",
                                                    ":material/show_chart: Signal", ":material/verified: Quality check"])

    with rhythm_tab:
        st.caption("Imagining a hand movement lowers mu (8–12 Hz) and beta (13–30 Hz) power over the opposite side. "
                   "Lateralization = (C4 − C3) / (C4 + C3): below 0 looks like LEFT, above 0 like RIGHT.")
        li = {b: float(core.lateralization(v, CH)) for b, v in powers.items()}
        with st.container(horizontal=True):
            for b in core.BANDS:
                st.metric(f"{b} lateralization", f"{li[b]:+.3f}", border=True,
                          help="Looks like " + ("RIGHT" if li[b] > 0 else "LEFT") + " imagery")
            st.metric("mu power C3 / C4", f"{powers['mu'][CH.index('C3')]:.2f} / {powers['mu'][CH.index('C4')]:.2f}",
                      border=True)
        ref = core.training_reference(FS, CH)
        if ref is not None:
            b = st.segmented_control("Band", list(core.BANDS), default="mu", key="li_band") or "mu"
            lo, hi = ref[b].quantile([0.01, 0.99])
            dens = alt.Chart(ref).transform_filter((alt.datum[b] >= lo) & (alt.datum[b] <= hi)).transform_density(
                b, groupby=["label"], as_=["lateralization", "density"], extent=[lo, hi]
            ).mark_area(opacity=0.3, line={"strokeWidth": 2}).encode(
                x=alt.X("lateralization:Q", title=f"{b} lateralization (C4 − C3) / (C4 + C3)"),
                y=alt.Y("density:Q", title="Share of training trials", axis=alt.Axis(labels=False), stack=None),
                color=alt.Color("label:N", title="Training label",
                                scale=alt.Scale(domain=list(core.CLASS_COLORS), range=list(core.CLASS_COLORS.values()))),
                tooltip=["label:N", alt.Tooltip("lateralization:Q", format="+.3f")],
            )
            here = alt.Chart(pd.DataFrame({"x": [li[b]], "t": ["this trial"]})).mark_rule(
                strokeWidth=2, color="#e6e8f0", strokeDash=[4, 3]).encode(
                x="x:Q", tooltip=["t:N", alt.Tooltip("x:Q", format="+.3f")])
            st.altair_chart(dens + here, width="stretch")
            st.caption("With correct labels the blue (LEFT) and orange (RIGHT) curves would sit apart. "
                       "They lie on top of each other, so there is nothing for a model to learn. "
                       "Dashed line = this trial.")

    with spec_tab:
        keep = (freqs >= 1) & (freqs <= 40)
        df_s = pd.DataFrame(psd[:, keep].T, columns=CH).assign(Hz=freqs[keep]).melt(
            "Hz", var_name="channel", value_name="power")
        bands_df = pd.DataFrame([{"band": k, "lo": lo, "hi": hi, "mid": (lo + hi) / 2}
                                 for k, (lo, hi) in core.BANDS.items()])
        shade = alt.Chart(bands_df).mark_rect(opacity=0.08, color="#e6e8f0").encode(x="lo:Q", x2="hi:Q")
        shade_lab = alt.Chart(bands_df).mark_text(baseline="top", dy=4, color="#a9b0cc").encode(
            x="mid:Q", y=alt.value(0), text="band:N")
        lines = alt.Chart(df_s).mark_line(strokeWidth=2).encode(
            x=alt.X("Hz:Q", title="Frequency (Hz)"),
            y=alt.Y("power:Q", scale=alt.Scale(type="log"), title="Power (log scale)"),
            color=alt.Color("channel:N", sort=CH, scale=alt.Scale(domain=CH, range=core.CHANNEL_COLORS)),
            tooltip=["channel", alt.Tooltip("Hz:Q", format=".1f"), alt.Tooltip("power:Q", format=".3g")],
        )
        st.altair_chart(shade + shade_lab + lines, width="stretch")
        st.caption(f"Welch spectrum of the model window ({cfg['epoch_seconds'][0]:g}–{cfg['epoch_seconds'][1]:g} s), "
                   f"after band-pass and {cfg['notch_hz']} Hz notch.")

    with sig_tab:
        with st.container(horizontal=True):
            view = st.segmented_control("View", ["Preprocessed", "Raw"], default="Preprocessed",
                                        key="sig_view") or "Preprocessed"
            chans = st.pills("Channels", CH, default=CH, selection_mode="multi", key="sig_ch") or CH
        sig, t0 = (x - x.mean(1, keepdims=True), 0.0) if view == "Raw" else (ep, cfg["epoch_seconds"][0])
        df_t = pd.DataFrame(sig.T, columns=CH).assign(t=t0 + np.arange(sig.shape[1]) / FS)
        df_t = df_t[["t"] + chans].melt("t", var_name="channel", value_name="value")
        chart = alt.Chart(df_t).mark_line(strokeWidth=1).encode(
            x=alt.X("t:Q", title="Time (s)"), y=alt.Y("value:Q", title="Amplitude (mean removed)"),
            color=alt.Color("channel:N", sort=CH, scale=alt.Scale(domain=CH, range=core.CHANNEL_COLORS)),
            tooltip=["channel", alt.Tooltip("t:Q", format=".2f"), alt.Tooltip("value:Q", format=".1f")],
        )
        if view == "Raw":
            win = alt.Chart(pd.DataFrame({"a": [cfg["epoch_seconds"][0]], "b": [cfg["epoch_seconds"][1]]})).mark_rect(
                opacity=0.07, color="#e6e8f0").encode(x="a:Q", x2="b:Q")
            chart = win + chart
            st.caption("Light area = the window the models see. The first and last second are cut (filter edges).")
        st.altair_chart(chart.interactive(bind_y=False), width="stretch")

    with q_tab:
        lim = card["artifact_limits"]
        labels_q = {"peak_to_peak": "Peak-to-peak", "std": "Std", "flat_share": "Flat samples"}
        for k, lab in labels_q.items():
            fmt = "{:.1%}" if k == "flat_share" else "{:,.0f}"
            st.progress(min(values[k] / lim[k], 1.0),
                        text=f"{lab}: {fmt.format(values[k])} of limit {fmt.format(lim[k])}"
                             + ("  :red[too high]" if k in problems else ""))
        if problems:
            st.warning("Would have been rejected in training: " + ", ".join(labels_q[p] for p in problems) + ".",
                       icon=":material/report:")
        else:
            st.success("Quality check passed (same limits as training, Step 4).", icon=":material/check_circle:")
