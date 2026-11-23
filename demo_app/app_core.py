"""
Shared code for the Streamlit demo (app.py + app_pages/).

Loading (cached), the brain-rhythm math, and the animated brain picture.
All model inference goes through bci_pipeline.py, the same code as training.
"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

import bci_pipeline as bp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# LEFT = blue, RIGHT = orange in every chart, badge and hand.
CLASS_COLORS = {"LEFT": "#3987e5", "RIGHT": "#d95926"}
CHANNEL_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]
BANDS = {"mu": (8, 12), "beta": (13, 30)}
UNDECIDED = 0.05          # |P(RIGHT) - 0.5| below this = the model has no real opinion

MODEL_INFO = {
    "CSP + SVM": {
        "icon": ":material/hub:",
        "kind": "Spatial filters + SVM",
        "how": "Mixes the 4 channels into 4 spatial filters whose 8–30 Hz power differs most "
               "between LEFT and RIGHT, then an SVM draws a line between the two classes.",
    },
    "EEGNet": {
        "icon": ":material/grain:",
        "kind": "Compact CNN",
        "how": "Learns its own frequency filters, then one spatial filter per frequency filter, "
               "end to end from the raw epoch. About 2k parameters.",
    },
    "CNN-LSTM": {
        "icon": ":material/timeline:",
        "kind": "CNN + LSTM",
        "how": "Convolutions pick short local patterns, then an LSTM reads them in order "
               "to follow how the trial changes over time.",
    },
}


# ================================================================ loading
@st.cache_resource
def get_bundle():
    """Model card + every saved model, loaded once for all users."""
    return bp.load_bundle(os.path.join(HERE, "models"))


@st.cache_data
def dataset_index():
    """Every trial file of the dataset with its label, rejection reason and duplicate partner."""
    cfg = bp.load_config()
    files = glob.glob(os.path.join(HERE, cfg["data"]["trial_glob"]), recursive=True)
    if not files:
        return None
    idx = pd.DataFrame({"path": files})
    idx["filename"] = idx.path.map(os.path.basename)
    idx["number"] = idx.filename.str.extract(r"(\d+)", expand=False).astype(int)
    idx = idx.sort_values("number").set_index("filename")

    idx["label"] = pd.read_csv(os.path.join(OUT, "label_map.csv")).set_index("filename")["label"]
    idx["rejected"] = pd.read_csv(os.path.join(OUT, "removed_trials.csv")).set_index("filename")["reason"]
    idx["copy_of"] = None
    for pair, labs in duplicates()[["files", "labels"]].values:
        a, b = pair.split(" = ")
        la, lb = labs.split(" / ")
        idx.loc[a, "copy_of"] = f"{b} (label {lb})"
        idx.loc[b, "copy_of"] = f"{a} (label {la})"
    return idx


@st.cache_data
def duplicates():
    return pd.read_csv(os.path.join(OUT, "duplicates.csv"))


@st.cache_data
def cv_results():
    """outputs/final_results.csv ('0.504 ± 0.029' strings) -> numbers: mean and std per metric."""
    raw = pd.read_csv(os.path.join(OUT, "final_results.csv")).set_index("Model")
    out = {}
    for col in raw.columns:
        parts = raw[col].astype(str).str.split("±", expand=True)
        out[col] = parts[0].astype(float)
        if parts.shape[1] > 1:
            out[col + " std"] = parts[1].astype(float)
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Reading the training trials...")
def training_reference(fs, channels):
    """Mu / beta lateralization of every training trial (outputs/epochs_noCAR.npy)."""
    ep_path, lab_path = os.path.join(OUT, "epochs_noCAR.npy"), os.path.join(OUT, "labels.npy")
    if not (os.path.exists(ep_path) and os.path.exists(lab_path)):
        return None
    _, _, powers = band_power(np.load(ep_path), fs)
    df = pd.DataFrame({b: lateralization(v, channels) for b, v in powers.items()})
    df["label"] = np.load(lab_path, allow_pickle=True)
    return df


# ================================================================ brain rhythms
def band_power(ep, fs):
    """Epoch(s) (..., channels, samples) -> freqs, PSD, {band: mean power (..., channels)}."""
    f, p = welch(ep, fs=fs, nperseg=2 * fs, axis=-1)
    return f, p, {b: p[..., (f >= lo) & (f <= hi)].mean(-1) for b, (lo, hi) in BANDS.items()}


def lateralization(power, channels):
    """(C4 - C3) / (C4 + C3).  < 0: C4 quieter (typical of LEFT imagery), > 0: C3 quieter (RIGHT)."""
    c3, c4 = power[..., channels.index("C3")], power[..., channels.index("C4")]
    return (c4 - c3) / (c4 + c3)


def decision(p):
    if abs(p - 0.5) < UNDECIDED:
        return "Undecided"
    return "RIGHT" if p > 0.5 else "LEFT"


def vote(probs):
    """Majority of the models that have an opinion; 'Undecided' when there is no majority."""
    votes = [decision(p) for p in probs.values()]
    left, right = votes.count("LEFT"), votes.count("RIGHT")
    if left == right:
        return "Undecided"
    return "RIGHT" if right > left else "LEFT"


# ================================================================ animated brain
def _hand(x, mirror, color, cls):
    """A simple hand (palm + 4 fingers + thumb) centered on x. mirror=True draws a right hand."""
    s = -1 if mirror else 1
    fingers = [(-19, 36), (-6, 46), (7, 50), (20, 44)]        # pinky -> index (x offset, length)
    f = "".join(
        f'<rect class="bm-finger" x="{x + s * dx - 5.5:.1f}" y="{392 - ln}" width="11" height="{ln}" rx="5.5"/>'
        for dx, ln in fingers
    )
    thumb_x = x + s * 34                     # thumb points to the midline (palm down)
    rot = 35 if not mirror else -35
    return f"""
    <g class="bm-hand {cls}" style="--c:{color}">
      <g class="bm-fingers">{f}</g>
      <rect x="{thumb_x - 6}" y="388" width="12" height="34" rx="6"
            transform="rotate({rot} {thumb_x} 420)"/>
      <rect x="{x - 28}" y="386" width="56" height="54" rx="16"/>
      <rect x="{x - 22}" y="436" width="44" height="22" rx="8" opacity=".55"/>
    </g>"""


def brain_svg(c3_activity=None, c4_activity=None, side=None):
    """Top view of the head (nose up). C3 / C4 glow with their motor activity (0..1),
    and the hand of `side` ("LEFT" / "RIGHT") squeezes; side=None or "Undecided" = both hands wait.

    C3 (left motor cortex) moves the RIGHT hand, C4 (right motor cortex) moves the LEFT hand,
    so each electrode wears the color of the hand it controls.
    """
    blue, orange = CLASS_COLORS["LEFT"], CLASS_COLORS["RIGHT"]
    a3 = 0.15 if c3_activity is None else float(np.clip(c3_activity, 0, 1))
    a4 = 0.15 if c4_activity is None else float(np.clip(c4_activity, 0, 1))
    active = side if side in ("LEFT", "RIGHT") else None
    lcls = "bm-on" if active == "LEFT" else ("bm-wait" if active is None else "bm-off")
    rcls = "bm-on" if active == "RIGHT" else ("bm-wait" if active is None else "bm-off")

    # Signal path: motor cortex -> opposite hand (crosses the midline, like the real pathway).
    path = ""
    if active == "RIGHT":
        path = f'<path class="bm-path" d="M130 178 C 150 300, 250 290, 290 372" stroke="{orange}"/>'
    elif active == "LEFT":
        path = f'<path class="bm-path" d="M270 178 C 250 300, 150 290, 110 372" stroke="{blue}"/>'
    question = '<text x="200" y="430" class="bm-q">?</text>' if active is None else ""

    gyri = "".join(
        f'<path d="{d}" class="bm-gyrus"/>' for d in [
            "M110 95 C 130 85, 150 100, 170 88", "M90 140 C 115 128, 140 150, 175 132",
            "M85 205 C 110 195, 140 220, 178 200", "M100 255 C 125 245, 150 268, 178 250",
            "M290 95 C 270 85, 250 100, 230 88", "M310 140 C 285 128, 260 150, 225 132",
            "M315 205 C 290 195, 260 220, 222 200", "M300 255 C 275 245, 250 268, 222 250",
        ])

    def electrode(x, y, name, color, a, pulse):
        ring = f'<circle class="bm-ring" cx="{x}" cy="{y}" r="13" stroke="{color}"/>' if pulse else ""
        return f"""
        {ring}
        <circle cx="{x}" cy="{y}" r="13" fill="{color}" fill-opacity="{0.15 + 0.85 * a:.2f}"
                stroke="{color}" stroke-width="2"/>
        <text x="{x}" y="{y + 4}" class="bm-el">{name}</text>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 470">
<style>
  .bm-head {{ fill:url(#bmHead); stroke:#34406a; stroke-width:2; }}
  .bm-hemi {{ fill:url(#bmHemi); stroke:#56639a; stroke-width:1.5; }}
  .bm-gyrus {{ fill:none; stroke:#6b78b3; stroke-width:1.4; stroke-linecap:round; opacity:.45; }}
  .bm-el {{ font: 600 10px Inter, sans-serif; fill:#e6e8f0; text-anchor:middle; }}
  .bm-side {{ font: 600 12px Inter, sans-serif; fill:#8a93b8; text-anchor:middle; }}
  .bm-q {{ font: 700 44px 'Space Grotesk', sans-serif; fill:#8b7cf6; text-anchor:middle;
           animation: bm-breathe 1.8s ease-in-out infinite; }}
  .bm-hand rect {{ fill: var(--c); }}
  .bm-hand {{ transition: opacity .4s; }}
  .bm-off {{ opacity:.18; }}
  .bm-wait {{ opacity:.45; animation: bm-breathe 2.4s ease-in-out infinite; }}
  .bm-on {{ filter: drop-shadow(0 0 10px var(--c)); }}
  .bm-on .bm-finger {{ transform-box: fill-box; transform-origin: 50% 100%;
                      animation: bm-squeeze 1.1s ease-in-out infinite; }}
  .bm-on .bm-finger:nth-child(2) {{ animation-delay: .06s; }}
  .bm-on .bm-finger:nth-child(3) {{ animation-delay: .12s; }}
  .bm-on .bm-finger:nth-child(4) {{ animation-delay: .18s; }}
  .bm-ring {{ fill:none; stroke-width:2; transform-box: fill-box; transform-origin:center;
              animation: bm-pulse 1.4s ease-out infinite; }}
  .bm-path {{ fill:none; stroke-width:3; stroke-linecap:round; stroke-dasharray: 8 10;
              animation: bm-flow .9s linear infinite; opacity:.9; }}
  @keyframes bm-squeeze {{ 0%,100% {{ transform: scaleY(1); }} 50% {{ transform: scaleY(.45); }} }}
  @keyframes bm-pulse {{ 0% {{ transform: scale(1); opacity:.9; }} 100% {{ transform: scale(2.6); opacity:0; }} }}
  @keyframes bm-flow {{ to {{ stroke-dashoffset: -36; }} }}
  @keyframes bm-breathe {{ 0%,100% {{ opacity:.35; }} 50% {{ opacity:.8; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<defs>
  <radialGradient id="bmHead" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#18203c"/><stop offset="1" stop-color="#0e1428"/></radialGradient>
  <radialGradient id="bmHemi" cx="50%" cy="45%" r="65%"><stop offset="0" stop-color="#2a3566"/><stop offset="1" stop-color="#18203f"/></radialGradient>
  <radialGradient id="bmGlowL"><stop offset="0" stop-color="{blue}" stop-opacity=".75"/><stop offset="1" stop-color="{blue}" stop-opacity="0"/></radialGradient>
  <radialGradient id="bmGlowR"><stop offset="0" stop-color="{orange}" stop-opacity=".75"/><stop offset="1" stop-color="{orange}" stop-opacity="0"/></radialGradient>
</defs>
  <polygon points="185,42 200,18 215,42" class="bm-head"/>
  <ellipse cx="58" cy="175" rx="12" ry="26" class="bm-head"/>
  <ellipse cx="342" cy="175" rx="12" ry="26" class="bm-head"/>
  <circle cx="200" cy="175" r="140" class="bm-head"/>
  <path d="M196 52 C 125 48, 75 100, 75 178 C 75 255, 125 300, 196 300 Z" class="bm-hemi"/>
  <path d="M204 52 C 275 48, 325 100, 325 178 C 325 255, 275 300, 204 300 Z" class="bm-hemi"/>
  {gyri}
  <path d="M150 62 C 140 120, 125 160, 118 205" class="bm-gyrus" style="opacity:.9"/>
  <path d="M250 62 C 260 120, 275 160, 282 205" class="bm-gyrus" style="opacity:.9"/>
  <text x="40" y="118" class="bm-side">L</text><text x="360" y="118" class="bm-side">R</text>
  {electrode(200, 110, "Fz", "#8a93b8", 0.1, False)}
  {electrode(200, 178, "Cz", "#8a93b8", 0.1, False)}
  <circle cx="130" cy="178" r="{16 + 30 * a3:.0f}" fill="url(#bmGlowR)"/>
  <circle cx="270" cy="178" r="{16 + 30 * a4:.0f}" fill="url(#bmGlowL)"/>
  {electrode(130, 178, "C3", orange, a3, active == "RIGHT")}
  {electrode(270, 178, "C4", blue, a4, active == "LEFT")}
  {path}
  {_hand(110, False, blue, lcls)}
  {_hand(290, True, orange, rcls)}
  <text x="110" y="466" class="bm-side">LEFT hand</text>
  <text x="290" y="466" class="bm-side">RIGHT hand</text>
  {question}
</svg>"""
    return svg


def motor_activity(powers, channels):
    """Mu suppression -> activity 0..1 for C3 and C4 (a quieter mu rhythm = a busier motor cortex)."""
    c3, c4 = powers["mu"][channels.index("C3")], powers["mu"][channels.index("C4")]
    share3 = c4 / (c3 + c4)          # > 0.5 when C3 is the quieter one
    to01 = lambda s: (s - 0.3) / 0.4  # 0.3 -> 0, 0.7 -> 1
    return to01(share3), to01(1 - share3)


# ================================================================ page styling + HTML cards
# The theme (.streamlit/config.toml) sets colors and fonts; this adds the background glow and the
# card look. Containers are styled through their key (Streamlit adds the class st-key-<key>).
CSS = """
<style>
[data-testid="stApp"] {
  background:
    radial-gradient(900px 500px at 8% -8%, rgba(57,135,229,.18), transparent 60%),
    radial-gradient(800px 500px at 100% -5%, rgba(217,89,38,.13), transparent 55%),
    #070b18;
}
[data-testid="stHeader"] { background: transparent; }
[class*="st-key-card"] {
  background: linear-gradient(180deg, rgba(24,31,56,.85), rgba(15,20,38,.85));
  border: 1px solid #222b4a; border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.bci-eyebrow { font: 600 12px Inter, sans-serif; letter-spacing:.12em; text-transform:uppercase; color:#8b7cf6; }
.bci-step { display:inline-flex; align-items:center; gap:8px; font: 600 14px Inter, sans-serif; color:#e6e8f0; }
.bci-step b { display:inline-grid; place-items:center; width:22px; height:22px; border-radius:50%;
              background:#8b7cf6; color:#0b1020; font-size:12px; }
.bci-verdict { padding: 6px 2px 2px; }
.bci-verdict .label { font: 500 13px Inter, sans-serif; color:#a9b0cc; }
.bci-verdict .big { font: 700 46px/1.05 'Space Grotesk', sans-serif; letter-spacing:-.02em; margin: 4px 0 10px; }
.bci-pills { display:flex; flex-wrap:wrap; gap:6px; }
.bci-pill { display:inline-block; padding:3px 10px; border-radius:999px; font: 600 12px Inter, sans-serif;
            border:1px solid currentColor; }
.bci-models { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-top:16px; }
.bci-model { background:#0d1326; border:1px solid #222b4a; border-radius:14px; padding:14px 14px 12px; }
.bci-model .top { display:flex; justify-content:space-between; align-items:center; gap:6px; }
.bci-model .name { font: 600 14px Inter, sans-serif; color:#e6e8f0; }
.bci-model .kind { font: 500 11.5px Inter, sans-serif; color:#7f88ad; margin: 2px 0 10px; }
.bci-model .p { font: 700 26px 'Space Grotesk', sans-serif; color:#e6e8f0; }
.bci-model .p small { font: 500 11px Inter, sans-serif; color:#7f88ad; margin-left:4px; }
.bci-gauge { position:relative; height:10px; border-radius:99px; background:#1a2240; margin: 10px 0 4px; }
.bci-gauge .zone { position:absolute; left:45%; width:10%; top:0; bottom:0; background:rgba(230,232,240,.10); }
.bci-gauge .mid { position:absolute; left:50%; top:-4px; bottom:-4px; width:2px; margin-left:-1px; background:#5b6590; }
.bci-gauge .fill { position:absolute; top:0; bottom:0; border-radius:99px; }
.bci-gauge .dot { position:absolute; top:50%; width:14px; height:14px; margin:-7px 0 0 -7px; border-radius:50%;
                  border:2px solid #070b18; }
.bci-scale { display:flex; justify-content:space-between; font: 600 10.5px Inter, sans-serif; color:#7f88ad; }
.bci-note { font: 500 13px Inter, sans-serif; color:#a9b0cc; margin-top:14px; line-height:1.5; }
.bci-score { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:10px; }
.bci-score > div { background:#0d1326; border:1px solid #222b4a; border-radius:12px; padding:10px 12px; }
.bci-score .who { font: 600 12px Inter, sans-serif; color:#a9b0cc; }
.bci-score .val { font: 700 24px 'Space Grotesk', sans-serif; color:#e6e8f0; }
.bci-score .sub { font: 500 11px Inter, sans-serif; color:#7f88ad; }
.bci-compare { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:16px; }
.bci-compare > div { border-radius:12px; padding:12px 14px; background:#0d1326; border:1px solid #222b4a; }
.bci-compare .k { font: 600 11px Inter, sans-serif; letter-spacing:.08em; text-transform:uppercase; color:#7f88ad; }
.bci-compare .v { font: 700 22px 'Space Grotesk', sans-serif; margin-top:2px; }
.bci-compare .s { font: 500 11.5px Inter, sans-serif; color:#7f88ad; margin-top:2px; }
</style>
"""

NEUTRAL = "#a99cff"


def side_color(side):
    return CLASS_COLORS.get(side, NEUTRAL)


def step_title(n, text):
    return f'<div class="bci-step"><b>{n}</b>{text}</div>'


def verdict_html(verdict, n_undecided, n_models, given=None, guess=None):
    big = {"LEFT": "LEFT hand", "RIGHT": "RIGHT hand"}.get(verdict, "No clear answer")
    pills = []
    if given:
        pills.append(f'<span class="bci-pill" style="color:#a9b0cc">Given label · {given}</span>')
    if guess and given:
        ok = guess == given
        pills.append(f'<span class="bci-pill" style="color:{"#3fbf6b" if ok else "#e66767"}">'
                     f'Your guess {guess} · {"match" if ok else "no match"}</span>')
    pills.append(f'<span class="bci-pill" style="color:{NEUTRAL}">{n_undecided} of {n_models} models undecided</span>')
    return (f'<div class="bci-verdict"><div class="label">The models say</div>'
            f'<div class="big" style="color:{side_color(verdict)}">{big}</div>'
            f'<div class="bci-pills">{"".join(pills)}</div></div>')


def model_card_html(name, p):
    d = decision(p)
    left = max(0.0, 0.5 - p) * 100          # % of the full bar, drawn from the center
    right = max(0.0, p - 0.5) * 100
    fill = (f'<div class="fill" style="right:50%;width:{left:.1f}%;background:{CLASS_COLORS["LEFT"]}"></div>' if left
            else f'<div class="fill" style="left:50%;width:{right:.1f}%;background:{CLASS_COLORS["RIGHT"]}"></div>')
    info = MODEL_INFO.get(name, {"kind": ""})
    return f"""
<div class="bci-model">
  <div class="top"><span class="name">{name}</span>
    <span class="bci-pill" style="color:{side_color(d)}">{d}</span></div>
  <div class="kind">{info['kind']}</div>
  <div class="p">{p:.3f}<small>P(RIGHT)</small></div>
  <div class="bci-gauge"><div class="zone"></div>{fill}<div class="mid"></div>
    <div class="dot" style="left:{p * 100:.1f}%;background:{side_color(d) if d != 'Undecided' else '#e6e8f0'}"></div></div>
  <div class="bci-scale"><span>LEFT</span><span>50/50</span><span>RIGHT</span></div>
</div>"""


def score_html(items):
    """items: list of (who, value, sub)."""
    cells = "".join(f'<div><div class="who">{w}</div><div class="val">{v}</div><div class="sub">{s}</div></div>'
                    for w, v, s in items)
    return f'<div class="bci-score">{cells}</div>'


def compare_html(brain=None, models=None, given=None):
    """Three tiles: what the signal says, what the models say, what the label says."""
    def tile(k, v, sub):
        color = side_color(v) if v in CLASS_COLORS else (NEUTRAL if v not in (None, "?") else "#5b6590")
        return f'<div><div class="k">{k}</div><div class="v" style="color:{color}">{v or "?"}</div><div class="s">{sub}</div></div>'
    return ('<div class="bci-compare">'
            + tile("Brain signal", brain, "quieter mu side, C3 vs C4")
            + tile("Models", "No call" if models == "Undecided" else models, "majority vote")
            + tile("Given label", given, "from labels.csv")
            + "</div>")
