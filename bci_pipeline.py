"""Inference code for the LEFT/RIGHT motor-imagery demo (section 11 of main_pipeline.ipynb).

The Streamlit app (app.py) and the notebook both use this file, so a raw trial file goes
through the same steps as in training:

    raw trial (2500 x [FZ, C3, CZ, C4])
      -> remove offset, band-pass 0.5-40 Hz, notch 50 Hz        (main_pipeline section 2)
      -> artifact check with the limits found on the training set (section 2)
      -> common average reference, epoch 1-9 s, remove mean      (sections 2 and 8)
      -> model input (CSP + SVM: the epoch; EEGNet / CNN-LSTM: decimate + channel z-score
         with the training statistics; Transformer: decimate + per-trial z-score)

The network classes are copies of the ones in the notebook, so the saved state_dicts load
into them. Section 11.2 of the notebook checks that this file reproduces the notebook's
epochs and predictions exactly.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn

import bci_features  # noqa: F401  (the CSP + SVM pickle needs bci_features.FilterBankCSP)

CLASS_NAMES = ["LEFT", "RIGHT"]
MODEL_FILES = {"CSP + SVM": "csp_svm.joblib", "EEGNet": "eegnet.pt",
               "CNN-LSTM": "cnn_lstm.pt", "Transformer": "transformer.pt"}


# --------------------------------------------------------------------------- reading
def read_trial(source, name=None):
    """Read one trial from a path or an uploaded file (CSV or Excel)."""
    name = str(name or getattr(source, "name", source))
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def trial_to_array(df, cfg):
    """DataFrame -> (channels, samples) array in the training channel order."""
    cols = {str(c).strip().upper(): c for c in df.columns}
    missing = [ch for ch in cfg["channels"] if ch not in cols]
    if missing:
        raise ValueError(f"missing channel column(s) {missing}; expected {cfg['channels']}")
    n = cfg["fs"] * cfg["trial_seconds"]
    x = df[[cols[ch] for ch in cfg["channels"]]].to_numpy(dtype=float)
    if len(x) < n:
        raise ValueError(f"the trial has {len(x)} rows; at least {n} are needed "
                         f"({cfg['trial_seconds']} s at {cfg['fs']} Hz)")
    if np.isnan(x[:n]).any():
        raise ValueError("the trial contains empty / NaN values")
    return x[:n].T


# --------------------------------------------------------------------- preprocessing
def filter_trials(X_raw, cfg):
    """(trials, channels, samples) raw -> offset removed, band-pass + notch filtered."""
    sos_bp = signal.butter(4, cfg["bandpass_hz"], btype="bandpass", fs=cfg["fs"], output="sos")
    b_notch, a_notch = signal.iirnotch(cfg["notch_hz"], cfg["notch_q"], fs=cfg["fs"])
    X = signal.sosfiltfilt(sos_bp, X_raw - X_raw.mean(-1, keepdims=True), axis=-1)
    return signal.filtfilt(b_notch, a_notch, X, axis=-1)


def artifact_values(X_raw, cfg, X_filt=None):
    """Per-trial artifact measures (worst channel), as in main_pipeline section 2."""
    X_filt = filter_trials(X_raw, cfg) if X_filt is None else X_filt
    e = int(cfg["artifact"]["edge_seconds"] * cfg["fs"])
    core = X_filt[..., e:-e]
    return {"peak_to_peak": np.ptp(core, axis=-1).max(1),
            "std": core.std(-1).max(1),
            "flat_share": (np.diff(X_raw, axis=-1) == 0).mean(-1).max(1)}


def artifact_flags(values, limits):
    """True for trials the training pipeline would have removed."""
    return np.any([values[k] > limits[k] for k in limits], axis=0)


def preprocess(X_raw, cfg, X_filt=None):
    """(trials, channels, samples) raw -> CAR epochs (trials, channels, epoch samples), float32."""
    X_filt = filter_trials(X_raw, cfg) if X_filt is None else X_filt
    X_car = X_filt - X_filt.mean(axis=1, keepdims=True)
    s0, s1 = (int(s * cfg["fs"]) for s in cfg["epoch_seconds"])
    ep = X_car[..., s0:s1]
    return (ep - ep.mean(-1, keepdims=True)).astype(np.float32)


def quality_check(x_raw, card):
    """One raw trial (channels, samples) -> (measured values, list of problems)."""
    cfg, limits = card["config"], card["artifact_limits"]
    vals = {k: float(v[0]) for k, v in artifact_values(x_raw[None], cfg).items()}
    problems = [f"{k} = {vals[k]:.3g} is above the limit {limits[k]:.3g}" for k in limits if vals[k] > limits[k]]
    return vals, problems


def prep_for_transformer(X, tf_cfg):
    """Epochs -> Transformer input: decimate, then z-score each channel of each trial."""
    if tf_cfg["decimate"] > 1:
        X = signal.decimate(X, tf_cfg["decimate"], axis=-1, zero_phase=True)
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return X.astype(np.float32)


# --------------------------------------------------------------------------- models
class EEGNet(nn.Module):
    def __init__(self, n_ch, n_t, F1=8, D=2, F2=16, k=64, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
                                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2),
                                nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_flat = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.fc = nn.Linear(n_flat, 2)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x)).flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, n_ch, n_t, hidden=32, drop=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ELU(), nn.MaxPool1d(4), nn.Dropout(drop),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ELU(), nn.MaxPool1d(4), nn.Dropout(drop))
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        z = self.cnn(x.squeeze(1)).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.fc(h[-1])


class PatchEmbedding(nn.Module):
    def __init__(self, n_ch, cfg):
        super().__init__()
        f, k = cfg["conv_filters"], cfg["temporal_kernel"]
        self.net = nn.Sequential(
            nn.Conv2d(1, f, (1, k), padding=(0, k // 2), bias=False),
            nn.Conv2d(f, f, (n_ch, 1), bias=False),
            nn.BatchNorm2d(f), nn.ELU(),
            nn.AvgPool2d((1, cfg["pool"]), (1, cfg["pool_stride"])),
            nn.Dropout(cfg["dropout"]),
            nn.Conv2d(f, cfg["d_model"], 1))

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(2).transpose(1, 2)


class EEGTransformer(nn.Module):
    def __init__(self, n_ch, n_times, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.embed = PatchEmbedding(n_ch, cfg)
        self.n_tokens = (n_times - cfg["pool"]) // cfg["pool_stride"] + 1
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg["n_heads"], d * cfg["ff_mult"], cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["n_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        z = self.embed(x)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


NETS = {"EEGNet": EEGNet, "CNN-LSTM": CNNLSTM}


# ------------------------------------------------------------------ load + predict
def load_bundle(models_dir):
    """Model card + every model listed in it, ready for predict_epochs()."""
    with open(os.path.join(models_dir, "model_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    models = {}
    for name, fname in card["model_files"].items():
        path = os.path.join(models_dir, fname)
        if name == "CSP + SVM":
            models[name] = joblib.load(path)
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if name == "Transformer":
            net = EEGTransformer(ckpt["n_channels"], ckpt["tf_cfg"]["crop_len"], ckpt["tf_cfg"])
        else:
            net = NETS[name](ckpt["n_ch"], ckpt["n_t"])
        net.load_state_dict(ckpt["model_state"])
        models[name] = (net.eval(), ckpt)
    return {"card": card, "models": models}


@torch.no_grad()
def predict_epochs(X_ep, bundle, name):
    """CAR epochs (trials, channels, samples) -> P(RIGHT) per trial."""
    model = bundle["models"][name]
    if name == "CSP + SVM":
        return model.predict_proba(X_ep)[:, 1]
    net, ckpt = model
    if name == "Transformer":
        tf_cfg = ckpt["tf_cfg"]
        X = torch.from_numpy(prep_for_transformer(X_ep, tf_cfg))
        T, L = X.shape[-1], tf_cfg["crop_len"]
        starts = np.linspace(0, T - L, tf_cfg["test_crops"]).round().astype(int)
        return (sum(F.softmax(net(X[..., s:s + L]).float(), dim=1)[:, 1] for s in starts) / len(starts)).numpy()
    Xd = X_ep[..., ::bundle["card"]["dl_decimate"]]
    mean, std = ckpt["scaler_mean"].numpy(), ckpt["scaler_std"].numpy()
    X = torch.from_numpy(((Xd - mean) / std).astype(np.float32)[:, None])
    return torch.softmax(net(X), 1)[:, 1].numpy()


def predict_trial(x_raw, bundle, name):
    """One raw trial (channels, samples) -> P(RIGHT) from the chosen model."""
    return float(predict_epochs(preprocess(x_raw[None], bundle["card"]["config"]), bundle, name)[0])
