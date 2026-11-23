"""
Shared code for the BCI motor-imagery pipeline.

Both the training notebook (BCI_MI_Pipeline.ipynb) and the demo app (app.py)
import this file, so a trial is processed in exactly the same way for training
and for prediction (task, section 6.2: "Keep preprocessing code identical
between training and inference").

Contents
    1. Config            load_config()
    2. Reading a trial   read_trial(), trial_to_array()
    3. Processing        bandpass(), notch(), car(), cut_epoch(), baseline(), zscore()
                         preprocess()  -> Table 1 steps 2, 3, 5, 6, 7 in order
    4. Model inputs      csp_input(), dl_input()
    5. Quality check     artifact_measures()
    6. Deep models       EEGNet, CNNLSTM, train_torch(), predict_torch()
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, iirnotch

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 1. config
def load_config(path=None):
    """Read the single shared config file (task tip: one config for the team)."""
    path = path or os.path.join(HERE, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------- 2. read a trial
def read_trial(file, name=None):
    """CSV or Excel trial -> DataFrame. `file` can be a path or an uploaded file."""
    name = (name or getattr(file, "name", None) or str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)          # needs openpyxl
    return pd.read_csv(file)


def trial_to_array(df, cfg):
    """DataFrame -> array (channels, samples), in the channel order of the config."""
    channels = cfg["channels"]
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing channel columns: {missing}. Expected {channels}.")
    n = cfg["fs"] * cfg["trial_seconds"]
    if len(df) < n:
        raise ValueError(f"Trial has {len(df)} rows, needs {n} ({cfg['trial_seconds']} s at {cfg['fs']} Hz).")
    return df[channels].to_numpy(dtype=float)[:n].T


# ---------------------------------------------------------- 3. processing
def bandpass(x, low, high, fs, order=4):
    """Zero-phase Butterworth band-pass along the last axis (time)."""
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=-1)


def notch(x, freq, fs, q=30):
    """Zero-phase notch filter that removes one frequency (power line)."""
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def car(x):
    """Common Average Reference: subtract the mean of all channels at every time point."""
    return x - x.mean(axis=-2, keepdims=True)


def cut_epoch(x, cfg):
    """Keep the time window from the config (in seconds)."""
    s, e = (int(t * cfg["fs"]) for t in cfg["epoch_seconds"])
    return x[..., s:e]


def baseline(x):
    """Center every channel of every epoch on 0 (no pre-cue rest period exists)."""
    return x - x.mean(axis=-1, keepdims=True)


def zscore(x):
    """Scale every channel of every epoch to mean 0 and std 1."""
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-12)


def preprocess(x_raw, cfg, use_car=None):
    """Raw trial(s) (..., channels, samples) -> clean epochs.

    Order = Table 1 of the task: band-pass -> notch -> (artifact check is done
    outside) -> re-reference -> epoch -> baseline. Normalization (step 8) is
    model-specific and done in dl_input().
    """
    use_car = cfg["use_car"] if use_car is None else use_car
    x = bandpass(x_raw, *cfg["bandpass_hz"], cfg["fs"])
    x = notch(x, cfg["notch_hz"], cfg["fs"], cfg["notch_q"])
    if use_car:
        x = car(x)
    x = cut_epoch(x, cfg)
    return baseline(x)


# ---------------------------------------------------------- 4. model inputs
def csp_input(epochs, cfg):
    """CSP + SVM input: keep mu + beta only (task: 'tighter 8-30 Hz')."""
    return bandpass(epochs, *cfg["csp_band_hz"], cfg["fs"])


def dl_input(epochs, cfg):
    """Deep-learning input: downsample, normalize (step 8), shape (N, 1, C, T)."""
    x = decimate(epochs, cfg["dl_downsample"], axis=-1, zero_phase=True)
    x = zscore(x)
    if x.ndim == 2:
        x = x[np.newaxis]
    return x[:, np.newaxis].astype(np.float32)


# ---------------------------------------------------------- 5. quality check
def artifact_measures(x_raw, cfg):
    """Three numbers per trial used by artifact rejection (step 4).

    peak_to_peak : largest max-min over channels (blinks, electrode pops)
    std          : largest standard deviation over channels (muscle, big slow waves)
    flat_share   : share of raw samples that did not change (disconnected electrode)
    The first and last `edge_seconds` are skipped: the filter bends them.
    """
    x = bandpass(x_raw, *cfg["bandpass_hz"], cfg["fs"])
    x = notch(x, cfg["notch_hz"], cfg["fs"], cfg["notch_q"])
    edge = int(cfg["artifact"]["edge_seconds"] * cfg["fs"])
    mid = x[..., edge:-edge]
    return {
        "peak_to_peak": (mid.max(-1) - mid.min(-1)).max(-1),
        "std": mid.std(-1).max(-1),
        "flat_share": (np.diff(x_raw, axis=-1) == 0).mean(axis=(-2, -1)),
    }


# ---------------------------------------------------------- 6. deep models
try:
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        """EEGNet (Lawhern et al., 2018): temporal filter -> spatial filter -> separable filter."""

        def __init__(self, n_channels=4, n_samples=1000, n_classes=2, F1=8, D=2, F2=16, dropout=0.5):
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(F1),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            )
            with torch.no_grad():
                n_out = self._features(torch.zeros(1, 1, n_channels, n_samples)).shape[1]
            self.classify = nn.Linear(n_out, n_classes)

        def _features(self, x):
            return self.separable(self.spatial(self.temporal(x))).flatten(1)

        def forward(self, x):
            return self.classify(self._features(x))

    class CNNLSTM(nn.Module):
        """1-D CNN learns short local patterns, the LSTM reads how they change over time."""

        def __init__(self, n_channels=4, n_samples=1000, n_classes=2, hidden=32, dropout=0.5):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ELU(),
                nn.MaxPool1d(4), nn.Dropout(dropout),
                nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ELU(),
                nn.MaxPool1d(4), nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(32, hidden, batch_first=True)
            self.classify = nn.Linear(hidden, n_classes)

        def forward(self, x):
            x = self.conv(x.squeeze(1))           # (N, 32, T/16)
            _, (h, _) = self.lstm(x.transpose(1, 2))
            return self.classify(h[-1])

    MODELS = {"EEGNet": EEGNet, "CNN-LSTM": CNNLSTM}

    def predict_torch(net, X, batch_size=256):
        """Probability of class 1 (RIGHT) for every trial."""
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size])
                out.append(torch.softmax(net(xb), dim=1)[:, 1].numpy())
        return np.concatenate(out)

    def train_torch(model_name, X_tr, y_tr, X_val, y_val, cfg, seed=0, fixed_epochs=None):
        """Train with Adam and early stopping on a validation set taken from the training sessions.

        The test fold is never used for any decision, so the test score stays honest.
        Returns the model with its best weights and the history per epoch.
        With `fixed_epochs` (final model on all data) there is no validation set:
        it trains exactly that many epochs and keeps the last weights.
        """
        if fixed_epochs is not None:
            return _train_fixed(model_name, X_tr, y_tr, cfg, seed, fixed_epochs)
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(seed)
        np.random.seed(seed)
        d = cfg["dl"]
        net = MODELS[model_name](n_channels=X_tr.shape[2], n_samples=X_tr.shape[3])
        opt = torch.optim.Adam(net.parameters(), lr=d["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                            batch_size=d["batch_size"], shuffle=True, generator=g)
        Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val).long()

        best_loss, best_state, wait, history = np.inf, None, 0, []
        for ep in range(d["max_epochs"]):
            net.train()
            correct, total, run_loss = 0, 0, 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
            net.eval()
            with torch.no_grad():
                vout = net(Xv)
                vloss = loss_fn(vout, yv).item()
                vacc = (vout.argmax(1) == yv).float().mean().item()
            history.append({"epoch": ep + 1, "train_loss": run_loss / total, "train_acc": correct / total,
                            "val_loss": vloss, "val_acc": vacc})
            if vloss < best_loss - 1e-4:
                best_loss, wait = vloss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= d["patience"]:
                    break
        net.load_state_dict(best_state)
        return net, pd.DataFrame(history)

    def _train_fixed(model_name, X_tr, y_tr, cfg, seed, n_epochs):
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(seed)
        np.random.seed(seed)
        d = cfg["dl"]
        net = MODELS[model_name](n_channels=X_tr.shape[2], n_samples=X_tr.shape[3])
        opt = torch.optim.Adam(net.parameters(), lr=d["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                            batch_size=d["batch_size"], shuffle=True, generator=g)
        history = []
        for ep in range(n_epochs):
            net.train()
            correct, total, run_loss = 0, 0, 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
            history.append({"epoch": ep + 1, "train_loss": run_loss / total, "train_acc": correct / total})
        net.eval()
        return net, pd.DataFrame(history)

except ImportError:          # the app can still run CSP + SVM without torch
    torch = None


# ---------------------------------------------------------- 7. inference
MODEL_FILES = {"CSP + SVM": "csp_svm.joblib", "EEGNet": "eegnet.pt", "CNN-LSTM": "cnn_lstm.pt"}


def load_bundle(models_dir=None):
    """Load the model card and every saved model from the models folder."""
    import joblib

    models_dir = models_dir or os.path.join(HERE, "models")
    with open(os.path.join(models_dir, "model_card.json"), encoding="utf-8") as f:
        card = json.load(f)
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            continue
        if fname.endswith(".joblib"):
            models[name] = joblib.load(path)
        elif torch is not None:
            _, n_ch, n_s = card["dl_input_shape"]
            net = MODELS[name](n_channels=n_ch, n_samples=n_s)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            models[name] = net
    return {"card": card, "config": card["config"], "models": models}


def quality_check(x_raw, card):
    """Compare one raw trial with the artifact limits learned in training (step 4)."""
    m = artifact_measures(x_raw[np.newaxis], card["config"])
    lim = card["artifact_limits"]
    values = {k: float(v[0]) for k, v in m.items()}
    problems = [k for k in ("peak_to_peak", "std", "flat_share") if values[k] > lim[k]]
    return values, problems


def predict_trial(x_raw, bundle, model_name):
    """One raw trial (channels, samples) -> probability of RIGHT, with the exact training preprocessing."""
    cfg = bundle["config"]
    ep = preprocess(x_raw[np.newaxis], cfg)
    model = bundle["models"][model_name]
    if model_name == "CSP + SVM":
        return float(model.predict_proba(csp_input(ep, cfg))[0, 1])
    return float(predict_torch(model, dl_input(ep, cfg))[0])
