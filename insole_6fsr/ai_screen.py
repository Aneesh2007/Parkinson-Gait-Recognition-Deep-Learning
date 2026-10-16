"""
AI gait screening for the dashboard: the CNN from Task 1 (3 sensors per foot), run in plain NumPy.

Why NumPy: the cloud app then needs no PyTorch install. The weights come from the Colab export
cell (cnn_s3.npz). The model is tiny (37,090 parameters).

What it does with an uploaded recording (6 FSR channels, ADC counts, about 100 Hz):
  1. Convert each channel to a "load" signal: how far the reading dropped below its unloaded level.
     (The training data is force, which rises with load; our ADC falls with load.)
  2. Reorder channels to the training order: L heel, L mid, L toe, R heel, R mid, R toe.
  3. Same preprocessing as training: trim the first and last 2 s of long walks, z-score each channel
     over the recording, average pairs of samples to 50 Hz, cut 2 s windows with 50% overlap.
  4. Run the CNN on every window. Output = mean probability of a Parkinson's-like pattern.

IMPORTANT: the model was trained on the public PhysioNet gaitpdb data (a different insole, mostly
older adults). It has not been validated on this insole. Treat the output as a research prototype
screening score, never as a diagnosis, and never on patients who do not have Parkinson's questions
(it only knows "Parkinson's vs healthy").
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("INSOLE_MODEL", os.path.join(HERE, "cnn_s3.npz"))

FS_IN = 100          # recorder sample rate (Hz)
WIN = 200            # 2 s at 100 Hz
DS = 2               # 100 Hz -> 50 Hz
TRIM = 200           # samples dropped at each end of a long recording (as in training)
# recorder order: L_toe, L_mid, L_heel, R_toe, R_mid, R_heel -> training order
ORDER = [2, 1, 0, 5, 4, 3]
MIN_RANGE = 150      # a channel that moves less than this (ADC counts) is treated as dead

_model = None
_meta = {}


def available():
    return os.path.exists(MODEL_PATH)


def _load():
    global _model, _meta
    if _model is None:
        z = np.load(MODEL_PATH)
        _model = {k: z[k].astype(np.float32) for k in z.files if not k.endswith("num_batches_tracked")}
        meta_path = os.path.splitext(MODEL_PATH)[0] + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _meta = json.load(f)
    return _model


def info():
    if not available():
        return {"available": False}
    _load()
    return {"available": True, "model": "CNN, 3 sensors per foot, trained on PhysioNet gaitpdb", **_meta}


# ----------------------------------------------------------------------------- preprocessing
def windows_from_signal(x, win_raw=WIN, ds=DS, trim=TRIM):
    """x: (N, C) at 100 Hz, channels already in training order. Mirrors task1_rerun.make_windows."""
    T = win_raw // ds
    step = max(T // 2, 1)
    x = x.astype(np.float32)
    if len(x) > 2 * trim + 4 * win_raw:
        x = x[trim:-trim]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    if ds > 1:
        n = len(x) // ds
        x = x[: n * ds].reshape(n, ds, -1).mean(1)
    if len(x) < T:
        return np.zeros((0, T, x.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(x, T, axis=0)[::step]   # (N, C, T)
    return np.ascontiguousarray(w.transpose(0, 2, 1))                    # (N, T, C)


def load_signal(t, raw):
    """t: seconds, raw: (N, 6) ADC counts in recorder order. Returns (N', 6) load signal at 100 Hz
    in training channel order, and the list of dead channel names."""
    t = np.asarray(t, dtype=float)
    raw = np.asarray(raw, dtype=float)
    grid = np.arange(t[0], t[-1], 1.0 / FS_IN)
    res = np.column_stack([np.interp(grid, t, raw[:, i]) for i in range(raw.shape[1])])
    names = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
    dead = []
    sig = np.zeros_like(res)
    for i in range(res.shape[1]):
        col = res[:, i]
        if np.percentile(col, 99) - np.percentile(col, 1) < MIN_RANGE:
            dead.append(names[i])
            continue
        sig[:, i] = np.clip(np.percentile(col, 99) - col, 0, None)
    return sig[:, ORDER], dead


# ----------------------------------------------------------------------------- the CNN in NumPy
def _conv1d(x, w, b, pad):
    """x: (N, C, T), w: (O, C, K) -> (N, O, T) with 'same' padding."""
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)      # (N, C, T, K)
    return np.einsum("nctk,ock->not", win, w, optimize=True) + b[None, :, None]


def _bn(x, p, name):
    g, bt = p[name + ".weight"], p[name + ".bias"]
    m, v = p[name + ".running_mean"], p[name + ".running_var"]
    return (x - m[None, :, None]) / np.sqrt(v[None, :, None] + 1e-5) * g[None, :, None] + bt[None, :, None]


def _pool2(x):
    n, c, t = x.shape
    return x[:, :, : t // 2 * 2].reshape(n, c, t // 2, 2).max(-1)


def predict_proba(windows):
    """windows: (N, T, C). Returns P(Parkinson's-like) per window. Matches task1_rerun.CNN in eval mode."""
    p = _load()
    x = np.ascontiguousarray(windows.transpose(0, 2, 1)).astype(np.float32)   # (N, C, T)
    x = np.maximum(_bn(_conv1d(x, p["f.0.weight"], p["f.0.bias"], 3), p, "f.1"), 0)
    x = _pool2(x)
    x = np.maximum(_bn(_conv1d(x, p["f.4.weight"], p["f.4.bias"], 2), p, "f.5"), 0)
    x = _pool2(x)
    x = np.maximum(_bn(_conv1d(x, p["f.8.weight"], p["f.8.bias"], 1), p, "f.9"), 0)
    x = x.mean(-1)                                                           # adaptive average pool
    logits = x @ p["h.1.weight"].T + p["h.1.bias"]
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return (e[:, 1] / e.sum(1)).astype(float)


# ----------------------------------------------------------------------------- public entry point
def band(score):
    return "low" if score < 0.35 else "high" if score > 0.65 else "uncertain"


def screen(t, raw):
    """Return a dict for the dashboard, or None if no model is installed."""
    if not available():
        return None
    sig, dead = load_signal(t, raw)
    if dead:
        return {"ok": False, "reason": "Not screened: no signal on " + ", ".join(dead)
                + ". Fix those sensors and record again."}
    w = windows_from_signal(sig)
    if len(w) < 3:
        return {"ok": False, "reason": "Not screened: recording too short (needs about 10 s of walking)."}
    pr = predict_proba(w)
    score = float(pr.mean())
    return {
        "ok": True, "score": round(score, 3), "band": band(score),
        "windows": int(len(pr)), "windows_pd": int((pr > 0.5).sum()),
        "model": "CNN (3 sensors per foot), PhysioNet gaitpdb", "preview": False,
    }
