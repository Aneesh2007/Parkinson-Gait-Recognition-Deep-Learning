#!/usr/bin/env python3
"""
Task 1 re-run: gait recognition on PhysioNet gaitpdb (Parkinson's vs control).

What it does
  1. Loads every gaitpdb recording (GaCo/GaPt/JuCo/JuPt/SiCo/SiPt *.txt).
  2. Picks heel / midfoot / toe sensors per foot (auto-detected from stance timing,
     or set by hand with --pos).
  3. Builds 2 s windows for three sensor sets and compares them:
        S2 = heel + toe per foot       (2 FSR per foot, as in the LMIS paper)
        S3 = heel + mid + toe per foot (your 3 FSR per foot)
        S8 = all 8 sensors per foot    (full dataset)
  4. Trains CNN, LSTM, CNN-LSTM and Transformer with a SUBJECT-WISE split
     (GroupKFold), so no subject appears in both train and test.
  5. Reports window accuracy, macro F1, AUC, subject-level accuracy,
     parameter count and single-window inference time.

Run
  python task1_rerun.py --suggest-only --data path/to/gaitpdb     # check sensor positions first
  python task1_rerun.py --data path/to/gaitpdb --epochs 15 --folds 5
  python task1_rerun.py --data path/to/gaitpdb --epochs 3 --folds 3 --models cnn,transformer   # quick check
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

FS = 100  # gaitpdb sampling rate in Hz


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def list_records(data_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        m = re.match(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", os.path.basename(p))
        if not m:
            continue
        recs.append(
            dict(
                path=p,
                label=int(m.group(2) == "Pt"),  # 1 = Parkinson's, 0 = control
                subject=f"{m.group(1)}{m.group(2)}{m.group(3)}",
            )
        )
    return recs


def load_all(recs):
    arrays, kept = [], []
    for r in recs:
        try:
            a = np.loadtxt(r["path"], dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"  skipped {os.path.basename(r['path'])}: {e}")
            continue
        if a.ndim != 2 or a.shape[1] < 19 or a.shape[0] < 400:
            print(f"  skipped {os.path.basename(r['path'])}: unexpected shape {a.shape}")
            continue
        arrays.append(a)
        kept.append(r)
    return arrays, kept


# ----------------------------------------------------------------------------
# Sensor position detection
# Column layout: 0 time | 1-8 left sensors | 9-16 right sensors | 17 left total | 18 right total
# A heel sensor loads early in stance, a toe sensor late. For each sensor we take the
# force-weighted centre of its loading within stance (0 = foot strike, 1 = toe off).
# ----------------------------------------------------------------------------
def stance_centroids(arrays, thr_frac=0.05, min_len=30):
    per_foot = {"L": [], "R": []}
    for a in arrays:
        for foot, cols, tot in (("L", slice(1, 9), 17), ("R", slice(9, 17), 18)):
            F = a[:, tot]
            thr = thr_frac * np.percentile(F, 95)
            on = (F > thr).astype(int)
            d = np.diff(on, prepend=0, append=0)
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            S = a[:, cols]
            for s, e in zip(starts, ends):
                if e - s < min_len:
                    continue
                seg = np.clip(S[s:e], 0, None)
                t = np.linspace(0, 1, e - s)[:, None]
                den = seg.sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cent = (t * seg).sum(0) / np.where(den > 0, den, np.nan)
                per_foot[foot].append(cent)
    out = {k: np.nanmean(np.vstack(v), axis=0) for k, v in per_foot.items() if v}
    return out


def auto_positions(cents):
    pooled = np.nanmean(np.vstack(list(cents.values())), axis=0)
    order = np.argsort(pooled)
    heel, toe = int(order[0]), int(order[-1])
    rest = [i for i in range(8) if i not in (heel, toe)]
    target = (pooled[heel] + pooled[toe]) / 2
    mid = int(min(rest, key=lambda i: abs(pooled[i] - target)))
    return heel, mid, toe, pooled


# ----------------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------------
def make_windows(arrays, recs, sel, win_raw, ds, trim=200):
    T = win_raw // ds
    step = max(T // 2, 1)
    Xs, ys, gs = [], [], []
    cols = [1 + i for i in sel] + [9 + i for i in sel]  # left sensors then right sensors
    for a, r in zip(arrays, recs):
        x = a[:, cols].astype(np.float32)
        if len(x) > 2 * trim + 4 * win_raw:
            x = x[trim:-trim]  # drop start-up and stopping
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)  # per-recording, per-channel z-score
        if ds > 1:
            n = len(x) // ds
            x = x[: n * ds].reshape(n, ds, -1).mean(1)
        if len(x) < T:
            continue
        w = np.lib.stride_tricks.sliding_window_view(x, T, axis=0)[::step]  # (N, C, T)
        w = np.ascontiguousarray(w.transpose(0, 2, 1))  # (N, T, C)
        Xs.append(w)
        ys.append(np.full(len(w), r["label"], dtype=np.int64))
        gs.extend([r["subject"]] * len(w))
    return np.concatenate(Xs), np.concatenate(ys), np.array(gs)


# ----------------------------------------------------------------------------
# Models (input: batch, time, channels)
# ----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, C, T, nc=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(C, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, nc))

    def forward(self, x):
        return self.h(self.f(x.transpose(1, 2)).squeeze(-1))


class LSTMNet(nn.Module):
    def __init__(self, C, T, nc=2):
        super().__init__()
        self.rnn = nn.LSTM(C, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, nc))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.h(out.mean(1))


class CNNLSTM(nn.Module):
    def __init__(self, C, T, nc=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(64, 64, batch_first=True)
        self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, nc))

    def forward(self, x):
        z = self.f(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(z)
        return self.h(out.mean(1))


class TransformerNet(nn.Module):
    def __init__(self, C, T, nc=2, d=64):
        super().__init__()
        self.proj = nn.Linear(C, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.h = nn.Linear(d, nc)

    def forward(self, x):
        z = self.enc(self.proj(x) + self.pos)
        return self.h(self.norm(z.mean(1)))


MODELS = {"cnn": CNN, "lstm": LSTMNet, "cnnlstm": CNNLSTM, "transformer": TransformerNet}
LR = {"cnn": 1e-3, "lstm": 2e-3, "cnnlstm": 1e-3, "transformer": 5e-4}


# ----------------------------------------------------------------------------
# Train and evaluate one model on one fold
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, X, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(X[i : i + bs].to(device)), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_eval(name, Xtr, ytr, Xte, yte, gte, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    C, T = Xtr.shape[2], Xtr.shape[1]
    model = MODELS[name](C, T).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xte_t = torch.from_numpy(Xte)
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    w = torch.tensor(len(ytr) / (2 * np.maximum(counts, 1)), device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    train_s = time.time() - t0

    prob = predict_proba(model, Xte_t, device)
    pred = (prob > 0.5).astype(int)
    try:
        auc = roc_auc_score(yte, prob)
    except ValueError:
        auc = float("nan")

    df = pd.DataFrame({"g": gte, "y": yte, "p": prob})
    sub = df.groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    subj_acc = accuracy_score(sub["y"], (sub["p"] > 0.5).astype(int))

    model_cpu = model.to("cpu").eval()
    x1 = Xte_t[:1]
    with torch.no_grad():
        for _ in range(5):
            model_cpu(x1)
        t1 = time.time()
        for _ in range(50):
            model_cpu(x1)
        ms = (time.time() - t1) / 50 * 1000

    return dict(
        acc=accuracy_score(yte, pred),
        f1=f1_score(yte, pred, average="macro"),
        auc=auc,
        subj_acc=subj_acc,
        params=n_params,
        infer_ms=ms,
        train_s=train_s,
    )


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the gaitpdb *.txt files")
    ap.add_argument("--pos", default="", help="1-based sensor numbers heel,mid,toe (overrides auto)")
    ap.add_argument("--suggest-only", action="store_true")
    ap.add_argument("--models", default="cnn,lstm,cnnlstm,transformer")
    ap.add_argument("--sets", default="S2,S3,S8")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--win", type=int, default=200, help="window length in samples at 100 Hz")
    ap.add_argument("--ds", type=int, default=2, help="downsample factor (2 = 50 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_task1")
    args = ap.parse_args()

    recs = list_records(args.data)
    if not recs:
        sys.exit(f"No gaitpdb files (e.g. GaCo01_01.txt) found in {args.data}")
    print(f"Found {len(recs)} recordings, loading...")
    arrays, recs = load_all(recs)
    subj = {r["subject"]: r["label"] for r in recs}
    print(f"Loaded {len(recs)} recordings from {len(subj)} subjects "
          f"({sum(subj.values())} Parkinson's, {len(subj) - sum(subj.values())} control)")

    cents = stance_centroids(arrays)
    heel, mid, toe, pooled = auto_positions(cents)
    print("\nStance-loading centre per sensor (0 = foot strike, 1 = toe off), sensors 1..8:")
    for foot, c in cents.items():
        print(f"  {foot}: " + " ".join(f"{v:.2f}" for v in c))
    print("  pooled: " + " ".join(f"{v:.2f}" for v in pooled))
    if args.pos:
        heel, mid, toe = [int(v) - 1 for v in args.pos.split(",")]
        print(f"Using manual positions (1-based): heel={heel+1}, mid={mid+1}, toe={toe+1}")
    else:
        print(f"Auto positions (1-based): heel={heel+1}, mid={mid+1}, toe={toe+1}")
        print("Check these against the sensor figure on the PhysioNet gaitpdb page; "
              "override with --pos heel,mid,toe if they differ.")
    if args.suggest_only:
        return

    sets = {"S2": [heel, toe], "S3": [heel, mid, toe], "S8": list(range(8))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    rows = []
    for sname in [s.strip() for s in args.sets.split(",")]:
        X, y, g = make_windows(arrays, recs, sets[sname], args.win, args.ds)
        print(f"\n=== {sname}: {X.shape[0]} windows, {X.shape[2]} channels, "
              f"{X.shape[1]} samples each, {100*y.mean():.0f}% Parkinson's ===")
        gkf = GroupKFold(n_splits=args.folds)
        for fold, (tr, te) in enumerate(gkf.split(X, y, g)):
            for mname in [m.strip() for m in args.models.split(",")]:
                r = train_eval(mname, X[tr], y[tr], X[te], y[te], g[te], args, device,
                               args.seed + fold)
                r.update(sensors=sname, model=mname, fold=fold)
                rows.append(r)
                print(f"  fold {fold} {mname:12s} acc={r['acc']:.3f} f1={r['f1']:.3f} "
                      f"auc={r['auc']:.3f} subj={r['subj_acc']:.3f} ({r['train_s']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}_per_fold.csv", index=False)
    agg = (
        df.groupby(["sensors", "model"])
        .agg(acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean"), auc=("auc", "mean"),
             subj_acc=("subj_acc", "mean"), params=("params", "first"),
             infer_ms=("infer_ms", "mean"))
        .round(3)
        .reset_index()
    )
    agg.to_csv(f"{args.out}_summary.csv", index=False)
    with open(f"{args.out}_summary.md", "w") as f:
        f.write(agg.to_markdown(index=False))
    with open(f"{args.out}_config.json", "w") as f:
        json.dump(dict(vars(args), heel=heel + 1, mid=mid + 1, toe=toe + 1), f, indent=2)
    print("\n" + agg.to_string(index=False))
    print(f"\nSaved {args.out}_summary.csv, {args.out}_summary.md, {args.out}_per_fold.csv")


if __name__ == "__main__":
    main()
