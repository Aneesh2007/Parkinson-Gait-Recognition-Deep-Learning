#!/usr/bin/env python3
"""
Task 1 extras, built on task1_rerun.py (keep both files in the same folder).

Adds
  1. "tft": a Temporal Fusion Transformer style classifier (simplified, see TFTLite).
  2. Robustness test across conditions: leave-one-study-out. gaitpdb has three studies
     (Ga, Ju, Si) recorded with different protocols and groups, so training on two
     and testing on the third measures how well a model transfers.

Run
  python task1_extra.py --data gait --pos 1,4,8 --sets S3 --mode cv   --models tft --out tft_S3
  python task1_extra.py --data gait --pos 1,4,8 --sets S3 --mode loso --out loso_S3
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.model_selection import GroupKFold

import task1_rerun as base


# ----------------------------------------------------------------------------
# Temporal Fusion Transformer style model
# ----------------------------------------------------------------------------
class GRN(nn.Module):
    """Gated residual network: the TFT building block."""

    def __init__(self, d_in, d_h, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_h)
        self.gate = nn.Linear(d_h, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc2(Fn.elu(self.fc1(x)))
        h = self.drop(h)
        return self.norm(self.skip(x) + Fn.glu(self.gate(h), dim=-1))


class TFTLite(nn.Module):
    """TFT components for classification of a multichannel window:
    per-channel embedding, GRN per channel, variable selection weights, LSTM encoder,
    gated skip, GRN enrichment, multi-head self-attention with gated skip, output GRN.
    Not included: static covariates and known-future inputs (not used in this task),
    and quantile forecasting heads (the task is classification)."""

    def __init__(self, C, T, nc=2, d=32, heads=4):
        super().__init__()
        self.emb_w = nn.Parameter(torch.randn(C, d) * 0.1)
        self.emb_b = nn.Parameter(torch.zeros(C, d))
        self.feat_grns = nn.ModuleList([GRN(d, d, d) for _ in range(C)])
        self.select = GRN(C, d, C)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate = nn.Linear(d, 2 * d)
        self.norm1 = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.attn_gate = nn.Linear(d, 2 * d)
        self.norm2 = nn.LayerNorm(d)
        self.out_grn = GRN(d, d, d)
        self.head = nn.Linear(d, nc)

    def forward(self, x):                                   # x: (B, T, C)
        emb = x.unsqueeze(-1) * self.emb_w + self.emb_b     # (B, T, C, d)
        emb = torch.stack([g(emb[:, :, i]) for i, g in enumerate(self.feat_grns)], dim=2)
        w = torch.softmax(self.select(x), dim=-1)           # (B, T, C) variable selection
        z = (emb * w.unsqueeze(-1)).sum(2)                  # (B, T, d)
        h, _ = self.lstm(z)
        h = self.norm1(z + Fn.glu(self.lstm_gate(h), dim=-1))
        h = self.enrich(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm2(h + Fn.glu(self.attn_gate(a), dim=-1))
        h = self.out_grn(h)
        return self.head(h.mean(1))


base.MODELS["tft"] = TFTLite
base.LR["tft"] = 1e-3


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pos", default="", help="1-based heel,mid,toe (as used in the main run)")
    ap.add_argument("--mode", choices=["cv", "loso"], default="cv",
                    help="cv = subject-wise k-fold, loso = leave one study out")
    ap.add_argument("--models", default="cnn,lstm,cnnlstm,transformer,tft")
    ap.add_argument("--sets", default="S3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--ds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="extra")
    args = ap.parse_args()

    recs = base.list_records(args.data)
    if not recs:
        sys.exit(f"No gaitpdb files found in {args.data}")
    arrays, recs = base.load_all(recs)
    cents = base.stance_centroids(arrays)
    heel, mid, toe, _ = base.auto_positions(cents)
    if args.pos:
        heel, mid, toe = [int(v) - 1 for v in args.pos.split(",")]
    print(f"Sensor positions (1-based): heel={heel+1}, mid={mid+1}, toe={toe+1}")

    sets = {"S2": [heel, toe], "S3": [heel, mid, toe], "S8": list(range(8))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   mode: {args.mode}")
    models = [m.strip() for m in args.models.split(",")]
    rows = []
    for sname in [s.strip() for s in args.sets.split(",")]:
        X, y, g = base.make_windows(arrays, recs, sets[sname], args.win, args.ds)
        study = np.array([s[:2] for s in g])
        print(f"\n=== {sname}: {len(X)} windows, {X.shape[2]} channels ===")
        if args.mode == "cv":
            splits = [(f"fold{i}", tr, te)
                      for i, (tr, te) in enumerate(GroupKFold(args.folds).split(X, y, g))]
        else:
            splits = []
            for held in sorted(set(study)):
                te = np.where(study == held)[0]
                tr = np.where(study != held)[0]
                print(f"  hold out {held}: train {len(tr)} windows, test {len(te)} windows, "
                      f"{100*y[te].mean():.0f}% Parkinson's in test")
                splits.append((f"hold_{held}", tr, te))
        for tag, tr, te in splits:
            for k, mname in enumerate(models):
                r = base.train_eval(mname, X[tr], y[tr], X[te], y[te], g[te], args, device,
                                    args.seed + k)
                r.update(sensors=sname, model=mname, split=tag)
                rows.append(r)
                print(f"  {tag:9s} {mname:12s} acc={r['acc']:.3f} f1={r['f1']:.3f} "
                      f"auc={r['auc']:.3f} subj={r['subj_acc']:.3f} ({r['train_s']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}_per_split.csv", index=False)
    agg = (df.groupby(["sensors", "model"])
             .agg(acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean"),
                  auc=("auc", "mean"), subj_acc=("subj_acc", "mean"),
                  params=("params", "first"), infer_ms=("infer_ms", "mean"))
             .round(3).reset_index())
    agg.to_csv(f"{args.out}_summary.csv", index=False)
    with open(f"{args.out}_summary.md", "w") as f:
        f.write(agg.to_markdown(index=False))
    if args.mode == "loso":
        piv = df.pivot_table(index=["sensors", "model"], columns="split",
                             values=["f1", "auc"]).round(3)
        piv.to_csv(f"{args.out}_by_heldout_study.csv")
        print("\nPer held-out study:\n" + piv.to_string())
    with open(f"{args.out}_config.json", "w") as f:
        json.dump(dict(vars(args), heel=heel + 1, mid=mid + 1, toe=toe + 1), f, indent=2)
    print("\n" + agg.to_string(index=False))
    print(f"\nSaved {args.out}_summary.csv / .md / _per_split.csv")


if __name__ == "__main__":
    main()
