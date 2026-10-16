# ONE Colab cell: retrain the gait CNN so it ignores our insole's flaws, then download cnn_s3.npz + cnn_s3_meta.json.
# Run it after the Task 1 / export cells (task1_rerun.py in the Colab folder, gaitpdb files in the "gaitpdb" folder).
#
# What is simulated (applied to the PhysioNet force recordings BEFORE z-scoring, as our insole would distort them):
#   - clipping: a pad bottoms out, so its top is flat (heels most often)
#   - compression: the voltage divider bends the response (x ** gamma)
#   - weak pad: lower gain plus noise, so a badly seated pad is mostly noise
#   - random per-pad gain
#   - time stretch: stepping slower or faster than the original walking
# The model is trained on clean + distorted copies; it is checked subject-wise on clean AND distorted held-out subjects.
import json, os, sys, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import task1_rerun as base

DATA = os.environ.get("GAITPDB", "gaitpdb")
EPOCHS = int(os.environ.get("EPOCHS", 12)); K_AUG = int(os.environ.get("K_AUG", 2)); FOLDS = int(os.environ.get("FOLDS", 5))
BATCH, LR, SEED = 64, 1e-3, 0
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng_global = np.random.default_rng(SEED)

recs = base.list_records(DATA)
arrays, recs = base.load_all(recs)
heel, mid, toe, _ = base.auto_positions(base.stance_centroids(arrays))
SEL = [heel, mid, toe]                       # output channel order: heel, mid, toe (left), heel, mid, toe (right)
print("sensor positions (1-based): heel", heel + 1, "mid", mid + 1, "toe", toe + 1, "| recordings:", len(arrays))


def distort(a, rng, strength=1.0):
    """Return a distorted copy of one recording (columns as in gaitpdb: 0 time, 1-8 left, 9-16 right)."""
    a = a.copy()
    if rng.random() < 0.5 * strength:                      # time stretch (whole recording)
        f = rng.uniform(0.7, 1.5); n = int(len(a) * f)
        a = np.column_stack([np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a[:, j]) for j in range(a.shape[1])])
    for foot_off in (1, 9):
        for k, s in enumerate(SEL):                        # k: 0 heel, 1 mid, 2 toe
            c = foot_off + s
            x = np.clip(a[:, c], 0, None)
            p95 = np.percentile(x, 95) + 1e-6
            x = x * rng.uniform(0.5, 1.5)                  # random pad gain
            if rng.random() < 0.5 * strength:              # compression
                g = rng.uniform(0.5, 1.0); x = p95 * (x / p95) ** g
            if rng.random() < (0.6 if k == 0 else 0.2) * strength:   # clipping: heels bottom out most
                ceil = np.percentile(x, rng.uniform(70, 95)); x = np.minimum(x, ceil)
            if rng.random() < 0.15 * strength:             # weak, noisy pad
                x = x * rng.uniform(0.2, 0.6) + rng.normal(0, 0.05 * p95, len(x))
            a[:, c] = x
    return a


def windows(arrs, rs, aug_seed=None, copies=1):
    Xs, ys, gs = [], [], []
    for cidx in range(copies):
        rng = np.random.default_rng(None if aug_seed is None else aug_seed + cidx)
        src = arrs if aug_seed is None else [distort(a, rng) for a in arrs]
        X, y, g = base.make_windows(src, rs, SEL, 200, 2)
        Xs.append(X); ys.append(y); gs.append(g)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def train(X, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = base.CNN(X.shape[2], X.shape[1]).to(dev)
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev)
    cnt = np.bincount(y, minlength=2).astype(np.float32)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (2 * cnt), device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(model(Xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    return model.eval().cpu()


def prob(model, X):
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.from_numpy(X[i:i + 2048])), 1)[:, 1].numpy() for i in range(0, len(X), 2048)])


def f1(y, p): return f1_score(y, (p > 0.5).astype(int), average="macro")


# ---------- subject-wise check: baseline (clean only) versus robust (clean + distorted) ----------
subj = np.array([r["subject"] for r in recs]); usub = np.unique(subj)
gkf = GroupKFold(n_splits=min(FOLDS, len(usub)))
res = {"base": {"clean": [], "bent": [], "healthy_clean": [], "healthy_bent": []}, "robust": {"clean": [], "bent": [], "healthy_clean": [], "healthy_bent": []}}
sub_of_rec = subj
for fold, (tr_s, te_s) in enumerate(gkf.split(usub, groups=usub)):
    tr_set, te_set = set(usub[tr_s]), set(usub[te_s])
    tri = [i for i, s in enumerate(sub_of_rec) if s in tr_set]; tei = [i for i, s in enumerate(sub_of_rec) if s in te_set]
    A_tr, R_tr = [arrays[i] for i in tri], [recs[i] for i in tri]
    A_te, R_te = [arrays[i] for i in tei], [recs[i] for i in tei]
    Xc, yc, _ = windows(A_tr, R_tr)
    Xa, ya, _ = windows(A_tr, R_tr, aug_seed=100 + fold * 10, copies=K_AUG)
    Xte, yte, _ = windows(A_te, R_te)
    Xtb, ytb, _ = windows(A_te, R_te, aug_seed=999 + fold)      # distorted held-out subjects
    m_base = train(Xc, yc, SEED + fold)
    m_rob = train(np.concatenate([Xc, Xa]), np.concatenate([yc, ya]), SEED + fold)
    for name, m in (("base", m_base), ("robust", m_rob)):
        pc, pb = prob(m, Xte), prob(m, Xtb)
        res[name]["clean"].append(f1(yte, pc)); res[name]["bent"].append(f1(ytb, pb))
        res[name]["healthy_clean"].append(float(pc[yte == 0].mean())); res[name]["healthy_bent"].append(float(pb[ytb == 0].mean()))
    print(f"fold {fold + 1}: clean F1 base {res['base']['clean'][-1]:.3f} robust {res['robust']['clean'][-1]:.3f} | "
          f"distorted F1 base {res['base']['bent'][-1]:.3f} robust {res['robust']['bent'][-1]:.3f}")

summ = {n: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for n, d in res.items()}
print("\n=== subject-wise means over folds ===")
print("macro F1 on clean held-out subjects      : base", summ["base"]["clean"], " robust", summ["robust"]["clean"])
print("macro F1 on DISTORTED held-out subjects  : base", summ["base"]["bent"], " robust", summ["robust"]["bent"])
print("mean P(Parkinson) on HEALTHY, clean      : base", summ["base"]["healthy_clean"], " robust", summ["robust"]["healthy_clean"])
print("mean P(Parkinson) on HEALTHY, distorted  : base", summ["base"]["healthy_bent"], " robust", summ["robust"]["healthy_bent"])
print("(a good result: robust F1 on distorted data clearly above base, and healthy P(PD) staying low when distorted)\n")

# ---------- final model on ALL subjects, clean + distorted ----------
Xc, yc, _ = windows(arrays, recs)
Xa, ya, _ = windows(arrays, recs, aug_seed=7, copies=K_AUG)
X, y = np.concatenate([Xc, Xa]), np.concatenate([yc, ya])
print("final training windows:", X.shape, "parkinson share:", round(float(y.mean()), 3))
model = train(X, y, SEED)
p = prob(model, Xc[:2000])
print("sanity (training windows, not accuracy): PD", round(float(p[yc[:2000] == 1].mean()), 3), " healthy", round(float(p[yc[:2000] == 0].mean()), 3))

sd = {k: v.numpy() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
np.savez("cnn_s3.npz", **sd)
json.dump({"positions_1based": {"heel": heel + 1, "mid": mid + 1, "toe": toe + 1},
           "channels": ["L_heel", "L_mid", "L_toe", "R_heel", "R_mid", "R_toe"],
           "epochs": EPOCHS, "seed": SEED, "classes": ["healthy", "parkinsons"],
           "windows_train": int(len(y)), "trained_on": "PhysioNet gaitpdb, all subjects, with simulated clipping, compression, weak pads, gain and time stretch",
           "validation": f"{len(usub)}-subject {gkf.n_splits}-fold subject-wise macro F1: clean {summ['robust']['clean']}, distorted {summ['robust']['bent']} (clean-only model: clean {summ['base']['clean']}, distorted {summ['base']['bent']})"},
          open("cnn_s3_meta.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("cnn_s3.npz"); files.download("cnn_s3_meta.json")
except ImportError:
    print("saved cnn_s3.npz and cnn_s3_meta.json here")
