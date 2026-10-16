# ONE Colab cell. Run it after your Task 1 cells (task1_rerun.py must be in the Colab folder,
# gaitpdb files in the "gaitpdb" folder). It trains the final CNN on ALL subjects and downloads
# two files: cnn_s3.npz and cnn_s3_meta.json
import json, numpy as np, torch, torch.nn as nn
import task1_rerun as base

DATA = "gaitpdb"
EPOCHS, BATCH, LR, SEED = 15, 64, 1e-3, 0

recs = base.list_records(DATA)
arrays, recs = base.load_all(recs)
heel, mid, toe, _ = base.auto_positions(base.stance_centroids(arrays))
print("sensor positions (1-based): heel", heel + 1, "mid", mid + 1, "toe", toe + 1)

X, y, g = base.make_windows(arrays, recs, [heel, mid, toe], 200, 2)   # channels: L heel/mid/toe, R heel/mid/toe
print("windows:", X.shape, "parkinson share:", round(float(y.mean()), 3))

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = base.CNN(X.shape[2], X.shape[1]).to(dev)
Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev)
cnt = np.bincount(y, minlength=2).astype(np.float32)
lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (2 * cnt), device=dev))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
for ep in range(EPOCHS):
    model.train(); tot = 0.0
    perm = torch.randperm(len(Xt), device=dev)
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        if len(idx) < 2: continue
        opt.zero_grad(); loss = lossf(model(Xt[idx]), yt[idx]); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item() * len(idx)
    sched.step(); print(f"epoch {ep + 1}/{EPOCHS}  loss {tot / len(Xt):.4f}")

model.eval().cpu()
with torch.no_grad():
    p = torch.softmax(model(torch.from_numpy(X[:2000])), 1)[:, 1].numpy()
print("training-set mean score  Parkinson's windows:", round(float(p[y[:2000] == 1].mean()), 3),
      " healthy windows:", round(float(p[y[:2000] == 0].mean()), 3), "(sanity check only, not accuracy)")

sd = {k: v.numpy() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
np.savez("cnn_s3.npz", **sd)
json.dump({"positions_1based": {"heel": heel + 1, "mid": mid + 1, "toe": toe + 1},
           "channels": ["L_heel", "L_mid", "L_toe", "R_heel", "R_mid", "R_toe"],
           "epochs": EPOCHS, "seed": SEED, "classes": ["healthy", "parkinsons"],
           "windows_train": int(len(y)), "trained_on": "PhysioNet gaitpdb, all subjects",
           "validation": "5-fold subject-wise macro F1 0.759 (Task 1); leave-one-study-out 0.706"},
          open("cnn_s3_meta.json", "w"), indent=1)
from google.colab import files
files.download("cnn_s3.npz"); files.download("cnn_s3_meta.json")
