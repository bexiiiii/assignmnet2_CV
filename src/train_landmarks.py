"""Our own gesture classifier on MediaPipe hand landmarks + fusion with DINOv2.

1. extract landmarks for train/val/test and every stress condition (cached)
2. train a small MLP on the 78-d landmark features of the TRAIN frames
   (landmark-space augmentation: rotation, scale, jitter), keep best val epoch
3. evaluate: detection rate, accuracy on detected frames, and the FUSED system
   (landmark MLP when a hand is found, DINOv2 + env-aug otherwise)

python src/train_landmarks.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from common import CLASSES, RESULTS, STRESS_DIR, WEIGHTS, list_folder, list_split, seed_all
from landmarks import augment_pts, detect, features
from stress import CONDITIONS

CACHE = os.path.join(RESULTS, "landmarks_cache.npz")


class LandmarkMLP(nn.Module):
    def __init__(self, d=78, h=64, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


def extract(items):
    P, H = [], []
    for p, _ in items:
        pts, hand = detect(Image.open(p).convert("RGB"))
        P.append(pts if pts is not None else np.full((21, 3), np.nan, np.float32))
        H.append(hand or "")
    return np.stack(P), np.array(H)


def load_cache():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    data = {}
    sets = {f"split_{s}": list_split(s) for s in ("train", "val", "test")}
    sets.update({f"cond_{c}": list_folder(os.path.join(STRESS_DIR, c)) for c in CONDITIONS})
    for k, items in sets.items():
        P, H = extract(items)
        data[k + "_pts"], data[k + "_hand"] = P, H
        data[k + "_y"] = np.array([l for _, l in items])
        print(k, "detected", int((~np.isnan(P[:, 0, 0])).sum()), "/", len(items), flush=True)
    np.savez(CACHE, **data)
    return data


def feats(P, H):
    ok = ~np.isnan(P[:, 0, 0])
    X = np.zeros((len(P), 78), np.float32)
    for i in np.where(ok)[0]:
        X[i] = features(P[i], H[i])
    return X, ok


def main():
    seed_all()
    d = load_cache()
    rng = np.random.default_rng(0)
    # training set with landmark-space augmentation (x10)
    Ptr, Htr, ytr = d["split_train_pts"], d["split_train_hand"], d["split_train_y"]
    ok = ~np.isnan(Ptr[:, 0, 0])
    Xa, ya = [], []
    for i in np.where(ok)[0]:
        Xa.append(features(Ptr[i], Htr[i])); ya.append(ytr[i])
        for _ in range(10):
            Xa.append(features(augment_pts(Ptr[i], rng), Htr[i])); ya.append(ytr[i])
    Xa, ya = torch.tensor(np.stack(Xa)), torch.tensor(ya)
    Xv, okv = feats(d["split_val_pts"], d["split_val_hand"])
    Xv, yv = torch.tensor(Xv[okv]), torch.tensor(d["split_val_y"][okv])

    model = LandmarkMLP()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best = (-1, 1e9)
    for ep in range(80):
        model.train()
        perm = torch.randperm(len(Xa))
        L = C = 0
        for b in range(0, len(Xa), 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            out = model(Xa[idx]); loss = crit(out, ya[idx]); loss.backward(); opt.step()
            L += loss.item() * len(idx); C += (out.argmax(1) == ya[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            ov = model(Xv); vl = crit(ov, yv).item(); va = (ov.argmax(1) == yv).float().mean().item()
        hist["train_loss"].append(L / len(Xa)); hist["train_acc"].append(C / len(Xa))
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if (va, -vl) > (best[0], -best[1]):
            best = (va, vl)
            torch.save(model.state_dict(), os.path.join(WEIGHTS, "landmark_mlp.pt"))
        if ep % 10 == 9:
            print(f"[landmark] ep {ep+1} loss {L/len(Xa):.3f} acc {C/len(Xa):.3f} | val {vl:.3f} {va:.3f}", flush=True)
    hist["n_train_frames_detected"] = int(ok.sum())
    hist["n_train_after_aug"] = int(len(Xa))
    json.dump(hist, open(os.path.join(RESULTS, "history_landmark.json"), "w"), indent=1)

    # ---------------- evaluation + fusion with DINOv2 ----------------
    model.load_state_dict(torch.load(os.path.join(WEIGHTS, "landmark_mlp.pt")))
    model.eval()
    z = np.load(os.path.join(RESULTS, "preds.npz"))
    out = {}
    for c in CONDITIONS:
        X, okc = feats(d[f"cond_{c}_pts"], d[f"cond_{c}_hand"])
        y = d[f"cond_{c}_y"]
        with torch.no_grad():
            pl = model(torch.tensor(X)).argmax(1).numpy()
        pdino = z[f"dino_env|{c}"][1]
        assert (z[f"dino_env|{c}"][0] == y).all(), "order mismatch"
        fused = np.where(okc, pl, pdino)
        out[c] = {"detected": float(okc.mean()),
                  "landmark_acc_on_detected": float((pl[okc] == y[okc]).mean()) if okc.any() else None,
                  "landmark_only_acc": float(((pl == y) & okc).mean()),  # missed hand = wrong
                  "fused_acc": float(accuracy_score(y, fused)),
                  "fused_macro_f1": float(f1_score(y, fused, average="macro")),
                  "dino_env_acc": float(accuracy_score(y, pdino))}
        print(c, {k: (round(v, 3) if v is not None else None) for k, v in out[c].items()}, flush=True)
    st = [c for c in CONDITIONS if c != "clean"]
    out["stress_mean"] = {k: float(np.mean([out[c][k] for c in st]))
                          for k in ("detected", "landmark_only_acc", "fused_acc", "fused_macro_f1", "dino_env_acc")}
    json.dump(out, open(os.path.join(RESULTS, "landmark_metrics.json"), "w"), indent=1)
    print("stress mean", out["stress_mean"])


if __name__ == "__main__":
    main()
