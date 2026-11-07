"""Evaluates two cascades (the app uses Google -> DINOv2)  used in the app on every stress condition:
   1) Google Gesture Recognizer when it is sure (score >= 0.5)
   2) else our landmark MLP when MediaPipe finds a hand
   3) else DINOv2-reg + env-aug on the whole image
python src/eval_cascade.py   -> results/cascade_metrics.json
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from common import CLASSES, RESULTS, STRESS_DIR, WEIGHTS, list_folder
from landmarks import gesture
from stress import CONDITIONS
from train_landmarks import LandmarkMLP, feats

d = np.load(os.path.join(RESULTS, "landmarks_cache.npz"))
z = np.load(os.path.join(RESULTS, "preds.npz"))
mlp = LandmarkMLP(); mlp.load_state_dict(torch.load(os.path.join(WEIGHTS, "landmark_mlp.pt"))); mlp.eval()
out = {}
for c in CONDITIONS:
    items = list_folder(os.path.join(STRESS_DIR, c))
    y = np.array([l for _, l in items])
    X, ok = feats(d[f"cond_{c}_pts"], d[f"cond_{c}_hand"])
    with torch.no_grad():
        pl = mlp(torch.tensor(X)).argmax(1).numpy()
    pd = z[f"dino_env|{c}"][1]
    g = [gesture(Image.open(p).convert("RGB"))[0] for p, _ in items]
    gi = np.array([CLASSES.index(x) if x else -1 for x in g])
    used_g = gi >= 0
    pred = np.where(used_g, gi, np.where(ok, pl, pd))
    pred_gd = np.where(used_g, gi, pd)  # Google if sure, else DINOv2 (the app's Auto)
    out[c] = {"google_answered": float(used_g.mean()),
              "google_acc_when_answered": float((gi[used_g] == y[used_g]).mean()) if used_g.any() else None,
              "cascade_acc": float(accuracy_score(y, pred)),
              "cascade_macro_f1": float(f1_score(y, pred, average="macro")),
              "google_then_dino_acc": float(accuracy_score(y, pred_gd)),
              "google_then_dino_macro_f1": float(f1_score(y, pred_gd, average="macro")),
              "dino_env_acc": float(accuracy_score(y, pd))}
    print(c, {k: (round(v, 3) if v is not None else None) for k, v in out[c].items()}, flush=True)
st = [c for c in CONDITIONS if c != "clean"]
out["stress_mean"] = {k: float(np.mean([out[c][k] for c in st])) for k in ("google_answered", "cascade_acc", "cascade_macro_f1",
                                                                       "google_then_dino_acc", "google_then_dino_macro_f1", "dino_env_acc")}
json.dump(out, open(os.path.join(RESULTS, "cascade_metrics.json"), "w"), indent=1)
print("stress mean", out["stress_mean"])
