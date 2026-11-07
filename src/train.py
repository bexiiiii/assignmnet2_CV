"""Trains every model used in the report.

  logreg          A1 baseline (32x32 gray pixels)
  cnn             A1 deep CNN, A1 augmentation
  cnn_env         A1 CNN + our environment augmentation   (ablation)
  dino            DINOv2 ViT-S/14, A1 augmentation only    (ablation)
  dino_env        DINOv2 + environment augmentation        (main model)
  dino_env_nobg   DINOv2 + env aug but WITHOUT background swap (ablation)

python src/train.py            # all
python src/train.py dino_env   # one
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from common import (CLASSES, RESULTS, WEIGHTS, GestureDS, get_device,
                    list_split, seed_all)
from models import DeepGestureCNN, DinoGesture, count_params
from stress import EnvAugment


def a1_augment(img, path=None):
    """A1 augmentation: 20% h-flip + rotation +-15."""
    import random
    if random.random() < 0.2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)


def flat_gray(items):
    X = [np.asarray(Image.open(p).convert("L").resize((32, 32)), np.float32).ravel() / 255
         for p, _ in items]
    return np.array(X), np.array([y for _, y in items])


def train_logreg():
    X, y = flat_gray(list_split("train"))
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    with open(os.path.join(WEIGHTS, "logreg.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print("logreg done")


def run_epochs(model, name, train_ds, val_ds, epochs, opt, device, bs=16):
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    vl = DataLoader(val_ds, batch_size=32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best, t0 = -1, time.time()
    for ep in range(epochs):
        model.train()
        L = C = N = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            L += loss.item() * len(y); C += (out.argmax(1) == y).sum().item(); N += len(y)
        sched.step()
        model.eval()
        vL = vC = vN = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vL += crit(out, y).item() * len(y); vC += (out.argmax(1) == y).sum().item(); vN += len(y)
        hist["train_loss"].append(L / N); hist["train_acc"].append(C / N)
        hist["val_loss"].append(vL / vN); hist["val_acc"].append(vC / vN)
        print(f"[{name}] ep {ep+1:02d}/{epochs} loss {L/N:.3f} acc {C/N:.3f} | "
              f"val loss {vL/vN:.3f} acc {vC/vN:.3f}")
        if vC / vN >= best:  # keep last best (ties -> later epoch)
            best = vC / vN
            torch.save(model.state_dict(), os.path.join(WEIGHTS, f"{name}.pt"))
    hist["train_time_s"] = time.time() - t0
    hist["params_total"] = count_params(model)
    hist["params_trainable"] = count_params(model, trainable=True)
    with open(os.path.join(RESULTS, f"history_{name}.json"), "w") as f:
        json.dump(hist, f, indent=1)


def train_cnn(name, env):
    seed_all()
    device = get_device()
    aug = EnvAugment(size=128, use_bg=True, seed=1) if env else a1_augment
    tr = GestureDS(list_split("train"), size=128, augment=aug)
    va = GestureDS(list_split("val"), size=128)
    m = DeepGestureCNN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    run_epochs(m, name, tr, va, epochs=20 if env else 15, opt=opt, device=device)


def train_dino(name, env, use_bg=True):
    seed_all()
    device = get_device()
    aug = EnvAugment(size=224, use_bg=use_bg, seed=1) if env else a1_augment
    tr = GestureDS(list_split("train"), size=224, augment=aug)
    va = GestureDS(list_split("val"), size=224)
    m = DinoGesture().to(device)
    opt = torch.optim.AdamW([
        {"params": [p for p in m.backbone.parameters() if p.requires_grad], "lr": 2e-5},
        {"params": m.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.05)
    run_epochs(m, name, tr, va, epochs=10, opt=opt, device=device)


JOBS = {
    "logreg": train_logreg,
    "cnn": lambda: train_cnn("cnn", env=False),
    "cnn_env": lambda: train_cnn("cnn_env", env=True),
    "dino": lambda: train_dino("dino", env=False),
    "dino_env": lambda: train_dino("dino_env", env=True),
    "dino_env_nobg": lambda: train_dino("dino_env_nobg", env=True, use_bg=False),
}

if __name__ == "__main__":
    for j in (sys.argv[1:] or JOBS):
        JOBS[j]()
