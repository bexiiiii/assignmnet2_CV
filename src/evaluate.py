"""Evaluates all trained models on the clean test split and on every
synthetic condition, measures latency, explanation focus and retrieval P@k,
and writes all figures/tables into results/.

python src/evaluate.py
"""
import json
import os
import pickle
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common import (CLASSES, EXTRA_DIR, RESULTS, STRESS_DIR, WEIGHTS,
                    get_device, list_folder, list_split, to_tensor)
from enhance import refine
from explain import attention_rollout, embed_items, gradcam, hand_focus
from models import DeepGestureCNN, DinoGesture
from stress import CONDITIONS, hand_mask

DEVICE = get_device()
NICE = {
    "logreg": "LogReg (A1 baseline)",
    "cnn": "Deep CNN (A1)",
    "cnn_env": "Deep CNN + env-aug",
    "dino": "DINOv2-S (A1 aug)",
    "dino_env": "DINOv2-S + env-aug",
    "dino_env_nobg": "DINOv2-S + env-aug, no bg-swap",
    "dino_env+refine": "DINOv2-S + env-aug + refine",
}


def load_models():
    M = {}
    with open(os.path.join(WEIGHTS, "logreg.pkl"), "rb") as f:
        M["logreg"] = pickle.load(f)
    for n in ("cnn", "cnn_env"):
        m = DeepGestureCNN().to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(WEIGHTS, f"{n}.pt"), map_location=DEVICE))
        M[n] = m.eval()
    for n in ("dino", "dino_env", "dino_env_nobg"):
        p = os.path.join(WEIGHTS, f"{n}.pt")
        if not os.path.exists(p):
            continue
        m = DinoGesture(pretrained=False).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        M[n] = m.eval()
    return M


@torch.no_grad()
def predict(name, model, imgs):
    """imgs: list of PIL. Returns (pred array, prob array)."""
    if name == "logreg":
        X = np.array([np.asarray(i.convert("L").resize((32, 32)), np.float32).ravel() / 255
                      for i in imgs])
        return model.predict(X), model.predict_proba(X)
    size = 128 if name.startswith("cnn") else 224
    P = []
    for s in range(0, len(imgs), 32):
        x = torch.stack([to_tensor(i, size) for i in imgs[s:s + 32]]).to(DEVICE)
        P.append(torch.softmax(model(x), 1).cpu().numpy())
    P = np.concatenate(P)
    return P.argmax(1), P


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"acc": accuracy_score(y, p), "precision": pr, "recall": rc, "macro_f1": f1}


def latency(name, model, img, runs=50):
    out = {}
    devices = ["cpu"] if name == "logreg" else ["cpu", str(DEVICE)]
    for dev in dict.fromkeys(devices):
        if name == "logreg":
            f = lambda: predict(name, model, [img])
        else:
            m = model.to(dev)
            size = 128 if name.startswith("cnn") else 224
            def f():
                with torch.no_grad():
                    x = to_tensor(img, size).unsqueeze(0).to(dev)
                    r = m(x).argmax(1).item()
                return r
        for _ in range(5):
            f()
        if dev == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            f()
        if dev == "mps":
            torch.mps.synchronize()
        out[dev] = (time.perf_counter() - t) / runs * 1000
        if name != "logreg":
            model.to(DEVICE)
    return out


def main():
    M = load_models()
    sets = {c: list_folder(os.path.join(STRESS_DIR, c)) for c in CONDITIONS}
    if os.path.isdir(EXTRA_DIR) and list_folder(EXTRA_DIR):
        sets["real_new_place"] = list_folder(EXTRA_DIR)
    res = {}
    preds_store = {}
    for cond, items in sets.items():
        imgs = [Image.open(p).convert("RGB") for p, _ in items]
        y = np.array([l for _, l in items])
        for name, model in M.items():
            p, _ = predict(name, model, imgs)
            res.setdefault(name, {})[cond] = scores(y, p)
            preds_store[(name, cond)] = (y, p)
        if "dino_env" in M:  # ArtAug-style refine applied at test time
            p, _ = predict("dino_env", M["dino_env"], [refine(i) for i in imgs])
            res.setdefault("dino_env+refine", {})[cond] = scores(y, p)
            preds_store[("dino_env+refine", cond)] = (y, p)
        print(cond, {k: round(v[cond]["acc"], 3) for k, v in res.items()})

    stress_conds = [c for c in sets if c != "clean"]
    for name in res:
        res[name]["stress_mean"] = {
            k: float(np.mean([res[name][c][k] for c in stress_conds if c in res[name]]))
            for k in ("acc", "macro_f1")}
    # save stage 1 right away (8 GB laptop: later stages can get OOM-killed)
    with open(os.path.join(RESULTS, "per_condition.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(RESULTS, "preds.npz"),
             **{f"{n}|{c}": np.stack(v) for (n, c), v in preds_store.items()})
    figures(res, sets, preds_store, None)
    del imgs
    import gc; gc.collect()
    extras(M, res, sets, preds_store)


def extras(M, res, sets, preds_store):
    # ---- latency ----
    sample = Image.open(list_split("test")[0][0]).convert("RGB")
    lat = {n: latency(n, m, sample) for n, m in M.items()}
    print("latency", lat, flush=True)

    # ---- explanation focus on the hand (clean test) ----
    focus = {"cnn": [], "cnn_env": [], "dino": [], "dino_env": []}
    for p, _ in list_split("test"):
        img = Image.open(p).convert("RGB")
        mask = hand_mask(p, img.resize((224, 224)))
        for n in focus:
            if n not in M:
                continue
            if n.startswith("cnn"):
                heat = gradcam(M[n], to_tensor(img, 128).unsqueeze(0).to(DEVICE))
            else:
                heat, _ = attention_rollout(M[n], to_tensor(img, 224).unsqueeze(0).to(DEVICE))
            focus[n].append(hand_focus(heat, mask))
    print("focus done", flush=True)
    focus_summary = {n: {"hand_share": float(np.mean([a for a, _ in v])),
                         "hand_area": float(np.mean([b for _, b in v]))}
                     for n, v in focus.items() if v}

    # ---- retrieval P@5 with DINOv2 embeddings ----
    retr = {}
    for n in ("dino", "dino_env"):
        if n not in M:
            continue
        tr, te = list_split("train"), list_split("test")
        bank = embed_items(M[n], tr, DEVICE)
        q = embed_items(M[n], te, DEVICE)
        ytr = torch.tensor([l for _, l in tr]); yte = torch.tensor([l for _, l in te])
        idx = (q @ bank.T).topk(5, dim=-1).indices
        retr[n] = {"P@1": float((ytr[idx[:, :1]] == yte[:, None]).float().mean()),
                   "P@5": float((ytr[idx] == yte[:, None]).float().mean())}
        torch.save({"bank": bank, "items": tr}, os.path.join(RESULTS, f"bank_{n}.pt"))
        print("retrieval", n, retr[n], flush=True)

    env = {"python": platform.python_version(), "torch": torch.__version__,
           "machine": platform.machine(), "system": platform.platform(),
           "accelerator": str(DEVICE)}
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"per_condition": res, "latency_ms": lat, "hand_focus": focus_summary,
                   "retrieval": retr, "environment": env}, f, indent=1)

    print(json.dumps({"latency": lat, "focus": focus_summary, "retrieval": retr}, indent=1))


def figures(res, sets, preds, lat):
    conds = list(sets)
    show = [n for n in ("logreg", "cnn", "dino", "dino_env", "dino_env+refine") if n in res]
    # bar chart accuracy by condition
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(show)
    for i, n in enumerate(show):
        ax.bar(np.arange(len(conds)) + i * w, [res[n][c]["acc"] * 100 for c in conds],
               w, label=NICE[n])
    ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=9)
    ax.set_ylabel("Test accuracy, %"); ax.set_ylim(0, 105)
    ax.axhline(100 / 3, ls=":", c="gray", lw=1); ax.text(len(conds) - 0.6, 35, "chance", color="gray", fontsize=8)
    ax.legend(fontsize=8, ncol=3, loc="lower left"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "acc_by_condition.png"), dpi=160); plt.close()

    # confusion matrices: CNN vs DINOv2-env on the worst condition for CNN
    worst = min([c for c in conds if c != "clean"], key=lambda c: res["cnn"][c]["acc"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, n in zip(axes, ("cnn", "dino_env")):
        y, p = preds[(n, worst)]
        cm = confusion_matrix(y, p, labels=range(3))
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{NICE[n]}\n({worst})", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "cm_worst_condition.png"), dpi=160); plt.close()

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for n in ("cnn", "cnn_env", "dino", "dino_env"):
        hp = os.path.join(RESULTS, f"history_{n}.json")
        if not os.path.exists(hp):
            continue
        h = json.load(open(hp))
        e = range(1, len(h["train_loss"]) + 1)
        l, = axes[0].plot(e, h["train_loss"], label=f"{NICE[n]} train")
        axes[0].plot(e, h["val_loss"], "--", c=l.get_color(), label=f"{NICE[n]} val")
        axes[1].plot(e, np.array(h["val_acc"]) * 100, c=l.get_color(), label=NICE[n])
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=6)
    axes[1].set_title("Validation accuracy, %"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=7)
    for a in axes: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "training_curves.png"), dpi=160); plt.close()


def resume():
    """python src/evaluate.py --resume : reuse stage-1 results, run the rest."""
    import sys
    M = load_models()
    res = json.load(open(os.path.join(RESULTS, "per_condition.json")))
    z = np.load(os.path.join(RESULTS, "preds.npz"))
    preds_store = {tuple(k.split("|")): (z[k][0], z[k][1]) for k in z.files}
    sets = {c: None for c in res["cnn"] if c != "stress_mean"}
    figures(res, sets, preds_store, None)
    extras(M, res, sets, preds_store)


if __name__ == "__main__":
    import sys
    resume() if "--resume" in sys.argv else main()
