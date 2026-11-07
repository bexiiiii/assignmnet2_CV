"""Stage 2 of the evaluation, memory friendly (one model in RAM at a time).
Writes results/metrics.json step by step so a crash never loses finished parts.

python src/extras_light.py
"""
import gc
import json
import os
import platform
import pickle
import time

import numpy as np
import torch
from PIL import Image

from common import RESULTS, WEIGHTS, get_device, list_split, to_tensor
from explain import attention_rollout, embed_items, gradcam, hand_focus
from models import DeepGestureCNN, DinoGesture
from stress import hand_mask

DEV = get_device()
OUT = os.path.join(RESULTS, "metrics.json")


def save(key, value):
    m = json.load(open(OUT)) if os.path.exists(OUT) else {}
    m[key] = value
    json.dump(m, open(OUT, "w"), indent=1)
    print(key, "saved", flush=True)


def load(name):
    if name.startswith("cnn"):
        m = DeepGestureCNN()
    else:
        m = DinoGesture(pretrained=False)
    m.load_state_dict(torch.load(os.path.join(WEIGHTS, f"{name}.pt"), map_location="cpu"))
    return m.to(DEV).eval()


def lat_ms(model, size, runs=40):
    x = to_tensor(Image.open(list_split("test")[0][0]).convert("RGB"), size).unsqueeze(0).to(DEV)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t = time.perf_counter()
        for _ in range(runs):
            model(x).argmax(1).item()
    return (time.perf_counter() - t) / runs * 1000


def main():
    m = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "per_condition" not in m:
        save("per_condition", json.load(open(os.path.join(RESULTS, "per_condition.json"))))
    save("environment", {"python": platform.python_version(), "torch": torch.__version__,
                         "machine": platform.machine(), "system": platform.platform(),
                         "device_used": str(DEV)})
    test = list_split("test")
    masks = {p: hand_mask(p, Image.open(p).convert("RGB").resize((224, 224))) for p, _ in test}

    lat, focus, retr = {}, {}, {}
    with open(os.path.join(WEIGHTS, "logreg.pkl"), "rb") as f:  # our own file
        clf = pickle.load(f)
    X = np.asarray(Image.open(test[0][0]).convert("L").resize((32, 32)), np.float32).reshape(1, -1) / 255
    t = time.perf_counter()
    for _ in range(200):
        clf.predict(X)
    lat["logreg"] = (time.perf_counter() - t) / 200 * 1000

    for name in ("cnn", "cnn_env", "dino", "dino_env"):
        model = load(name)
        size = 128 if name.startswith("cnn") else 224
        lat[name] = lat_ms(model, size)
        vals = []
        for p, _ in test:
            x = to_tensor(Image.open(p).convert("RGB"), size).unsqueeze(0).to(DEV)
            heat = gradcam(model, x) if name.startswith("cnn") else attention_rollout(model, x)[0]
            vals.append(hand_focus(heat, masks[p]))
        focus[name] = {"hand_share": float(np.mean([a for a, _ in vals])),
                       "hand_area": float(np.mean([b for _, b in vals]))}
        if name.startswith("dino"):
            tr = list_split("train")
            bank = embed_items(model, tr, DEV)
            q = embed_items(model, test, DEV)
            ytr = torch.tensor([l for _, l in tr]); yte = torch.tensor([l for _, l in test])
            idx = (q @ bank.T).topk(5, dim=-1).indices
            retr[name] = {"P@1": float((ytr[idx[:, :1]] == yte[:, None]).float().mean()),
                          "P@5": float((ytr[idx] == yte[:, None]).float().mean())}
            torch.save({"bank": bank, "items": tr}, os.path.join(RESULTS, f"bank_{name}.pt"))
        save("latency_ms_" + str(DEV), lat)
        save("hand_focus", focus)
        save("retrieval", retr)
        del model
        gc.collect()
    print(json.dumps({"lat": lat, "focus": focus, "retr": retr}, indent=1))


if __name__ == "__main__":
    main()
