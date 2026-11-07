"""Hand gesture recognition - Assignment 2 demo (web page + small JSON API).

    .venv/bin/python app.py      ->  http://127.0.0.1:7860

The page (web/index.html) asks for consent first; every API call checks the
X-Consent header as well. Photos are processed in memory and never written
to disk.

API
  POST /api/predict   photo -> class probabilities, model used, latency
                      (now + running average), device, browser, and four
                      explanations: hand landmarks, attention / Grad-CAM,
                      SAM 2.1 hand mask, closest training images
  POST /api/stress    photo -> 8 SYNTHETIC conditions, three models
  POST /api/clip      photo -> 16-frame SYNTHETIC clip + per-frame predictions
  GET  /api/results   numbers from the report
"""
import base64
import io
import json
import os
import platform
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from clips import CLIP_ENVS, make_clip, write_mp4
from common import CLASSES, RESULTS, WEIGHTS, get_device, list_split, to_tensor
from enhance import refine
from explain import attention_rollout, gradcam, overlay
from landmarks import bbox as lm_bbox
from landmarks import detect, draw, features, gesture
from models import DeepGestureCNN, DinoGesture
from stress import CONDITIONS, apply_condition, mask_from_image, sam2_box
from train_landmarks import LandmarkMLP

DEVICE = get_device()
ENGINE_NAME = {"google": "MediaPipe Gesture Recognizer (Google, pretrained)", "landmark": "hand landmarks + our MLP",
               "dino": "DINOv2-reg + env-aug", "cnn": "A1 CNN"}

# ------------------------------------------------------------------ models
cnn = DeepGestureCNN().to(DEVICE)
cnn.load_state_dict(torch.load(os.path.join(WEIGHTS, "cnn.pt"), map_location=DEVICE))
cnn.eval()
dino = DinoGesture(pretrained=False).to(DEVICE)
dino.load_state_dict(torch.load(os.path.join(WEIGHTS, "dino_env.pt"), map_location=DEVICE))
dino.eval()
lmlp = LandmarkMLP()
lmlp.load_state_dict(torch.load(os.path.join(WEIGHTS, "landmark_mlp.pt"), map_location="cpu"))
lmlp.eval()
_bank = torch.load(os.path.join(RESULTS, "bank_dino_env.pt"))
BANK, BANK_ITEMS = _bank["bank"].to(DEVICE), _bank["items"]
# landmark "bank": features of the training frames where MediaPipe found a hand,
# so the neighbour gallery can follow the model that actually made the decision
_lm = np.load(os.path.join(RESULTS, "landmarks_cache.npz"))
_tr_items = list_split("train")
LM_FEATS, LM_ITEMS = [], []
for _i, (_p, _y) in enumerate(_tr_items):
    if not np.isnan(_lm["split_train_pts"][_i, 0, 0]):
        LM_FEATS.append(features(_lm["split_train_pts"][_i], str(_lm["split_train_hand"][_i])))
        LM_ITEMS.append((_p, _y))
LM_FEATS = np.stack(LM_FEATS)
LM_MU, LM_SD = LM_FEATS.mean(0), LM_FEATS.std(0) + 1e-6
LM_Z = (LM_FEATS - LM_MU) / LM_SD
LAT = {k: [] for k in ("cnn", "dino", "landmark", "google")}
SYSTEM = f"{platform.machine()}, {platform.system()} {platform.release()}, torch {torch.__version__}, {DEVICE}"


# ------------------------------------------------------------------ helpers
def square(img, size=448):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size))


def b64(img, fmt="JPEG"):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt, quality=88)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def watermark(img, text="SYNTHETIC"):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, img.size[1] - 16, 82, img.size[1]], fill=(190, 30, 30))
    d.text((4, img.size[1] - 14), text, fill="white")
    return img


def browser_of(ua):
    for name, pat in (("Edge", r"Edg/([\d.]+)"), ("Opera", r"OPR/([\d.]+)"), ("Chrome", r"Chrome/([\d.]+)"),
                      ("Firefox", r"Firefox/([\d.]+)"), ("Safari", r"Version/([\d.]+).*Safari")):
        m = re.search(pat, ua or "")
        if m:
            return f"{name} {m.group(1).split('.')[0]}"
    return "unknown"


def timed(fn):
    t = time.perf_counter()
    out = fn()
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    return out, (time.perf_counter() - t) * 1000


@torch.no_grad()
def p_cnn(img):
    return torch.softmax(cnn(to_tensor(img, 128).unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy()


@torch.no_grad()
def p_dino(img):
    return torch.softmax(dino(to_tensor(img, 224).unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy()


@torch.no_grad()
def p_landmark(pts, hand):
    return torch.softmax(lmlp(torch.tensor(features(pts, hand))[None]), 1)[0].numpy()


def classify(img, engine, record=True):
    """-> probs (or None), engine actually used, latency ms, landmarks (or None)."""
    pts = None
    if engine == "auto":  # Google's gesture model when it is sure, otherwise DINOv2
        (g, score), t = timed(lambda: gesture(img))
        if g is not None:
            prob = np.full(3, (1 - score) / 2, np.float32)
            prob[CLASSES.index(g)] = score
            if record:
                LAT["google"].append(t)
            return prob, "google", t, None
    if engine == "landmark":
        (pts, hand), t_det = timed(lambda: detect(img))
        if pts is not None:
            prob, t = timed(lambda: p_landmark(pts, hand))
            if record:
                LAT["landmark"].append(t + t_det)
            return prob, "landmark", t + t_det, pts
        return None, "landmark", t_det, None
    fn, used = (p_cnn, "cnn") if engine == "cnn" else (p_dino, "dino")
    prob, t = timed(lambda: fn(img))
    if record:
        LAT[used].append(t)
    return prob, used, t, pts


async def read_photo(photo):
    try:
        return Image.open(io.BytesIO(await photo.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image.")


def check_consent(x_consent):
    if x_consent != "yes":
        raise HTTPException(403, "Consent is required before any photo is processed.")


# ------------------------------------------------------------------ app
app = FastAPI(title="Hand gesture recognition")


@app.post("/api/predict")
async def api_predict(request: Request, photo: UploadFile = File(...), engine: str = Form("auto"),
                      refine_dark: bool = Form(False), x_consent: str = Header(None)):
    check_consent(x_consent)
    if engine not in ("auto", "dino", "landmark", "cnn"):
        raise HTTPException(400, "unknown engine")
    img = square(await read_photo(photo))
    note = None
    if refine_dark:
        img, diag, log = refine(img, return_log=True)
        note = f"luminance {diag['luminance']:.2f}: {', '.join(log) or 'nothing to fix'}"
    prob, used, ms, pts = classify(img, engine)
    hand = None
    if pts is None:
        pts, hand = detect(img)  # for the landmark panel, SAM prompt and neighbours
    else:
        hand = detect(img)[1]
    small = img.resize((224, 224))
    arr = np.asarray(small)
    if used == "cnn":
        heat, heat_name = gradcam(cnn, to_tensor(small, 128).unsqueeze(0).to(DEVICE)), "Grad-CAM, A1 CNN"
    else:
        heat, _ = attention_rollout(dino, to_tensor(small, 224).unsqueeze(0).to(DEVICE))
        heat_name = ("attention rollout, DINOv2" if used == "dino"
                     else "DINOv2 attention, for comparison (not used for this answer)")
    att = overlay(arr, cv2.resize(heat, (224, 224)))
    if pts is not None:  # SAM 2.1 prompted by the landmark box, so a face is never prompted
        box = lm_bbox(pts, 224, 224, margin=0.15)
        palm = pts[[0, 5, 9, 13, 17], :2].mean(0) * 224
        mask, prompt = sam2_box(small, box, (int(palm[0]), int(palm[1]))), "landmark box"
    else:
        mask, prompt = mask_from_image(small), "U2-Net"
    mask_vis = (arr * 0.3 + np.stack([mask * 30, mask * 90, mask * 200], -1)).clip(0, 255).astype(np.uint8)
    if used in ("landmark", "google") and pts is not None:  # neighbours in hand-landmark space (same features the MLP sees)
        z = (features(pts, hand) - LM_MU) / LM_SD
        d = np.linalg.norm(LM_Z - z, axis=1)
        idx = np.argsort(d)[:6]
        neighbours = [{"img": b64(Image.open(LM_ITEMS[i][0]).resize((112, 112))),
                       "label": CLASSES[LM_ITEMS[i][1]], "sim": round(float(1 / (1 + d[i] / 10)), 3)} for i in idx]
        nn_space = "hand-landmark features"
    else:
        with torch.no_grad():
            q = torch.nn.functional.normalize(dino.embed(to_tensor(small, 224).unsqueeze(0).to(DEVICE)), dim=-1)
        sims, idx = (q @ BANK.T)[0].topk(6)
        neighbours = [{"img": b64(Image.open(BANK_ITEMS[i][0]).resize((112, 112))),
                       "label": CLASSES[BANK_ITEMS[i][1]], "sim": round(s, 3)}
                      for s, i in zip(sims.tolist(), idx.tolist())]
        nn_space = "DINOv2 embedding, cosine similarity"
    return {
        "probs": None if prob is None else {c: float(p) for c, p in zip(CLASSES, prob)},
        "label": None if prob is None else CLASSES[int(np.argmax(prob))],
        "used": ENGINE_NAME[used], "fallback": engine == "auto" and used == "dino",
        "google_note": ("Google's model gives only the top score; the other two are split evenly"
                        if used == "google" else None),
        "latency_ms": round(ms, 1),
        "latency_avg_ms": round(float(np.mean(LAT[used])), 1) if LAT[used] else round(ms, 1),
        "latency_n": len(LAT[used]),
        "device": SYSTEM, "browser": browser_of(request.headers.get("user-agent")), "refine": note,
        "explain": {"input": b64(small), "landmarks": b64(draw(small, pts)) if pts is not None else None,
                    "attention": b64(att), "attention_name": heat_name,
                    "mask": b64(mask_vis), "mask_prompt": prompt},
        "neighbours": neighbours, "nn_space": nn_space,
    }


@app.post("/api/stress")
async def api_stress(photo: UploadFile = File(...), x_consent: str = Header(None)):
    check_consent(x_consent)
    img = square(await read_photo(photo), 224)
    mask = mask_from_image(img)
    rng = np.random.default_rng(7)
    rows = []
    for cond in CONDITIONS:
        s = apply_condition(img, cond, rng, mask=mask)
        pa, used, _, _ = classify(s, "auto", record=False)
        pc, pd = p_cnn(s), p_dino(s)
        rows.append({"condition": cond, "img": b64(watermark(s) if cond != "clean" else s),
                     "cnn": [CLASSES[pc.argmax()], float(pc.max())],
                     "dino": [CLASSES[pd.argmax()], float(pd.max())],
                     "auto": [CLASSES[pa.argmax()], float(pa.max()), {"google": "Google", "landmark": "landmarks"}.get(used, "DINOv2")]})
    return {"rows": rows}


@app.post("/api/clip")
async def api_clip(photo: UploadFile = File(...), setting: str = Form("new_room_pan"),
                   x_consent: str = Header(None)):
    check_consent(x_consent)
    if setting not in CLIP_ENVS:
        raise HTTPException(400, "unknown setting")
    img = square(await read_photo(photo), 224)
    frames = make_clip(img, mask_from_image(img), setting, np.random.default_rng(11))
    with tempfile.TemporaryDirectory() as td:
        path = write_mp4(frames, os.path.join(td, "clip.mp4"))
        video = "data:video/mp4;base64," + base64.b64encode(open(path, "rb").read()).decode()
    series = {"A1 CNN": [], "DINOv2-reg": [], "Auto": []}
    for f in frames:
        fi = Image.fromarray(f)
        series["A1 CNN"].append(int(p_cnn(fi).argmax()))
        series["DINOv2-reg"].append(int(p_dino(fi).argmax()))
        series["Auto"].append(int(classify(fi, "auto", record=False)[0].argmax()))
    return {"video": video, "classes": CLASSES, "series": series}


@app.get("/api/results")
def api_results():
    out = {}
    for name in ("metrics.json", "landmark_metrics.json", "clips_metrics.json"):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            out[name.split(".")[0]] = json.load(open(p))
    return JSONResponse(out)


@app.get("/figures/{name}")
def figure(name: str):
    allowed = {"acc_by_condition.png", "clip_strip.png", "cm_worst_condition.png", "training_curves.png",
               "explain_grid.png", "mask_compare.png", "conditions.png", "pipeline.png"}
    if name not in allowed:
        raise HTTPException(404)
    return FileResponse(os.path.join(RESULTS, name))


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")


def warm_up():
    """First call of each model is slow (lazy init); do it before serving."""
    img = square(Image.open(list_split("test")[0][0]).convert("RGB"))
    for e in ("auto", "dino", "cnn"):
        classify(img, e, record=False)
    sam2_box(img.resize((224, 224)), [40, 40, 180, 180], (110, 110))
    attention_rollout(dino, to_tensor(img.resize((224, 224)), 224).unsqueeze(0).to(DEVICE))


if __name__ == "__main__":
    warm_up()
    print("Open http://127.0.0.1:7860", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
