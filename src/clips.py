"""Synthetic gesture CLIPS - laptop-sized stand-in for Step-Video-T2V.

The course row for gestures says: "Step-Video-T2V to generate short gesture
clips with varied backgrounds / camera motions". Step-Video-T2V needs ~80 GB
of GPU memory, so the clips are made from our own real frames: a smooth
random camera path (pan, zoom, roll, hand-held shake), motion blur that
follows the camera speed, a new background with parallax, and darkness.
Every clip is SYNTHETIC and carries a red label.

python src/clips.py     # writes results/clips/*.mp4, clip_strip.png, clips_metrics.json
"""
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from common import RESULTS, WEIGHTS, get_device, list_split, to_tensor
from stress import TEST_BG, hand_mask, make_background

CLIP_ENVS = ["handheld", "new_room_pan", "low_light_shake", "night_fast"]
N_FRAMES = 16
FPS = 8
SIZE = 224


def _path(rng, n, amp, speed):
    """Smooth random 1-D camera trajectory (two slow sines + jitter)."""
    t = np.linspace(0, 1, n)
    p = (amp * np.sin(2 * np.pi * (speed * t + rng.random()))
         + 0.5 * amp * np.sin(2 * np.pi * (2.3 * speed * t + rng.random())))
    return p + rng.normal(0, amp * 0.08, n)


def _motion_blur(img, dx, dy):
    L = int(min(25, np.hypot(dx, dy) * 1.5))
    if L < 3:
        return img
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1
    M = cv2.getRotationMatrix2D((L / 2 - .5, L / 2 - .5), np.degrees(np.arctan2(dy, dx)), 1)
    k = cv2.warpAffine(k, M, (L, L))
    return cv2.filter2D(img, -1, k / k.sum())


def make_clip(img, mask, env, rng, n=N_FRAMES):
    """img: PIL, mask: HxW float in [0,1] -> list of n uint8 frames."""
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), np.float32)
    m = cv2.resize(mask.astype(np.float32), (SIZE, SIZE))[..., None]
    fast = env == "night_fast"
    amp = {"handheld": 6, "new_room_pan": 18, "low_light_shake": 8, "night_fast": 22}[env]
    sp = 1.6 if fast else 0.8
    xs, ys = _path(rng, n, amp, sp), _path(rng, n, amp * 0.7, sp)
    zs = 1.05 + 0.08 * np.sin(np.linspace(0, np.pi, n) + rng.random())
    rs = _path(rng, n, 4 if fast else 2.5, sp)
    bg = None
    if env in ("new_room_pan", "night_fast"):
        bg = make_background(SIZE * 2, rng, TEST_BG[rng.integers(len(TEST_BG))])
    frames = []
    for i in range(n):
        dx = xs[i] - xs[i - 1] if i else 0.0
        dy = ys[i] - ys[i - 1] if i else 0.0
        if bg is not None:  # background moves at half speed -> parallax
            ox = int(np.clip(SIZE / 2 - xs[i] * 0.5, 0, SIZE))
            oy = int(np.clip(SIZE / 2 - ys[i] * 0.5, 0, SIZE))
            src = a * m + bg[oy:oy + SIZE, ox:ox + SIZE] * (1 - m)
        else:
            src = a
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rs[i], zs[i])
        M[:, 2] += (xs[i], ys[i])
        f = cv2.warpAffine(src.astype(np.float32), M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
        f = _motion_blur(f, dx, dy)
        if env in ("low_light_shake", "night_fast"):
            f = (np.clip(f, 0, 255) / 255.0) ** 1.6 * (0.22 if fast else 0.3) * 255
            f = f + rng.normal(0, 4, f.shape)
        f = np.clip(f, 0, 255).astype(np.uint8)
        if fast:
            ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 20])
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
        frames.append(f)
    return frames


def watermark(frame):
    im = Image.fromarray(frame)
    h = im.size[1]
    d = ImageDraw.Draw(im)
    d.rectangle([0, h - 16, 84, h], fill=(200, 30, 30))
    d.text((4, h - 14), "SYNTHETIC", fill="white")
    return np.asarray(im)


def write_mp4(frames, path, fps=FPS):
    """H.264 mp4 through ffmpeg so that browsers can play it."""
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(watermark(f)).save(os.path.join(td, f"{i:03d}.png"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", os.path.join(td, "%03d.png"), "-vf", "scale=448:448:flags=neighbor",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
    return path


@torch.no_grad()
def predict_frames(name, model, frames, dev):
    size = 128 if name.startswith("cnn") else 224
    x = torch.stack([to_tensor(Image.fromarray(f), size) for f in frames]).to(dev)
    return torch.softmax(model(x), 1).argmax(1).cpu().numpy()


def main():
    from models import DeepGestureCNN, DinoGesture
    dev = get_device()
    models = {}
    for n in ("cnn", "dino", "dino_env"):
        mdl = DeepGestureCNN() if n == "cnn" else DinoGesture(pretrained=False)
        mdl.load_state_dict(torch.load(os.path.join(WEIGHTS, f"{n}.pt"), map_location="cpu"))
        models[n] = mdl.to(dev).eval()
    out_dir = os.path.join(RESULTS, "clips")
    os.makedirs(out_dir, exist_ok=True)
    test = list_split("test")
    res = {n: {e: {"frame_acc": [], "clip_acc": [], "flips": []} for e in CLIP_ENVS} for n in models}
    strip = {}
    for idx, (p, y) in enumerate(test):
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
        mask = hand_mask(p, img)
        for e in CLIP_ENVS:
            frames = make_clip(img, mask, e, np.random.default_rng(5000 + idx * 7 + CLIP_ENVS.index(e)))
            if idx in (10, 40):
                write_mp4(frames, os.path.join(out_dir, f"SYN_{e}_{os.path.basename(p)[:-4]}.mp4"))
            if idx == 40:
                strip[e] = frames
            for n, mdl in models.items():
                pr = predict_frames(n, mdl, frames, dev)
                res[n][e]["frame_acc"].append(float((pr == y).mean()))
                res[n][e]["clip_acc"].append(float(np.bincount(pr, minlength=3).argmax() == y))
                res[n][e]["flips"].append(int((pr[1:] != pr[:-1]).sum()))
        if idx % 10 == 0:
            print("clips", idx, "/", len(test), flush=True)
    summary = {n: {e: {k: float(np.mean(v)) for k, v in d.items()} for e, d in r.items()}
               for n, r in res.items()}
    json.dump({"n_frames": N_FRAMES, "fps": FPS, "clips_per_env": len(test), "summary": summary},
              open(os.path.join(RESULTS, "clips_metrics.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(CLIP_ENVS), 6, figsize=(10, 6.8))
    for r, e in enumerate(CLIP_ENVS):
        for c, fi in enumerate(np.linspace(0, N_FRAMES - 1, 6).astype(int)):
            axes[r, c].imshow(watermark(strip[e][fi])); axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(f"{e}  (frame {fi})", fontsize=8, loc="left")
            else:
                axes[r, c].set_title(f"frame {fi}", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "clip_strip.png"), dpi=130); plt.close()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
