"""Extra figures for the report: pipeline diagram, condition strip,
explanation grid, nearest-neighbour gallery, refine before/after."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
from PIL import Image

from common import CLASSES, RESULTS, STRESS_DIR, WEIGHTS, get_device, list_folder, list_split, to_tensor
from enhance import refine
from explain import attention_rollout, gradcam, overlay
from models import DeepGestureCNN, DinoGesture
from stress import CONDITIONS, coarse_mask, hand_mask

DEV = get_device()


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.1); ax.axis("off")
    boxes = [
        (0.1, 1.9, "A1 frames\n450 real\n(train 315)", "#dbe9f6"),
        (2.2, 1.9, "Condition simulator\n(Step-Video-T2V idea)\ndark · blur · noise · zoom\nnew bg via SAM 2.1 mask", "#fde2c8"),
        (4.6, 1.9, "Refine (ArtAug idea)\ndiagnose → fix\ngamma · CLAHE · denoise", "#fde2c8"),
        (7.0, 1.9, "DINOv2-reg ViT-S/14\nfrozen 8 blocks\n+ 4 tuned + head", "#d5f0d5"),
        (9.3, 1.9, "thumbs_up / peace\n/ stop + prob.", "#eeeeee"),
        (7.0, 0.15, "Explain: attention rollout,\nGrad-CAM (A1 CNN), SAM 2\nhand mask, k-NN gallery", "#efe0f7"),
        (2.2, 0.15, "Stress test set\n8 conditions × 69\n(SYNTHETIC, held-out\nbackground styles)", "#fde2c8"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.9 if x < 9 else 1.6, 1.0,
                                    boxstyle="round,pad=0.05", fc=c, ec="#555"))
        ax.text(x + (0.95 if x < 9 else 0.8), y + 0.5, t, ha="center", va="center", fontsize=7.6)
    arrows = [((2.0, 2.4), (2.2, 2.4)), ((4.1, 2.4), (4.6, 2.4)), ((6.5, 2.4), (7.0, 2.4)),
              ((8.9, 2.4), (9.3, 2.4)), ((7.95, 1.9), (7.95, 1.15)), ((3.15, 1.9), (3.15, 1.15))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#333"))
    ax.text(5.55, 1.55, "training: random condition with p=0.7\n(train-only background styles)",
            ha="center", fontsize=7, style="italic", color="#444")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "pipeline.png"), dpi=170); plt.close()


def condition_strip():
    name = os.path.basename(list_split("test")[50][0])
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(14, 2.2))
    for ax, c in zip(axes, CONDITIONS):
        f = [p for p, _ in list_folder(os.path.join(STRESS_DIR, c)) if p.endswith(name)][0]
        ax.imshow(Image.open(f)); ax.set_title(c, fontsize=8); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "conditions.png"), dpi=150); plt.close()


def load():
    cnn = DeepGestureCNN().to(DEV)
    cnn.load_state_dict(torch.load(os.path.join(WEIGHTS, "cnn.pt"), map_location=DEV)); cnn.eval()
    dino = DinoGesture(pretrained=False).to(DEV)
    dino.load_state_dict(torch.load(os.path.join(WEIGHTS, "dino_env.pt"), map_location=DEV)); dino.eval()
    return cnn, dino


def explain_grid(cnn, dino):
    test = list_split("test")
    picks = [test[5], test[30], test[55]]
    rows = []
    for p, y in picks:
        rows.append((Image.open(p).convert("RGB").resize((224, 224)), p, y, "clean"))
    nb = list_folder(os.path.join(STRESS_DIR, "new_background"))
    q = nb[60]
    rows.append((Image.open(q[0]).convert("RGB"), test[60][0], q[1], "new background"))
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.4, 2.25 * len(rows)))
    for r, (img, p, y, tag) in enumerate(rows):
        a = np.asarray(img)
        xc = to_tensor(img, 128).unsqueeze(0).to(DEV)
        g = gradcam(cnn, xc)
        with torch.no_grad():
            pc = torch.softmax(cnn(xc), 1)[0].cpu().numpy()
        xd = to_tensor(img, 224).unsqueeze(0).to(DEV)
        rr, logit = attention_rollout(dino, xd)
        pd = torch.softmax(logit, 1)[0].cpu().numpy()
        m = hand_mask(p, Image.open(p).convert("RGB").resize((224, 224)))
        import cv2
        panels = [(a, f"input ({tag})\ntrue: {CLASSES[y]}"),
                  (overlay(a, cv2.resize(g, (224, 224))), f"A1 CNN Grad-CAM\n{CLASSES[pc.argmax()]} {pc.max()*100:.0f}%"),
                  (overlay(a, rr), f"DINOv2 rollout\n{CLASSES[pd.argmax()]} {pd.max()*100:.0f}%"),
                  ((a * 0.3 + np.stack([m * 40, m * 220, m * 90], -1)).clip(0, 255).astype(np.uint8), "SAM 2.1 hand mask")]
        for c, (im, t) in enumerate(panels):
            axes[r, c].imshow(im); axes[r, c].set_title(t, fontsize=7.5); axes[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "explain_grid.png"), dpi=150); plt.close()


def nn_gallery(dino):
    bank = torch.load(os.path.join(RESULTS, "bank_dino_env.pt"))
    B, items = bank["bank"].to(DEV), bank["items"]
    queries = [list_folder(os.path.join(STRESS_DIR, "low_light"))[10],
               list_folder(os.path.join(STRESS_DIR, "new_background"))[40]]
    fig, axes = plt.subplots(len(queries), 6, figsize=(11, 4))
    for r, (qp, qy) in enumerate(queries):
        img = Image.open(qp).convert("RGB")
        with torch.no_grad():
            e = torch.nn.functional.normalize(dino.embed(to_tensor(img, 224).unsqueeze(0).to(DEV)), dim=-1)
        s, idx = (e @ B.T)[0].topk(5)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"query (SYNTHETIC)\ntrue: {CLASSES[qy]}", fontsize=7.5)
        for c, (si, ii) in enumerate(zip(s.tolist(), idx.tolist()), 1):
            axes[r, c].imshow(Image.open(items[ii][0]))
            axes[r, c].set_title(f"{CLASSES[items[ii][1]]}  cos={si:.2f}", fontsize=7.5)
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "nn_gallery.png"), dpi=150); plt.close()


def refine_example():
    a = list_folder(os.path.join(STRESS_DIR, "low_light"))
    b = list_folder(os.path.join(STRESS_DIR, "night_combo"))
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
    for c, (p, y) in enumerate([a[3], a[30], b[60]]):
        img = Image.open(p).convert("RGB")
        out, d, log = refine(img, return_log=True)
        axes[0, c].imshow(img); axes[0, c].set_title(f"before  L={d['luminance']:.2f}", fontsize=8)
        axes[1, c].imshow(out); axes[1, c].set_title("after: " + ", ".join(log), fontsize=7)
        axes[0, c].axis("off"); axes[1, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "refine_example.png"), dpi=150); plt.close()


def mask_compare():
    test = list_split("test")
    picks = [test[30], test[5], test[55], test[60]]
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    for c, (p, _) in enumerate(picks):
        img = Image.open(p).convert("RGB").resize((224, 224))
        a = np.asarray(img)
        for r, (m, t) in enumerate([(coarse_mask(p, img), "U2-Net (prompt only)"),
                                    (hand_mask(p, img), "SAM 2.1 (final)")]):
            axes[r, c].imshow((a * 0.3 + np.stack([m * 40, m * 220, m * 90], -1)).clip(0, 255).astype(np.uint8))
            axes[r, c].set_title(t, fontsize=8); axes[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "mask_compare.png"), dpi=150); plt.close()


if __name__ == "__main__":
    mask_compare()
    pipeline_diagram()
    condition_strip()
    refine_example()
    cnn, dino = load()
    explain_grid(cnn, dino)
    nn_gallery(dino)
    print("figures done")
