"""Explainability tools: Grad-CAM (CNN), attention rollout (DINOv2),
nearest-neighbour gallery (DINOv2 embeddings) and a 'hand focus' score."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------- Grad-CAM for the A1 CNN ----------------
def gradcam(cnn, x, target=None):
    """x: 1x3xHxW tensor on the model device. Returns HxW map in [0,1]."""
    # output of the last conv block (8x8 for a 128px input). A tensor hook is
    # used because the model has inplace ReLUs, which break module backward hooks.
    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.setdefault("g", g))

    h1 = cnn.features.register_forward_hook(fwd)
    cnn.eval()
    x = x.clone().requires_grad_(True)
    out = cnn(x)
    t = int(out.argmax(1)) if target is None else target
    cnn.zero_grad()
    out[0, t].backward()
    h1.remove()
    a, g = acts["a"][0], grads["g"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * a).sum(0)).detach().cpu().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
    return cam / (cam.max() + 1e-8)


# ---------------- Attention rollout for DINOv2 ----------------
def _enable_attn_capture(vit):
    for blk in vit.blocks:
        blk.attn.fused_attn = False  # so the softmax matrix is materialised
        if not getattr(blk.attn, "_hooked", False):
            blk.attn.attn_drop.register_forward_hook(
                lambda m, i, o: store_ref[0].append(o.detach()))
            blk.attn._hooked = True


store_ref = [[]]


def attention_rollout(dino, x, discard=0.0):
    """Abnar & Zuidema (2020) rollout. Returns HxW map in [0,1]."""
    vit = dino.backbone
    _enable_attn_capture(vit)
    store_ref[0] = []
    dino.eval()
    with torch.no_grad():
        logits = dino(x)
    mats = store_ref[0]
    n = mats[0].shape[-1]
    R = torch.eye(n, device=x.device)
    for A in mats:
        A = A[0].mean(0)  # average heads
        if discard > 0:
            flat = A.view(-1)
            k = int(flat.numel() * discard)
            idx = flat.topk(k, largest=False).indices
            flat[idx[idx != 0]] = 0
        A = A + torch.eye(n, device=x.device)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    cls = R[0, vit.num_prefix_tokens:]  # CLS row, patch tokens only
    side = int(round(cls.numel() ** 0.5))
    m = cls.reshape(side, side).cpu().numpy()
    m = cv2.resize(m, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_CUBIC)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m, logits


# ---------------- helpers ----------------
def overlay(img_rgb, heat, alpha=0.45):
    """img_rgb uint8 HxWx3, heat HxW [0,1] -> uint8 overlay."""
    h = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    h = cv2.resize(h, (img_rgb.shape[1], img_rgb.shape[0]))
    return (img_rgb * (1 - alpha) + h * alpha).astype(np.uint8)


def hand_focus(heat, mask):
    """Share of the explanation mass that falls on the hand (0..1).
    1.0 = model only looks at the hand, low = it looks at background."""
    mask = cv2.resize(mask.astype(np.float32), (heat.shape[1], heat.shape[0])) > 0.5
    return float(heat[mask].sum() / (heat.sum() + 1e-8)), float(mask.mean())


@torch.no_grad()
def embed_items(dino, items, device, size=224, bs=32):
    from common import GestureDS
    from torch.utils.data import DataLoader
    dl = DataLoader(GestureDS(items, size=size), batch_size=bs)
    E = []
    dino.eval()
    for x, _ in dl:
        E.append(F.normalize(dino.embed(x.to(device)), dim=-1).cpu())
    return torch.cat(E)


def nearest(query_emb, bank, k=5):
    sims = query_emb @ bank.T
    v, i = sims.topk(k, dim=-1)
    return v, i
