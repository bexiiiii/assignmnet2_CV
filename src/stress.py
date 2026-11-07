"""Condition simulator ("stress-test generator").

Step-Video-T2V cannot run on a laptop (30B params, needs 80GB GPUs), so we
simulate the same kinds of shifts it would give us - darkness, camera motion,
sensor noise, new places - directly on our own frames. Every output of this
module is SYNTHETIC and is labelled as such in the app and report.
"""
import os

import cv2
import numpy as np
from PIL import Image

from common import ROOT

MASK_DIR = os.path.join(ROOT, "data", "masks")        # raw U2-Net output (coarse)
SAM_DIR = os.path.join(ROOT, "data", "masks_sam2")     # final SAM 2 masks
SAM_ID = "facebook/sam2.1-hiera-tiny"

CONDITIONS = [
    "clean",
    "low_light",
    "overexposed",
    "motion_blur",
    "noise_jpeg",
    "rotate_zoom_out",
    "new_background",
    "night_combo",
]


# ---------- hand mask: U2-Net (coarse prompt) -> SAM 2.1 (final) ----------
# SAM 2 needs a prompt (point / box) and does not know what a "hand" is.
# U2-Net gives a rough salient-object map without any prompt, so we use it
# only to place the prompt: a box around the rough mask + one positive point
# at the most "inner" pixel of it (distance-transform max = middle of palm).
# SAM 2 then draws the actual boundary, which fixes U2-Net's holes in the palm
# and the table patches between fingers.
_session = None
_sam = None


def _key(path):
    return path.replace(os.sep, "_").replace(":", "")[-120:] + ".png"


def coarse_mask(path, img):
    """U2-Net mask (cleaned), raw output cached on disk."""
    os.makedirs(MASK_DIR, exist_ok=True)
    cache = os.path.join(MASK_DIR, _key(path))
    if os.path.exists(cache):
        m = Image.open(cache).convert("L").resize(img.size)
        return _clean(np.asarray(m, dtype=np.float32) / 255.0)
    raw = _raw_mask(img)
    Image.fromarray((raw * 255).astype(np.uint8)).save(cache)
    return _clean(raw)


def hand_mask(path, img=None):
    """Final hand mask HxW in [0,1] (SAM 2 refined), cached as PNG."""
    os.makedirs(SAM_DIR, exist_ok=True)
    img = img if img is not None else Image.open(path).convert("RGB").resize((224, 224))
    cache = os.path.join(SAM_DIR, _key(path))
    if os.path.exists(cache):
        m = Image.open(cache).convert("L").resize(img.size)
        return np.asarray(m, dtype=np.float32) / 255.0
    m = sam2_refine(img, coarse_mask(path, img))
    Image.fromarray((m * 255).astype(np.uint8)).save(cache)
    return m


def _clean(m):
    """U2-Net gives a soft saliency map: the palm interior is often only
    0.1-0.5 and sometimes has a hole in the middle. Threshold, fill holes via
    the outer contours, then soften the edge a little."""
    m = np.clip((m - 0.08) / 0.27, 0.0, 1.0)
    b = (m > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    b = cv2.morphologyEx(np.pad(b, 8, mode="edge"), cv2.MORPH_CLOSE, k)[8:-8, 8:-8]
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.01 * b.size]  # drop specks
    filled = np.zeros_like(b)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.0)


def _raw_mask(img):
    global _session
    from rembg import new_session, remove

    if _session is None:
        _session = new_session("u2net")
    m = remove(img.convert("RGB"), session=_session, only_mask=True)
    return np.asarray(m.convert("L").resize(img.size), dtype=np.float32) / 255.0


def sam2_refine(img, coarse):
    """Prompt SAM 2.1 with a box + inner point taken from the coarse mask."""
    b = (coarse > 0.5).astype(np.uint8)
    if b.sum() < 50:  # nothing found, nothing to prompt with
        return coarse
    ys, xs = np.nonzero(b)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    y, x = np.unravel_index(dt.argmax(), dt.shape)
    return sam2_box(img, box, (int(x), int(y)))


def sam2_box(img, box, point):
    """SAM 2.1 mask from an explicit box [x0,y0,x1,y1] + one positive point (pixels).
    The app uses the MediaPipe hand landmarks for both, so a face in the frame
    is never part of the prompt."""
    global _sam
    import torch
    from transformers import Sam2Model, Sam2Processor

    if _sam is None:
        _sam = (Sam2Processor.from_pretrained(SAM_ID),
                Sam2Model.from_pretrained(SAM_ID).eval())
    proc, model = _sam
    x, y = point
    inp = proc(images=img.convert("RGB"), input_boxes=[[box]],
               input_points=[[[[int(x), int(y)]]]], input_labels=[[[1]]],
               return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    m = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
    m = np.asarray(m).squeeze().astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def mask_from_image(img):
    """Used by the app on a fresh photo (no cache)."""
    return sam2_refine(img, _clean(_raw_mask(img)))


# ---------- procedural backgrounds ----------
def make_background(size, rng, family):
    h = w = size
    if family == "solid":
        c = rng.integers(0, 255, 3)
        bg = np.ones((h, w, 3)) * c
    elif family == "gradient":
        c1, c2 = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        bg = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
    elif family == "noise":
        small = rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    elif family == "stripes":  # curtains / shelves
        period = rng.integers(12, 40)
        c1, c2 = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
        x = (np.arange(w) // period) % 2
        bg = np.where(x[None, :, None] == 0, c1, c2) * np.ones((h, w, 3))
        bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 3)
    elif family == "blobs":  # out-of-focus room
        bg = np.ones((h, w, 3)) * rng.integers(20, 120, 3)
        for _ in range(12):
            cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 60)
            col = tuple(int(v) for v in rng.integers(0, 255, 3))
            cv2.circle(bg, (int(cx), int(cy)), int(r), col, -1)
        bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 9)
    else:  # dark room
        bg = np.ones((h, w, 3)) * rng.integers(5, 40, 3)
        bg += rng.normal(0, 6, (h, w, 3))
    return np.clip(bg, 0, 255).astype(np.float32)


TRAIN_BG = ["solid", "gradient", "noise"]          # used for augmentation
TEST_BG = ["stripes", "blobs", "dark"]             # held-out families for testing


def swap_background(img, mask, rng, families):
    size = img.size[0]
    arr = np.asarray(img, dtype=np.float32)
    bg = make_background(size, rng, families[rng.integers(len(families))])
    m = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
    out = arr * m + bg * (1 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------- photometric / geometric shifts ----------
def low_light(arr, rng, k=None):
    k = k if k is not None else rng.uniform(0.18, 0.35)
    x = (arr / 255.0) ** 1.6 * k
    x = x + rng.normal(0, 0.015, x.shape)  # sensor noise in the dark
    return x * 255


def overexposed(arr, rng):
    return arr * rng.uniform(1.7, 2.2) + 30


def motion_blur(arr, rng, ksize=None):
    k = int(ksize or rng.integers(15, 27))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    ang = rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum()
    return cv2.filter2D(arr.astype(np.float32), -1, kern)


def noise_jpeg(arr, rng):
    x = arr + rng.normal(0, 18, arr.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 12])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)


def rotate_zoom_out(arr, rng):
    h, w = arr.shape[:2]
    ang = rng.uniform(25, 40) * rng.choice([-1, 1])
    sc = rng.uniform(0.6, 0.75)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    # replicate (not reflect) so we don't create mirrored copies of the hand
    return cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_condition(img, cond, rng, mask=None, bg_families=TEST_BG):
    """img: PIL RGB (square). Returns PIL RGB."""
    if cond == "clean":
        return img
    if cond == "new_background":
        assert mask is not None
        return swap_background(img, mask, rng, bg_families)
    arr = np.asarray(img, dtype=np.float32)
    if cond == "low_light":
        out = low_light(arr, rng)
    elif cond == "overexposed":
        out = overexposed(arr, rng)
    elif cond == "motion_blur":
        out = motion_blur(arr, rng)
    elif cond == "noise_jpeg":
        out = noise_jpeg(arr, rng)
    elif cond == "rotate_zoom_out":
        out = rotate_zoom_out(arr, rng)
    elif cond == "night_combo":
        out = motion_blur(arr, rng, ksize=13)
        out = low_light(out, rng, k=0.3)
        out = noise_jpeg(np.clip(out, 0, 255), rng)
    else:
        raise ValueError(cond)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class EnvAugment:
    """Training-time augmentation that samples a random 'environment'.

    Uses only TRAIN_BG background families so the test backgrounds stay unseen.
    """

    def __init__(self, size=224, p=0.7, use_bg=True, seed=0):
        self.size = size
        self.p = p
        self.use_bg = use_bg
        self.rng = np.random.default_rng(seed)
        self.conds = ["low_light", "overexposed", "motion_blur", "noise_jpeg",
                      "rotate_zoom_out"] + (["new_background"] * 2 if use_bg else [])

    def __call__(self, img, path=None):
        rng = self.rng
        img = img.resize((self.size, self.size))
        mask = Image.fromarray((hand_mask(path) * 255).astype(np.uint8)).resize(
            (self.size, self.size)) if (self.use_bg and path) else None
        # light geometric aug always (same as A1); mask follows the image
        flip = rng.random() < 0.2
        ang = rng.uniform(-15, 15)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT) if mask else None
        img = img.rotate(ang, resample=Image.BILINEAR)
        mask = mask.rotate(ang, resample=Image.BILINEAR) if mask else None
        if rng.random() > self.p:
            return img
        cond = self.conds[rng.integers(len(self.conds))]
        m = np.asarray(mask, dtype=np.float32) / 255.0 if mask else None
        return apply_condition(img, cond, rng, mask=m, bg_families=TRAIN_BG)
