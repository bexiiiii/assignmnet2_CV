"""'Refine' step inspired by ArtAug.

ArtAug (Neurohive, 2025) improves images in a loop: an understanding model
looks at the picture, says what is wrong (too dark, bad composition...) and
the generator fixes it. The real ArtAug sits on top of a FLUX diffusion model
which we cannot run locally, so we keep the idea but make both halves tiny:

  1. "understand": measure the frame (mean luminance, contrast, sharpness)
  2. "fix": apply only the corrections that the diagnosis asks for
     (gamma lift + CLAHE for dark frames, highlight compression for
      blown-out frames, light denoise for noisy frames)

It is used (a) before training on low-exposure frames and (b) optionally
at test time. The ablation in the report turns it on/off.
"""
import cv2
import numpy as np
from PIL import Image


def diagnose(img):
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum = g.mean() / 255.0
    contrast = g.std() / 255.0
    sharp = cv2.Laplacian(g, cv2.CV_32F).var()
    # high-frequency energy vs. a blurred copy ~ how much grain there is
    noise = float(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3)).mean())
    return {"luminance": lum, "contrast": contrast, "sharpness": sharp, "noise": noise}


def refine(img, return_log=False):
    d = diagnose(img)
    arr = np.asarray(img.convert("RGB"))
    log = []
    dark = d["luminance"] < 0.30
    # dark phone frames are almost always noisy/compressed: brightening first
    # would amplify the grain and JPEG blocks, so denoise before the lift
    if d["noise"] > 7 or dark:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 6 if dark else 8, 6, 7, 21)
        log.append("denoise")
    if dark:
        gamma = np.interp(d["luminance"], [0.05, 0.30], [0.35, 0.75])
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        arr = cv2.LUT(arr, lut)
        log.append(f"gamma {gamma:.2f}")
    if d["luminance"] > 0.80:
        lut = (np.linspace(0, 1, 256) ** 1.8 * 255).astype(np.uint8)
        arr = cv2.LUT(arr, lut)
        log.append("compress highlights")
    if d["contrast"] < 0.12 or "gamma" in " ".join(log):
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        lab[..., 0] = clahe.apply(lab[..., 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        log.append("CLAHE")
    out = Image.fromarray(arr)
    return (out, d, log) if return_log else out
