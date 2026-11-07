"""Builds data/stress_test/<condition>/<class>/*.jpg from the A1 TEST split.

Also caches hand masks for train/val/test (needed for background swap).
Run once:  python src/make_stress_set.py
"""
import os
import shutil

import numpy as np
from PIL import Image

from common import CLASSES, STRESS_DIR, list_split
from stress import CONDITIONS, apply_condition, hand_mask

SIZE = 224


def main():
    # 1) masks for every split (train masks are used by EnvAugment)
    for split in ("train", "val", "test"):
        items = list_split(split)
        for i, (p, _) in enumerate(items):
            hand_mask(p, Image.open(p).convert("RGB").resize((SIZE, SIZE)))
            if i % 50 == 0:
                print(f"masks {split}: {i}/{len(items)}")

    # 2) stress test set (fixed seed per image -> reproducible)
    if os.path.exists(STRESS_DIR):
        shutil.rmtree(STRESS_DIR)
    test = list_split("test")
    for cond in CONDITIONS:
        for c in CLASSES:
            os.makedirs(os.path.join(STRESS_DIR, cond, c), exist_ok=True)
        for idx, (p, y) in enumerate(test):
            rng = np.random.default_rng(1000 + idx)
            img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
            mask = hand_mask(p, img) if cond == "new_background" else None
            out = apply_condition(img, cond, rng, mask=mask)
            name = f"SYN_{cond}_{os.path.basename(p)}" if cond != "clean" else os.path.basename(p)
            out.save(os.path.join(STRESS_DIR, cond, CLASSES[y], name), quality=95)
        print("built", cond)


if __name__ == "__main__":
    main()
