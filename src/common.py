"""Shared paths, classes and data loading for Assignment 2."""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["thumbs_up", "peace", "stop"]
SEED = 42

# A1 dataset lives next to this project (Downloads/dataset). Can be overridden.
DATASET_DIR = os.environ.get(
    "GESTURE_DATA", os.path.join(os.path.dirname(ROOT), "dataset")
)
STRESS_DIR = os.path.join(ROOT, "data", "stress_test")
EXTRA_DIR = os.path.join(ROOT, "data", "real_new_place")  # optional: new real photos
RESULTS = os.path.join(ROOT, "results")
WEIGHTS = os.path.join(ROOT, "weights")
for d in (RESULTS, WEIGHTS):
    os.makedirs(d, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    forced = os.environ.get("GESTURE_DEVICE")  # e.g. cpu / mps / cuda
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_split(split, base=None):
    base = base or DATASET_DIR
    items = []
    for ci, c in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(base, split, c, "*.jpg"))):
            items.append((f, ci))
    return items


def list_folder(folder):
    """folder/<class>/*.jpg -> [(path, label)]"""
    items = []
    for ci, c in enumerate(CLASSES):
        for ext in ("jpg", "jpeg", "png", "JPG", "PNG"):
            for f in sorted(glob.glob(os.path.join(folder, c, f"*.{ext}"))):
                items.append((f, ci))
    return items


def to_tensor(img, size):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class GestureDS(Dataset):
    """Generic dataset: list of (path, label) + optional PIL-level augment fn."""

    def __init__(self, items, size=224, augment=None):
        self.items = items
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.augment is not None:
            img = self.augment(img, path)  # augment(img, path) -> PIL
        return to_tensor(img, self.size), y
