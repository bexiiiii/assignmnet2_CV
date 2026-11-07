"""MediaPipe Hand Landmarker -> 21 hand key-points -> features.

The image models (A1 CNN, DINOv2) look at the whole frame, so background,
face and lighting leak into the decision. Hand landmarks describe only the
hand's geometry, which is what a gesture actually is. We use Google's
pretrained Hand Landmarker (MediaPipe Tasks, 2023) only to get the points;
the gesture classifier on top of them is trained by us on the A1 frames.
"""
import os

import numpy as np
from PIL import Image

from common import WEIGHTS

TASK = os.path.join(WEIGHTS, "hand_landmarker.task")
_det = None

# MediaPipe landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_MCP, PINKY_TIP = 0, 4, 5, 8, 9, 12, 16, 17, 20
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]


def detector():
    global _det
    if _det is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=TASK),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
        _det = vision.HandLandmarker.create_from_options(opts)
    return _det


def detect(img):
    """PIL image -> (pts (21,3) in image-normalised coords, handedness str) or (None, None)."""
    import mediapipe as mp
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    res = detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.hand_landmarks:
        return None, None
    pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], np.float32)
    hand = res.handedness[0][0].category_name if res.handedness else "Right"
    return pts, hand


def features(pts, hand="Right"):
    """Translation/scale invariant 78-d vector:
    63 wrist-centred, size-normalised coordinates (left hands mirrored)
    + 10 fingertip pairwise distances + 5 finger 'extension' ratios."""
    p = pts.copy()
    p[:, :2] -= p[WRIST, :2]
    if hand == "Left":
        p[:, 0] *= -1
    scale = np.linalg.norm(p[MIDDLE_MCP, :2]) + 1e-6
    p /= scale
    tips = p[TIPS, :2]
    d = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)]
    ext = [np.linalg.norm(p[t, :2]) / (np.linalg.norm(p[q, :2]) + 1e-6) for t, q in zip(TIPS, PIPS)]
    return np.concatenate([p.ravel(), d, ext]).astype(np.float32)


def augment_pts(pts, rng):
    """Small in-plane rotation / scale / jitter in landmark space."""
    p = pts.copy()
    c = p[WRIST, :2].copy()
    a = np.radians(rng.uniform(-25, 25))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    p[:, :2] = (p[:, :2] - c) @ R.T * rng.uniform(0.85, 1.15) + c
    p[:, :2] += rng.normal(0, 0.004, p[:, :2].shape)
    return p


def bbox(pts, w, h, margin=0.25):
    """Hand bounding box in pixels from the landmarks (used as SAM 2 prompt)."""
    xs, ys = pts[:, 0] * w, pts[:, 1] * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return [int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w - 1, x1 + mx)), int(min(h - 1, y1 + my))]


def draw(img, pts):
    """Skeleton overlay for the app."""
    import cv2
    a = np.asarray(img.convert("RGB")).copy()
    h, w = a.shape[:2]
    conn = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
            (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
    P = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for i, j in conn:
        cv2.line(a, P[i], P[j], (0, 220, 120), 2)
    for q in P:
        cv2.circle(a, q, 3, (255, 60, 60), -1)
    return Image.fromarray(a)


# ---------------- Google's pretrained MediaPipe Gesture Recognizer ----------------
# Trained by Google on a large, varied hand dataset. Three of its canned gestures
# are exactly our classes. It is very precise but often answers "None" on our
# tight close-ups, so in the cascade it goes first and only when it is sure.
GR_TASK = os.path.join(WEIGHTS, "gesture_recognizer.task")
GR_MAP = {"Thumb_Up": "thumbs_up", "Victory": "peace", "Open_Palm": "stop"}
_gr = None


def gesture(img, min_score=0.5):
    """-> (our class name, score) or (None, None) when Google's model is not sure."""
    global _gr
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    if _gr is None:
        _gr = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=mpt.BaseOptions(model_asset_path=GR_TASK), num_hands=1,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    r = _gr.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not r.gestures:
        return None, None
    c = r.gestures[0][0]
    if c.category_name in GR_MAP and c.score >= min_score:
        return GR_MAP[c.category_name], float(c.score)
    return None, None
