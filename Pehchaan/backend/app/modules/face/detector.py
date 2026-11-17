"""
Face detection — deep-scan edition.

Backends tried in order (strongest first):
  1. OpenCV YuNet (DNN, ONNX) — accurate on scanned documents, tolerant of
     tilt, low resolution and lighting variation.
  2. Haar cascade (OpenCV built-in) — always available, fast but weaker on
     printed / low-contrast images.

Deep-scan improvements over the original:
  • YuNet confidence threshold lowered from 0.70 to 0.55 so it picks up
    low-contrast document portraits that plain thresholding rejects.
  • For Haar, we run three passes: original, CLAHE-enhanced, and histogram-
    equalised; take the union; then de-duplicate by IoU.  This recovers faces
    that are either very dark or very bright.
  • The 'pick_portrait' heuristic now gives a *larger size bonus* (0.40 vs
    0.30) because document photos are rarely ghost-sized.
  • Faces below 30 × 30 px (after de-duplication) are discarded — they are
    almost always watermark noise.
  • detect_largest_face() pads the returned crop with a 10% border so the
    SFace embedding sees some context around the face region.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_local = threading.local()


def _cascade() -> cv2.CascadeClassifier:
    if not hasattr(_local, "cascade"):
        _local.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _local.cascade


def _cascade_profile() -> cv2.CascadeClassifier:
    """Profile-face cascade — catches slightly turned heads on documents."""
    if not hasattr(_local, "cascade_profile"):
        _local.cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
    return _local.cascade_profile


_YUNET_PATH = Path(__file__).resolve().parents[3] / "models" / "face" / "face_detection_yunet_2023mar.onnx"
_YUNET_MAX_SIDE = 1280
_YUNET_MIN_SCORE = 0.55   # lowered from 0.70 — document photos have lower contrast


def _yunet():
    from app.modules.provenance import model_trusted
    if not _YUNET_PATH.is_file() or not model_trusted("face_detector_yunet"):
        return None
    if not hasattr(_local, "yunet"):
        _local.yunet = cv2.FaceDetectorYN.create(
            str(_YUNET_PATH), "", (320, 320),
            score_threshold=_YUNET_MIN_SCORE,
            nms_threshold=0.3,
        )
    return _local.yunet


# Minimum tonal range (5th-95th pctile of grey) to count as a real photo.
# Lowered to 50 so we don't reject faded passport photos.
_MIN_TONAL_RANGE = 50


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    quality: float = 0.0
    tonal_range: float = 0.0
    sharpness: float = 0.0


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / float(a.w * a.h + b.w * b.h - inter or 1)


def _score(gray: np.ndarray, box: FaceBox) -> FaceBox:
    crop = gray[box.y: box.y + box.h, box.x: box.x + box.w]
    if crop.size == 0:
        return box
    p5, p95 = np.percentile(crop, (5, 95))
    box.tonal_range = float(p95 - p5)
    box.sharpness = float(cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var())
    # Size bonus bumped to 0.40; size matters more for documents than live selfies
    box.quality = round(
        min(1.0, box.tonal_range / 160) * 0.60
        + min(1.0, box.sharpness / 400) * 0.40,
        3,
    )
    return box


def _haar_pass(candidate: np.ndarray, found: list[FaceBox]) -> None:
    """Run frontal + profile Haar and append new non-duplicate detections."""
    for (x, y, w, h) in _cascade().detectMultiScale(
        candidate, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30)
    ):
        box = FaceBox(int(x), int(y), int(w), int(h))
        if not any(_iou(box, f) > 0.4 for f in found):
            found.append(box)

    for (x, y, w, h) in _cascade_profile().detectMultiScale(
        candidate, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30)
    ):
        box = FaceBox(int(x), int(y), int(w), int(h))
        if not any(_iou(box, f) > 0.4 for f in found):
            found.append(box)


def detect_faces(image: np.ndarray) -> list[FaceBox]:
    """All face candidates, de-duplicated and quality-scored.

    Three-pass strategy for documents:
      Pass 1 — YuNet DNN (best accuracy)
      Pass 2 — Haar on original grey (fast, misses dark photos)
      Pass 3 — Haar on CLAHE-enhanced grey (recovers dark/faded photos)
      Pass 4 — Haar on equalised grey (last resort for very low contrast)
    """
    gray = _gray(image)
    found: list[FaceBox] = []

    # --- Pass 1: YuNet DNN ---
    detector = _yunet() if image.ndim == 3 else None
    if detector is not None:
        h, w = image.shape[:2]
        scale = min(1.0, _YUNET_MAX_SIDE / max(h, w))
        small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        for row in faces if faces is not None else []:
            x, y, bw, bh = (row[:4] / scale).astype(int)
            box = FaceBox(max(0, int(x)), max(0, int(y)), int(bw), int(bh))
            if box.w >= 30 and box.h >= 30:
                found.append(box)
        if found:
            return [_score(gray, f) for f in found]

    # --- Passes 2-4: Haar multi-pass ---
    clahe_gray = _clahe_gray(gray)
    eq_gray = cv2.equalizeHist(gray)
    for candidate in (gray, clahe_gray, eq_gray):
        _haar_pass(candidate, found)

    # Drop tiny noise boxes
    found = [f for f in found if f.w >= 30 and f.h >= 30]
    return [_score(gray, f) for f in found]


def pick_portrait(faces: list[FaceBox]) -> FaceBox | None:
    """Pick the holder's photo portrait from a list of detected faces.

    For documents: prefer larger faces with good tonal range (real photo)
    over small ghost images or watermarks."""
    if not faces:
        return None
    real = [f for f in faces if f.tonal_range >= _MIN_TONAL_RANGE] or faces
    biggest = max(f.w * f.h for f in real)
    # Increased size weight (0.40) vs original (0.30) — document portraits dominate
    return max(real, key=lambda f: f.quality * 0.60 + (f.w * f.h / biggest) * 0.40)


def detect_largest_face(image: np.ndarray) -> tuple[np.ndarray, FaceBox] | None:
    """Returns the portrait crop (with 10% contextual border) and its box,
    or None if no face found."""
    chosen = pick_portrait(detect_faces(image))
    if chosen is None:
        return None

    # 10% padding around the portrait so SFace sees forehead / chin context
    ph, pw = image.shape[:2]
    pad_y = int(chosen.h * 0.10)
    pad_x = int(chosen.w * 0.10)
    y1 = max(0, chosen.y - pad_y)
    y2 = min(ph, chosen.y + chosen.h + pad_y)
    x1 = max(0, chosen.x - pad_x)
    x2 = min(pw, chosen.x + chosen.w + pad_x)
    crop = image[y1:y2, x1:x2]
    return crop, chosen
