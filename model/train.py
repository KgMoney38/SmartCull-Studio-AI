#Kody Graham
#12/01/2025
#This class will load the images, auto generate tampered variants, and trains my CNN.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from model_def import build_tamper_model

#Reproduce
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = BASE_DIR / "data" / "original"
SAVE_DIR = BASE_DIR / "saved_models"
CLASS_MAP_PATH = BASE_DIR / "model" / "class_map.json"
MODEL_PATH = SAVE_DIR / "tamper_detector.h5"

IMG_SIZE = (224, 224)

#Labels
TAMPER_LABELS = ["original", "tampered"]
TYPE_LABELS = ["original", "jpeg", "blue", "noise", "copy_move", "splice", "inpaint"]

def ensure_dirs():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_original_images() -> list[np.ndarray]:
    if not ORIGINAL_DIR.exists():
        return []

    images = []
    for ext in ("*.jpg","*.jpeg", "*.png"):
        for path in ORIGINAL_DIR.glob(ext):
            img = cv2.imread(str(path))
            if img is None:
                continue
            images.append(img)

    return images

def preprocess_for_model(image_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(image_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype("float32")/ 255.0
    return img_array

def random_rect(h: int, w: int):
    x1 = random.randint(0, max(0, w - w // 2))
    y1 = random.randint(0, max(0, h - h // 2))
    x2 = random.randint(x1 + max(4, w // 8), min(w, x1 + w // 2))
    y2 = random.randint(y1 + max(4, h // 8), min(h, y1 + h // 2))

    return x1, y1, x2, y2

def tamper_jpeg(image_bgr: np.ndarray) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
    success, enc = cv2.imencode('.jpg', image_bgr, encode_param)
    if not success:
        return image_bgr.copy()
    dec =cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def tamper_blur(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    #Random rectangles
    x1, y1, x2, y2 = random_rect(h, w)

    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (15,15), 0)
    img[y1:y2, x1:x2] = roi_blur
    return img

def tamper_noise(image_bgr: np.ndarray) -> np.ndarray:

    h, w = image_bgr.shape[:2]
    img = image_bgr.copy()

    x1, y1, x2, y2 = random_rect(h, w)

    roi = img[y1:y2, x1:x2].astype("float32")
    noise = np.random.normal(loc=0.0,scale=30.0,size=roi.shape)
    roi_noisy= np.clip(roi + noise, 0, 255).astype("uint8")
    img[y1:y2, x1:x2] = roi_noisy
    return img

def tamper_copy_move(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    #Source path
    src_x1, src_y1, src_x2, src_y2 = random_rect(h,w)
    patch = img[src_y1:src_y2, src_x1:src_x2].copy()
    ph, pw = patch.shape[:2]

    dst_x1 = random.randint(0, w - ph)
    dst_y1 = random.randint(0, h - ph)
    dst_x2 = dst_x1 + pw
    dst_y2 = dst_y1 + ph

    img[dst_y1:dst_y2, dst_x1:dst_x2] = patch
    return img

def tamper_splice(target_bgr: np.ndarray, donor_bgr: np.array) -> np.ndarray:
    h,w = target_bgr.shape[:2]
    img = target_bgr.copy()

    donor = donor_bgr.copy()
    donor = cv2.resize(donor, (w,h))

    src_x1, src_y1, src_x2, src_y2 = random_rect(h, w)
    patch = donor[src_y1:src_y2, src_x1:src_x2].copy()
    ph, pw = patch.shape[:2]

    dst_x1 = random.randint(0, max(0, w - pw))
    dst_y1 = random.randint(0, max(0, h - ph))
    dst_x2 = dst_x1 + pw
    dst_y2 = dst_y1 + ph

    patch_float = patch.astype("float32")
    alpha = .8 + .4 * random.random()
    beta = random.randint(-15,15)
    patch_jitter = np.clip(alpha * patch_float + beta, 0, 255).astype("uint8")

    img[dst_y1:dst_y2, dst_x1:dst_x2] = patch_jitter
    return img

def tamper_inpaint(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    x1, y1, x2, y2 = random_rect(h, w)
    mask = np.zeros((h,w), dtype="uint8")
    mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags = cv2.INPAINT_TELEA)
    return inpainted

def build_dataset():

