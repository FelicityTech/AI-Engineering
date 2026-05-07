"""
logic.py
--------
All backend logic: model loading, image preprocessing, and inference.
No Streamlit imports — this module is pure Python and fully testable in isolation.
"""

import io
import os
import time
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import IMG_SIZE, MODEL_PATH, MODEL_TYPE

# ─── Optional heavy dependencies ──────────────────────────────────────────────

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ─── Types ────────────────────────────────────────────────────────────────────

# A prediction result is a plain dict so UI code never needs to import TF.
PredictionResult = dict   # keys: label, confidence, pneumonia_score, normal_score, threshold, demo


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model() -> Tuple[Optional[object], bool]:
    """
    Load the Keras model from MODEL_PATH.

    Returns:
        (model, is_demo)
        is_demo=True means the model file was not found or could not be loaded;
        the app will fall back to simulated predictions.
    """
    if not _TF_AVAILABLE:
        return None, True

    if not os.path.exists(MODEL_PATH):
        return None, True

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, False
    except Exception:
        return None, True


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Replicate the training pipeline exactly:
      1. Decode bytes → grayscale
      2. Resize to IMG_SIZE × IMG_SIZE
      3. Normalise pixels to [0, 1]
      4. Stack to pseudo-RGB  (3 identical channels)
      5. Scale back to [0, 255] and apply model-specific preprocess_input

    Args:
        image_bytes: Raw bytes of a JPEG or PNG file.

    Returns:
        NumPy array of shape (1, IMG_SIZE, IMG_SIZE, 3) ready for model.predict().

    Raises:
        ValueError: If the image cannot be decoded.
    """
    if _CV2_AVAILABLE:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image. Upload a valid JPEG or PNG.")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    else:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = np.array(pil, dtype=np.float32) / 255.0

    # (H, W) → (1, H, W, 3) in [0, 255]
    batch = np.expand_dims(np.stack([img, img, img], axis=-1) * 255.0, axis=0)

    if MODEL_TYPE == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    return preprocess_input(batch)


# ─── Inference helpers ────────────────────────────────────────────────────────

def _build_result(pneumonia_score: float, threshold: float, demo: bool) -> PredictionResult:
    """Assemble the standard result dict from a raw pneumonia probability."""
    normal_score = 1.0 - pneumonia_score
    label        = "PNEUMONIA" if pneumonia_score >= threshold else "NORMAL"
    confidence   = pneumonia_score if label == "PNEUMONIA" else normal_score
    return {
        "label":           label,
        "confidence":      confidence,
        "pneumonia_score": pneumonia_score,
        "normal_score":    normal_score,
        "threshold":       threshold,
        "demo":            demo,
    }


def _predict_real(model, image_bytes: bytes, threshold: float) -> PredictionResult:
    """Run the real Keras model."""
    batch = preprocess_image(image_bytes)
    # Model outputs P(Normal) — invert to get P(Pneumonia)
    raw             = float(model.predict(batch, verbose=0)[0][0])
    pneumonia_score = 1.0 - raw
    return _build_result(pneumonia_score, threshold, demo=False)


def _predict_demo(image_bytes: bytes, threshold: float) -> PredictionResult:
    """
    Simulate a prediction when no model is available.
    Score is derived deterministically from the image's pixel statistics
    so different uploads produce meaningfully different results.
    """
    pil  = Image.open(io.BytesIO(image_bytes)).convert("L").resize((32, 32))
    arr  = np.array(pil, dtype=np.float32) / 255.0
    seed = int(arr.mean() * 1000 + arr.std() * 500) % 10_000
    rng  = random.Random(seed)

    # Darker (hazier) images lean toward Pneumonia
    base            = 0.45 + (0.5 - float(arr.mean())) * 0.4
    pneumonia_score = float(np.clip(base + rng.uniform(-0.12, 0.12), 0.05, 0.95))
    return _build_result(pneumonia_score, threshold, demo=True)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_inference(
    model,
    is_demo: bool,
    image_bytes: bytes,
    threshold: float,
) -> PredictionResult:
    """
    Entry point for the UI layer.

    Args:
        model:       Loaded Keras model, or None in demo mode.
        is_demo:     If True, use simulated predictions.
        image_bytes: Raw bytes from the uploaded file.
        threshold:   Classification threshold (0–1).

    Returns:
        PredictionResult dict.
    """
    time.sleep(0.5)   # simulate processing time for a smoother UX
    if is_demo or model is None:
        return _predict_demo(image_bytes, threshold)
    return _predict_real(model, image_bytes, threshold)
