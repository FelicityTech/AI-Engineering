"""
config.py
---------
All application-wide constants and settings.
Change MODEL_PATH or MODEL_TYPE here when switching models.
"""

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_PATH = "best_transfer_learning_model.keras"

# "mobilenet"  → MobileNetV2  (default)
# "efficientnet" → EfficientNetB0
MODEL_TYPE = "mobilenet"

# ─── Image ────────────────────────────────────────────────────────────────────

IMG_SIZE = 150   # must match training resolution

# ─── Classification thresholds ────────────────────────────────────────────────

THRESHOLD_PRESETS: dict[str, float] = {
    "Screening  (high sensitivity)": 0.30,
    "Balanced":                      0.50,
    "High precision":                0.65,
}

DEFAULT_THRESHOLD = 0.30
