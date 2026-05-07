"""
PneumoScan — Chest X-Ray Pneumonia Detection
=============================================
Upload a chest X-ray and the model predicts whether pneumonia is present.

Run:
    pip install streamlit tensorflow opencv-python-headless pillow numpy
    streamlit run app.py

Place `best_transfer_learning_model.keras` in the same folder as `app.py`
for real inference. Without it, the app runs in demo mode.
"""

import io
import os
import time
import random
import numpy as np
import streamlit as st
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE   = 150
MODEL_PATH = "best_transfer_learning_model.keras"

# Change to "efficientnet" if EfficientNetB0 was your best model
MODEL_TYPE = "mobilenet"

THRESHOLD_PRESETS = {
    "Screening  (high sensitivity)": 0.30,
    "Balanced":                      0.50,
    "High precision":                0.65,
}


# ─── Model ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Keras model. Returns (model, is_demo)."""
    if not TF_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None, True
    try:
        return tf.keras.models.load_model(MODEL_PATH), False
    except Exception:
        return None, True


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Replicate the training pipeline exactly:
      grayscale → resize → normalise → pseudo-RGB → preprocess_input
    """
    if CV2_AVAILABLE:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image.")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    else:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = np.array(pil, dtype=np.float32) / 255.0

    # Stack grayscale to pseudo-RGB and scale back to [0, 255] for preprocess_input
    batch = np.expand_dims(np.stack([img, img, img], axis=-1) * 255.0, axis=0)

    if MODEL_TYPE == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    return preprocess_input(batch)


# ─── Inference ────────────────────────────────────────────────────────────────

def _build_result(pneumonia_score: float, threshold: float, demo: bool) -> dict:
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


def predict_real(model, image_bytes: bytes, threshold: float) -> dict:
    batch = preprocess_image(image_bytes)
    # model outputs P(Normal); invert for P(Pneumonia)
    pneumonia_score = 1.0 - float(model.predict(batch, verbose=0)[0][0])
    return _build_result(pneumonia_score, threshold, demo=False)


def predict_demo(image_bytes: bytes, threshold: float) -> dict:
    """Derive a deterministic score from the image so uploads feel realistic."""
    pil  = Image.open(io.BytesIO(image_bytes)).convert("L").resize((32, 32))
    arr  = np.array(pil, dtype=np.float32) / 255.0
    seed = int(arr.mean() * 1000 + arr.std() * 500) % 10000
    rng  = random.Random(seed)
    # Darker images → lean toward Pneumonia
    base            = 0.45 + (0.5 - float(arr.mean())) * 0.4
    pneumonia_score = float(np.clip(base + rng.uniform(-0.12, 0.12), 0.05, 0.95))
    return _build_result(pneumonia_score, threshold, demo=True)


def run_inference(model, is_demo: bool, image_bytes: bytes, threshold: float) -> dict:
    time.sleep(0.5)
    return predict_demo(image_bytes, threshold) if (is_demo or model is None) \
           else predict_real(model, image_bytes, threshold)


# ─── Styling ──────────────────────────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #080F1C; color: #C8D6E8; }

[data-testid="stSidebar"] {
    background: #0C1729 !important;
    border-right: 1px solid #172035;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1100px; }

[data-testid="stFileUploader"] section {
    border: 1.5px dashed #1E3356 !important;
    border-radius: 14px !important;
    background: #0C1729 !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] section:hover { border-color: #2563EB !important; }

[data-testid="stMetric"] {
    background: #0C1729;
    border: 1px solid #172035;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #4A6080 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #C8D6E8 !important; font-size: 1.3rem !important; }

.stButton > button {
    background: #1D4ED8; color: #fff; border: none;
    border-radius: 8px; font-family: 'DM Sans', sans-serif;
    font-weight: 500; transition: background 0.18s;
}
.stButton > button:hover { background: #1E40AF; }
.stSlider > div { color: #4A6080; }
.stSelectbox > div > div { background: #0C1729; border-color: #1E3356; color: #C8D6E8; }
hr { border-color: #172035; }
.stAlert { background: #0C1729; border-radius: 10px; }
</style>
"""


# ─── UI components ────────────────────────────────────────────────────────────

def _label(text: str) -> None:
    st.markdown(
        f"<div style='font-size:0.72rem; letter-spacing:2.5px; text-transform:uppercase;"
        f" color:#2563EB; font-weight:600; margin-bottom:10px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _empty_panel(icon: str, text: str, sub: str) -> None:
    st.markdown(
        f"<div style='height:290px; display:flex; flex-direction:column; align-items:center;"
        f" justify-content:center; border:1.5px dashed #172035; border-radius:14px;"
        f" color:#2A3F5A; font-size:0.86rem; gap:10px;'>"
        f"<div style='font-size:2.4rem;'>{icon}</div><div>{text}</div>"
        f"<div style='font-size:0.73rem; color:#172035;'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def _confidence_bar(label: str, score: float, color: str) -> str:
    pct = int(score * 100)
    return (
        f"<div style='margin:8px 0;'>"
        f"<div style='display:flex; justify-content:space-between; font-size:0.76rem;"
        f" color:#4A6080; margin-bottom:4px;'><span>{label}</span><span>{pct}%</span></div>"
        f"<div style='background:#101E35; border-radius:6px; height:7px; overflow:hidden;'>"
        f"<div style='width:{pct}%; height:100%; background:{color}; border-radius:6px;'>"
        f"</div></div></div>"
    )


def render_sidebar() -> float:
    with st.sidebar:
        st.markdown(
            "<div style='font-family:\"DM Serif Display\",serif; font-size:1.3rem;"
            " color:#EEF4FF; margin-bottom:4px;'>⚙ Settings</div>"
            "<div style='color:#4A6080; font-size:0.78rem; margin-bottom:18px;'>"
            "Configure detection parameters</div>",
            unsafe_allow_html=True,
        )
        preset = st.selectbox(
            "Threshold preset", list(THRESHOLD_PRESETS.keys()), index=0,
            help=(
                "Screening: maximise sensitivity — fewer missed cases.\n"
                "Balanced: equal precision and recall.\n"
                "High precision: fewer false alarms, may miss some cases."
            ),
        )
        threshold = THRESHOLD_PRESETS[preset]
        if st.checkbox("Custom threshold"):
            threshold = st.slider("Value", 0.10, 0.90, threshold, 0.05)

        st.divider()
        st.markdown(
            "<div style='font-size:0.78rem; color:#4A6080; line-height:1.8;'>"
            "<strong style='color:#6B8BAF;'>Threshold guide</strong><br>"
            "🔵 <b>0.30</b> — Screening mode<br>&emsp;Maximise sensitivity<br><br>"
            "🟡 <b>0.50</b> — Balanced mode<br>&emsp;Precision / recall trade-off<br><br>"
            "🔴 <b>0.65</b> — High precision<br>&emsp;Fewer false alarms</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<div style='font-size:0.73rem; color:#2A3F5A; line-height:1.6;'>"
            "⚠️ <strong>Disclaimer</strong><br>"
            "Research prototype only. Not a certified medical device. "
            "Always consult a qualified radiologist.</div>",
            unsafe_allow_html=True,
        )
    return threshold


def render_header() -> None:
    st.markdown(
        "<div style='margin-bottom:1.8rem;'>"
        "<div style='font-size:0.73rem; letter-spacing:3px; text-transform:uppercase;"
        " color:#2563EB; font-weight:600; margin-bottom:8px;'>AI Radiology Assistant</div>"
        "<div style='font-family:\"DM Serif Display\",serif; font-size:2.5rem;"
        " color:#EEF4FF; line-height:1.15; margin-bottom:10px;'>"
        "Chest X-Ray<br><em>Pneumonia Analysis</em></div>"
        "<div style='color:#4A6080; font-size:0.9rem; max-width:500px; line-height:1.65;'>"
        "Upload a chest radiograph. The model analyses it for signs of pneumonia "
        "and returns a confidence-scored prediction.</div></div>",
        unsafe_allow_html=True,
    )


def render_result_card(result: dict) -> None:
    is_pneu  = result["label"] == "PNEUMONIA"
    accent   = "#EF4444" if is_pneu else "#22C55E"
    bg       = "rgba(239,68,68,0.06)" if is_pneu else "rgba(34,197,94,0.06)"
    border   = "#EF4444" if is_pneu else "#22C55E"
    icon     = "🔴" if is_pneu else "🟢"
    title    = "Pneumonia Detected" if is_pneu else "No Pneumonia Detected"
    subtitle = (
        "Signs consistent with pneumonia are present in this X-ray."
        if is_pneu else
        "No significant pneumonia indicators found in this X-ray."
    )
    bars     = (
        _confidence_bar("Pneumonia", result["pneumonia_score"], "#EF4444") +
        _confidence_bar("Normal",    result["normal_score"],    "#22C55E")
    )
    demo_tag = (
        "<div style='font-size:0.72rem; color:#2A3F5A; margin-top:4px;'>"
        "⚠ Simulated score — demo mode</div>"
        if result["demo"] else ""
    )
    st.markdown(
        f"<div style='background:{bg}; border:1.5px solid {border}; border-radius:14px;"
        f" padding:22px 26px; margin-top:4px;'>"
        f"<div style='font-size:1.8rem; margin-bottom:6px;'>{icon}</div>"
        f"<div style='font-family:\"DM Serif Display\",serif; font-size:1.3rem;"
        f" color:{accent}; margin-bottom:6px;'>{title}</div>"
        f"<div style='color:#4A6080; font-size:0.85rem; margin-bottom:18px;'>{subtitle}</div>"
        f"{bars}"
        f"<div style='margin-top:14px; font-size:0.76rem; color:#2A3F5A;'>"
        f"Confidence: <strong style='color:#8AAFD4;'>{result['confidence']:.1%}</strong>"
        f"&nbsp;|&nbsp; Threshold: <strong style='color:#8AAFD4;'>{result['threshold']:.2f}</strong>"
        f"</div>{demo_tag}</div>",
        unsafe_allow_html=True,
    )


def render_metrics(result: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction",     result["label"])
    c2.metric("Confidence",     f"{result['confidence']:.1%}")
    c3.metric("Threshold used", f"{result['threshold']:.2f}")


def render_clinical_note(result: dict) -> None:
    thr = result["threshold"]
    if result["label"] == "PNEUMONIA":
        st.warning(
            f"**Clinical note:** This result suggests pneumonia may be present. "
            f"Refer to a qualified radiologist for confirmed diagnosis. "
            f"Threshold used: **{thr:.2f}** — lower values increase sensitivity.",
            icon="⚠️",
        )
    else:
        st.info(
            f"**Clinical note:** No significant pneumonia indicators detected. "
            f"If symptoms persist, a clinical assessment is still recommended. "
            f"Threshold used: **{thr:.2f}**.",
            icon="ℹ️",
        )


def render_how_it_works() -> None:
    steps = [
        ("📤", "Upload",  "JPEG or PNG chest X-ray"),
        ("🔬", "Analyse", "MobileNetV2 deep learning"),
        ("📊", "Result",  "Prediction + confidence"),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), steps):
        col.markdown(
            f"<div style='background:#0C1729; border:1px solid #172035; border-radius:12px;"
            f" padding:18px 20px; text-align:center;'>"
            f"<div style='font-size:1.6rem; margin-bottom:8px;'>{icon}</div>"
            f"<div style='font-size:0.85rem; font-weight:600; color:#C8D6E8; margin-bottom:4px;'>{title}</div>"
            f"<div style='font-size:0.76rem; color:#4A6080;'>{desc}</div></div>",
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        "<div style='text-align:center; color:#1E3356; font-size:0.73rem; margin-top:28px;'>"
        "PneumoScan &nbsp;·&nbsp; Research prototype &nbsp;·&nbsp;"
        " Not for clinical use &nbsp;·&nbsp; MobileNetV2 | AUC ≥ 0.97</div>",
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="PneumoScan — Chest X-Ray Analysis",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLES, unsafe_allow_html=True)

    threshold = render_sidebar()
    render_header()

    with st.spinner("Loading model…"):
        model, is_demo = load_model()

    if is_demo:
        st.warning(
            "**Demo mode** — `best_transfer_learning_model.keras` not found. "
            "Predictions are simulated for UI demonstration. "
            "Place your trained `.keras` file alongside `app.py` for real inference.",
            icon="🔧",
        )

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        _label("X-Ray Upload")
        uploaded = st.file_uploader(
            "Drop a chest X-ray here or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(Image.open(uploaded).convert("RGB"),
                     caption=f"📁 {uploaded.name}", use_column_width=True)
        else:
            _empty_panel("🩻", "No image uploaded yet", "JPEG · PNG")

    with right:
        _label("Analysis Result")
        if uploaded:
            with st.spinner("Analysing X-ray…"):
                try:
                    outcome = run_inference(model, is_demo, uploaded.getvalue(), threshold)
                except Exception as err:
                    st.error(f"Inference error: {err}")
                    st.stop()
            render_result_card(outcome)
            st.divider()
            render_metrics(outcome)
        else:
            _empty_panel("📋", "Awaiting image upload", "Results will appear here")

    if uploaded:
        render_clinical_note(outcome)

    st.divider()
    render_how_it_works()
    render_footer()


if __name__ == "__main__":
    main()
