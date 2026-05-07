import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- APP CONFIG ---
st.set_page_config(page_title="PneuScan AI v3", page_icon="🔬")
MODEL_PATH = 'best_transfer_learning_model.keras'

@st.cache_resource
def load_v3_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_v3_model()

st.title("🔬 PneuScan AI: Professional Diagnostic Suite")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Lung X-Ray", type=["jpg", "png", "jpeg"])

# Sensitivity Tuning: Allows user to reduce False Positives
sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 0.9, 0.5, 0.1)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB').resize((150, 150))
    st.image(img, caption="Patient Radiograph", use_column_width=True)
    
    # Preprocessing
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    # Recall: 0 = Pneumonia, 1 = Normal
    p_normal = model.predict(img_array)[0][0]
    p_pneumonia = 1.0 - p_normal
    
    # Logic based on custom sensitivity
    is_pneumonia = p_pneumonia > sensitivity
    
    st.subheader("Diagnostic Report")
    if is_pneumonia:
        st.error(f"**PNEUMONIA DETECTED** (Probability: {p_pneumonia*100:.1f}%)")
    else:
        st.success(f"**NORMAL LUNGS** (Confidence: {p_normal*100:.1f}%)")
        
    st.info(f"Technical Score: {p_normal:.4f} (Higher = More likely Normal)")

st.sidebar.markdown("""
**How to interpret:**
- If you get too many 'Pneumonia' results on healthy people, **Lower** the sensitivity.
- If you want to catch every possible case, **Raise** the sensitivity.
""")