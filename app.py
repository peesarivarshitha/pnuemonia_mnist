import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load model - Wrapped in cache to prevent reloading on every click
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Page config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# =========================
# 1️⃣ Header Section
# =========================
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.subheader("AI Assisted Screening Tool")

st.write("Upload a chest X-ray image to check **pneumonia risk**.")
st.markdown("---")

# =========================
# 2️⃣ File Upload Section
# =========================
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Most models trained on X-rays still expect 3 channels (RGB)
        image = Image.open(uploaded_file).convert("RGB")
    except:
        st.error("❌ Uploaded file is not a valid image.")
        st.stop()

    # =========================
    # 3️⃣ Image Preview Section
    # =========================
    st.subheader("🖼️ Image Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(image, use_container_width=True)

    # --- THE CRITICAL FIX ---
    # Change img_size from 128 to 150 to match the 36,992 feature requirement
    img_size = 152
    resized_img = image.resize((img_size, img_size))

    with col2:
        st.write(f"Resized Image ({img_size}x{img_size})")
        st.image(resized_img, use_container_width=True)

    st.markdown("---")

    # =========================
    # 4️⃣ Prediction Button
    # =========================
    if st.button("🔍 Analyze Image"):
        # Preprocess image
        img_array = np.array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 150, 150, 3)

        try:
            # Predict
            prediction = model.predict(img_array)
            
            # Robustly extract probability (handles [[val]] or [val] shapes)
            pneumonia_prob = float(np.squeeze(prediction))
            normal_prob = 1 - pneumonia_prob

            # Determine class
            result = "PNEUMONIA" if pneumonia_prob > 0.5 else "NORMAL"
            color = "red" if result == "PNEUMONIA" else "green"

            # =========================
            # A️⃣ Prediction Result Card
            # =========================
            st.markdown("## 📊 Prediction Result")
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:10px; background-color:#f5f5f5; text-align:center; border:2px solid {color};">
                <h2 style="color:{color};">Status: {result}</h2>
                <h3>Confidence: {max(pneumonia_prob, normal_prob)*100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================
            # B️⃣ Probability Display
            # =========================
            st.subheader("📈 Probability Distribution")
            probs = [normal_prob * 100, pneumonia_prob * 100]
            labels = ["Normal", "Pneumonia"]

            fig, ax = plt.subplots()
            ax.barh(labels, probs, color=['green', 'red'])
            ax.set_xlabel("Probability (%)")
            st.pyplot(fig)

            # =========================
            # C️⃣ Risk Indicator
            # =========================
            st.subheader("⚠️ Risk Level")
            if pneumonia_prob > 0.8:
                st.error("⚠️ Risk Level: HIGH")
            elif pneumonia_prob > 0.5:
                st.warning("⚠️ Risk Level: MODERATE")
            else:
                st.success("✅ Risk Level: LOW")

        except ValueError as e:
            st.error(f"Dimension mismatch: {e}")
            st.info("Try changing `img_size = 150` to **152** or **224** in the code if this persists.")

    # =========================
    # D️⃣ Disclaimer Section
    # =========================
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer**: For educational purposes only. Not for medical diagnosis.")
