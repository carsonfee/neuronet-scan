import streamlit as st
from PIL import Image
import numpy as np
import json
import torch
import tensorflow as tf
from src.train_torch import ClinicalNet
from src.preprocess import preprocess_image, preprocess_metadata
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.plots import plot_roc
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_models():
    tf_model = tf.keras.models.load_model("models/image_model.h5")
    torch_model = ClinicalNet(input_dim=3)
    torch_model.load_state_dict(torch.load("models/clinical_model.pt"))
    torch_model.eval()
    return tf_model, torch_model

tf_model, torch_model = load_models()

# Streamlit UI
st.set_page_config(page_title="NeuroNetScan", layout="centered")
st.title("🧠 NeuroNetScan")
st.markdown("AI-powered tool for early neurological disorder detection")

uploaded_image = st.file_uploader("Upload Brain Scan Image", type=["png", "jpg", "jpeg"])
uploaded_meta = st.file_uploader("Upload Patient Metadata (JSON)", type="json")

if uploaded_image:
    original_img = Image.open(uploaded_image).convert('RGB')
    st.image(original_img, caption="Uploaded Scan", use_column_width=True)

if uploaded_meta:
    metadata = json.load(uploaded_meta)
    st.json(metadata)

if st.button("Run Diagnostic Analysis"):
    if not uploaded_image or not uploaded_meta:
        st.warning("Please upload both an image and metadata.")
    else:
        # Preprocess inputs
        img_array = np.expand_dims(preprocess_image(uploaded_image), axis=0)
        meta_array = preprocess_metadata(uploaded_meta)
        meta_tensor = torch.tensor([meta_array], dtype=torch.float32)

        # Image model inference
        img_pred = tf_model.predict(img_array, verbose=0)
        img_class = np.argmax(img_pred[0])
        img_score = float(img_pred[0][img_class])

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, tf_model, last_conv_layer_name="conv2d_1")
        cam_image = overlay_heatmap(original_img, heatmap)
        st.subheader("Image Model Results")
        st.image(cam_image, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
        st.write(f"Prediction: **Class {img_class}**, Confidence: **{img_score:.2f}**")

        # Clinical model inference
        with torch.no_grad():
            meta_pred = torch_model(meta_tensor)
            meta_class = torch.argmax(meta_pred, dim=1).item()
            meta_score = float(meta_pred[0][meta_class])

        st.subheader("Metadata Model Results")
        st.write(f"Prediction: **Class {meta_class}**, Confidence: **{meta_score:.2f}**")

        # Combined scoring
        combined_score = 0.5 * img_score + 0.5 * meta_score
        final_diagnosis = "Likely Neurological Condition" if combined_score > 0.6 else "Condition Unlikely"

        st.subheader("🔬 Final Diagnosis")
        st.write(f"🧮 **Combined Confidence Score**: {combined_score:.2f}")
        st.write(f"📋 **Preliminary Diagnosis**: *{final_diagnosis}*")

        # Simulated ROC plot (replace with true labels & scores in practice)
        st.subheader("📈 Model Evaluation")
        y_true = [1, 0, 1, 0]           # Placeholder
        y_img_scores = [0.8, 0.3, 0.9, 0.4]
        y_meta_scores = [0.7, 0.2, 0.85, 0.5]
        
        st.write("Image Model ROC Curve:")
        st.pyplot(plot_roc(y_true, y_img_scores, label="Image Model"))

        st.write("Metadata Model ROC Curve:")
        st.pyplot(plot_roc(y_true, y_meta_scores, label="Metadata Model"))
        
