import streamlit as st
from PIL import Image
import numpy as np
import json
import torch
import tensorflow as tf
from src.train_torch import ClinicalNet
from src.preprocess import preprocess_image, preprocess_metadata

# Load models once
@st.cache_resource
def load_models():
    tf_model = tf.keras.models.load_model("models/image_model.h5")
    
    torch_model = ClinicalNet(input_dim=3)
    torch_model.load_state_dict(torch.load("models/clinical_model.pt"))
    torch_model.eval()
    
    return tf_model, torch_model

tf_model, torch_model = load_models()

st.set_page_config(page_title="NeuroNetScan", layout="centered")
st.title("🧠 NeuroNetScan")
st.markdown("AI-powered tool for early neurological disorder detection")

uploaded_image = st.file_uploader("Upload Brain Scan Image", type=["png", "jpg", "jpeg"])
uploaded_meta = st.file_uploader("Upload Patient Metadata (JSON)", type="json")

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Scan", use_column_width=True)

if uploaded_meta:
    metadata = json.load(uploaded_meta)
    st.json(metadata)

if st.button("Run Diagnostic Analysis"):
    if not uploaded_image or not uploaded_meta:
        st.warning("Please upload both an image and a metadata file.")
    else:
        # Run image inference
        img_array = np.expand_dims(preprocess_image(uploaded_image), axis=0)
        img_pred = tf_model.predict(img_array)
        img_class = np.argmax(img_pred)

        # Run metadata inference
        meta_array = preprocess_metadata(uploaded_meta)
        meta_tensor = torch.tensor([meta_array], dtype=torch.float32)
        with torch.no_grad():
            meta_pred = torch_model(meta_tensor)
            meta_class = torch.argmax(meta_pred, dim=1).item()

        st.success("✅ Inference complete.")
        st.write(f"🧠 **Image Model Prediction**: Class {img_class}")
        st.write(f"📋 **Metadata Model Prediction**: Class {meta_class}")
        
