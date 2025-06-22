import streamlit as st
from PIL import Image
import json

st.set_page_config(page_title="NeuroNetScan", layout="centered")
st.title("🧠 NeuroNetScan")
st.markdown("AI-powered tool for early neurological disorder detection")

uploaded_image = st.file_uploader("Upload Brain Scan Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Scan", use_column_width=True)

uploaded_meta = st.file_uploader("Upload Patient Metadata (JSON)", type="json")
if uploaded_meta:
    metadata = json.load(uploaded_meta)
    st.json(metadata)

if st.button("Run Diagnostic Analysis"):
    st.info("🔍 Inference code coming soon. Hang tight!")

st.markdown("---")
st.caption("Ethical AI for planetary health and human dignity.")
