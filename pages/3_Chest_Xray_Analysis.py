"""
MediAssist — Page 3: Chest X-Ray Analysis & Grad-CAM Visualization
MobileNetV2 Deep Learning Transfer Screening for Pulmonary Pathologies
"""
import streamlit as st
import plotly.express as px
from PIL import Image
from pathlib import Path
import io
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_xray_screener
from src.config import SAMPLE_XRAYS_DIR

st.set_page_config(page_title="MediAssist — Chest X-Ray Analysis", page_icon="🩻", layout="wide")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩻 Chest X-Ray Screening & Grad-CAM Saliency")
st.caption("Module B: Deep Learning Multi-Label Pulmonary Screening with Attention Heatmaps")

xray_screener = get_xray_screener()

# Image selection: Upload or Sample
col_mode, col_thresh = st.columns([3, 1])
with col_mode:
    source_choice = st.radio("Choose Radiograph Source:", ["Select from Curated Radiograph Cohort", "Upload Medical Image (PNG/JPEG/TIFF)"], horizontal=True)

with col_thresh:
    threshold = st.slider("Screening Threshold:", min_value=0.20, max_value=0.80, value=0.40, step=0.05)

image_bytes = None
selected_name = "radiograph"

if source_choice == "Select from Curated Radiograph Cohort":
    sample_files = list(SAMPLE_XRAYS_DIR.glob("*.png"))
    file_map = {f.name: f for f in sample_files}
    if file_map:
        selected_name = st.selectbox("Curated Radiograph Sample:", list(file_map.keys()))
        with open(file_map[selected_name], "rb") as f:
            image_bytes = f.read()
    else:
        st.warning("Sample radiographs directory is empty. Run `initialize_sample_data()`.")
else:
    uploaded = st.file_uploader("Upload Chest Radiograph (PA/AP view):", type=["png", "jpg", "jpeg", "tiff"])
    if uploaded is not None:
        image_bytes = uploaded.read()
        selected_name = uploaded.name

if image_bytes is not None:
    st.markdown("---")
    # Run Inference
    with st.spinner("Analyzing radiograph density and computing Grad-CAM saliency..."):
        result = xray_screener.analyze_radiograph(image_bytes, threshold=threshold)

    if not result.get("success"):
        st.error(f"Image analysis error: {result.get('error')}")
    else:
        # 10-Second Clinician Explanation Card
        st.subheader("⏱️ Clinician Radiologic Summary (10-Second Summary)")
        exp = result["explanation"]
        st.markdown(f"""
        > **Finding:** {exp['finding']}  
        > **Reason:** {exp['reason']}  
        > **Confidence:** {exp['confidence']}  
        > **Limitation:** {exp['limitation']}
        """)

        # Visuals: Side-by-side Original vs Grad-CAM
        col_orig, col_grad = st.columns(2)
        with col_orig:
            st.markdown("#### 🖼️ Original Input Radiograph")
            if "original_image" in result and result["original_image"] is not None:
                st.image(result["original_image"], use_column_width=True, caption=f"Input: {selected_name}")
            else:
                st.image(image_bytes, use_column_width=True, caption=f"Input: {selected_name}")

        with col_grad:
            st.markdown("#### 🔬 Grad-CAM Saliency Map (Model Attention)")
            if result.get("gradcam_available") and result.get("gradcam_image") is not None:
                st.image(result["gradcam_image"], use_column_width=True, caption=f"Attention heatmap for top class: {result.get('top_class')}")
            else:
                st.info("Grad-CAM visualization available when pathology activations exceed background baseline.")

        st.markdown("---")
        # Pathology Probabilities Bar Chart
        st.subheader("📊 Multi-Label Pathology Screening Probabilities")
        probs = result.get("probabilities", {})
        if probs:
            prob_list = [{"Pathology": k, "Probability": v, "Flagged": v >= threshold} for k, v in probs.items()]
            fig = px.bar(
                prob_list,
                x="Probability",
                y="Pathology",
                orientation="h",
                color="Flagged",
                color_discrete_map={True: "#ef4444", False: "#64748b"},
                text_auto=".1%",
                range_x=[0, 1]
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"Threshold ({threshold:.0%})")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ **Diagnostic Limitation Note:** Grad-CAM displays the mathematical gradient of model activations. It does not replace a board-certified radiologist's formal reading.")
