"""
MediAssist — Page 6: Explainability Center
LIME Text Interpretability, Grad-CAM Saliency, Feature Importance & 10-Second Explanations
"""
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_triage_engine, get_xray_screener, get_explainability_engine
from src.utils import DEMO_PATIENTS, generate_anatomical_xray

st.set_page_config(page_title="MediAssist — Explainability Center", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Clinical AI Explainability Center")
st.caption("Module E: Comprehensive Interpretability Suite (LIME, Grad-CAM, Attribution, and 10-Second Explanations)")

triage_engine = get_triage_engine()
xray_screener = get_xray_screener()
exp_engine = get_explainability_engine()

tab_nlp, tab_cv, tab_summary = st.tabs([
    "📝 NLP & Text Explainability (LIME)",
    "🩻 Vision Explainability (Grad-CAM)",
    "⏱️ 10-Second Explanations Showcase"
])

# TAB 1: NLP LIME EXPLAINABILITY
with tab_nlp:
    st.subheader("Local Interpretable Model-agnostic Explanations (LIME) for Symptom Triage")
    st.markdown("""
    LIME builds a local linear approximation around the patient's symptom narrative, identifying which specific words contributed toward or against the predicted clinical urgency category.
    """)

    sample_text = st.selectbox(
        "Choose Symptom Narrative to Interpret:",
        [
            "Crushing substernal chest pain radiating to left arm with profuse cold sweating and nausea",
            "High fever of 39 C for 3 days with productive cough, yellow phlegm, and localized pleuritic pain",
            "Mild dry cough and runny nose for 4 days with occasional sneezing and low grade fever",
            "Sudden onset of right-sided arm and leg weakness with facial drooping and slurred speech"
        ]
    )

    if st.button("Generate LIME Word Importance 🔎", key="btn_lime"):
        with st.spinner("Computing perturbation-based LIME token weights..."):
            lime_res = exp_engine.explain_text_with_lime(sample_text, triage_engine.model, triage_engine.vectorizer)

        if lime_res and lime_res.get("weights"):
            st.success(f"**Predicted Urgency Class:** {lime_res['predicted_label']}")
            lime_df = [{"Word/Phrase": w[0], "Weight (Contribution)": round(w[1], 4)} for w in lime_res["weights"]]
            fig = px.bar(
                lime_df,
                x="Weight (Contribution)",
                y="Word/Phrase",
                orientation="h",
                color="Weight (Contribution)",
                color_continuous_scale="RdBu_r",
                title=f"Token Attribution for '{lime_res['predicted_label']}'"
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LIME explanation ready; ensure model checkpoint is trained.")

# TAB 2: VISION GRAD-CAM EXPLAINABILITY
with tab_cv:
    st.subheader("Gradient-weighted Class Activation Mapping (Grad-CAM) for Chest Radiographs")
    st.markdown("""
    Grad-CAM computes the gradients of the target thoracic pathology score with respect to the final convolutional layer of MobileNetV2, highlighting the specific anatomical regions of interest.
    """)

    pathology_choice = st.selectbox("Select Radiograph Pathology Demonstration:", ["Cardiomegaly", "Effusion", "Infiltration", "Normal"])

    if st.button("Run Grad-CAM Saliency Analysis 🩻", key="btn_gradcam"):
        with st.spinner(f"Extracting feature maps and computing gradients for {pathology_choice}..."):
            img = generate_anatomical_xray(pathology_choice.lower())
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result = xray_screener.analyze_radiograph(buf.getvalue())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Input Radiograph")
            st.image(img, use_column_width=True)
        with c2:
            st.markdown(f"#### Grad-CAM Saliency: {result.get('top_class')}")
            if result.get("gradcam_available"):
                st.image(result["gradcam_image"], use_column_width=True)
            else:
                st.info("Feature activations within baseline threshold.")

        st.caption("Grad-CAM highlights CNN feature attention. It is a screening visualization, not definitive histological or margins demarcation.")

# TAB 3: 10-SECOND CLINICIAN EXPLANATIONS
with tab_summary:
    st.subheader("Clinician Decision Support: 10-Second Explanations")
    st.markdown("""
    Every AI and rule-generated decision in MediAssist adheres strictly to a standardized 4-part structure that a physician can read and assimilate in under 10 seconds:
    """)

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown("""
        #### 1. Symptom Triage Module
        > **Finding:** Immediate Emergency Medical Evaluation Required.  
        > **Reason:** Overt life-threatening warning signs detected: crushing chest pain radiating to left arm.  
        > **Confidence:** Deterministic clinical safety rule (100% rule-level priority).  
        > **Limitation:** Screening triage flag; does not constitute definitive diagnosis.
        """)

        st.markdown("""
        #### 2. Laboratory Interpreter Module
        > **Finding:** Potassium result is Critical High (6.8 mEq/L).  
        > **Reason:** Exceeds physiological boundary: reference is 3.5 - 5.0 mEq/L. Risk of lethal arrhythmia.  
        > **Confidence:** Authoritative Clinical Reference Interval Rule (Tietz/Harrison).  
        > **Limitation:** Requires stat specimen re-confirmation to rule out in vitro hemolysis.
        """)

    with ex2:
        st.markdown("""
        #### 3. Chest X-Ray Screening Module
        > **Finding:** Potential Cardiomegaly Detected (Probability: 88.5%).  
        > **Reason:** CNN detected cardiac silhouette enlargement exceeding standard cardiothoracic ratio.  
        > **Confidence:** Model probability is 88.5% (Screening Threshold: 40%).  
        > **Limitation:** Grad-CAM displays model attention, not a formal radiologist reading.
        """)

        st.markdown("""
        #### 4. Composite Risk Aggregator
        > **Finding:** Patient Priority: Emergency Review (Score: 92.0/100).  
        > **Reason:** Immediate life-safety triggers: Severe chest pain, elevated Troponin I, tachycardia.  
        > **Confidence:** Rule-governed multi-modal clinical aggregation (transparent weights).  
        > **Limitation:** Prioritization score assists clinical workflow; does not predict long-term mortality.
        """)
