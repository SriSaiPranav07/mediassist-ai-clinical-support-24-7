"""
MediAssist — Page 3: Multimodal Chest X-Ray & Symptom Clinical Decision Support
Author: MediAssist Clinical AI Team

Combines:
  - Structured Patient Demographics, Medical History & Vital Signs
  - Contextual NLP Symptom Parsing (Negation, Temporality, Severity)
  - Deep Learning Multi-Label Chest Radiograph Screening (MobileNetV2)
  - Grad-CAM Feature Attribution Heatmaps
  - Multimodal Concordance & Discordance Clinical Rules Engine
  - Independent Safety Screening & Emergency Red-Flag Overrides
  - Professional PDF Consultation Report Export
  - Model Performance & Evaluation Dashboard
"""
import io
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_loader import get_xray_screener
from src.config import SAMPLE_XRAYS_DIR, XRAY_CLASSES, EMERGENCY_CONTACT_INDIA, MODELS_DIR
from src.clinical_guardrails import ClinicalNarrativeParser
from src.multimodal_integration import MultimodalClinicalIntegrator
from src.data_validation import validate_vital_signs
from src.utils import DEMO_PATIENTS
from utils.data_processing import validate_radiograph_image
try:
    from utils.pdf_generator import generate_multimodal_xray_pdf
except Exception:
    def generate_multimodal_xray_pdf(report):
        return b"PDF export unavailable."

st.set_page_config(
    page_title="MediAssist — Multimodal Chest X-Ray & Symptom Analysis",
    page_icon="🩻",
    layout="wide"
)

# Hospital-grade CSS
st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child { display: none; }
    .status-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }
    .concordant-box {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .discordant-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .clinical-callout {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩻 Multimodal Chest X-Ray & Symptom Clinical Decision Support")
st.caption("Integrated Artificial Intelligence: MobileNetV2 Radiograph Screening, Grad-CAM Saliency, Contextual NLP & Clinical Concordance Architecture")

xray_screener = get_xray_screener()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE & PRESET MANAGER
# ─────────────────────────────────────────────────────────────────────────────
def set_multimodal_case(case_id: str):
    if case_id == "Custom":
        st.session_state["mm_age"] = 45
        st.session_state["mm_sex"] = "Male"
        st.session_state["mm_narrative"] = ""
        st.session_state["mm_symptoms"] = []
        st.session_state["mm_duration"] = 2.0
        st.session_state["mm_severity"] = 4
        st.session_state["mm_trajectory"] = "Unchanged / Static"
        st.session_state["mm_history"] = []
        st.session_state["mm_smoking"] = "Non-smoker"
        st.session_state["mm_meds"] = ""
        st.session_state["mm_allergies"] = ""
        st.session_state["mm_enable_vitals"] = False
        st.session_state["mm_temp"] = 37.0
        st.session_state["mm_spo2"] = 98
        st.session_state["mm_hr"] = 75
        st.session_state["mm_rr"] = 16
        st.session_state["mm_bp_sys"] = 120
        st.session_state["mm_bp_dia"] = 80
        st.session_state["mm_projection"] = "PA view"
        st.session_state["mm_sample_img"] = "normal_sample.png"
    else:
        pt = next((p for p in DEMO_PATIENTS if p["id"] == case_id), None)
        if pt:
            st.session_state["mm_age"] = int(pt.get("age", 45))
            st.session_state["mm_sex"] = pt.get("sex", "Male")
            st.session_state["mm_narrative"] = pt.get("chief_complaint", "")
            st.session_state["mm_duration"] = float(pt.get("duration_days", 2.0))
            st.session_state["mm_severity"] = int(pt.get("severity", 5))
            v = pt.get("vitals", {})
            st.session_state["mm_enable_vitals"] = bool(v)
            st.session_state["mm_temp"] = float(v.get("Temperature", 37.0))
            st.session_state["mm_spo2"] = int(v.get("SpO2", 98))
            st.session_state["mm_hr"] = int(v.get("HeartRate", 75))
            st.session_state["mm_rr"] = int(v.get("RespiratoryRate", 16))
            st.session_state["mm_bp_sys"] = int(v.get("SystolicBP", 120))
            st.session_state["mm_bp_dia"] = int(v.get("DiastolicBP", 80))
            st.session_state["mm_sample_img"] = pt.get("xray_sample", "normal_sample.png")
            st.session_state["mm_projection"] = "PA view"

            if pt["id"] == "PT-101":
                st.session_state["mm_symptoms"] = ["Chest pain", "Shortness of breath", "Fatigue"]
                st.session_state["mm_trajectory"] = "Worsening rapidly"
                st.session_state["mm_history"] = ["Coronary Artery Disease (CAD)", "Hypertension"]
                st.session_state["mm_smoking"] = "Current smoker"
                st.session_state["mm_meds"] = "Aspirin, Atorvastatin"
            elif pt["id"] == "PT-102":
                st.session_state["mm_symptoms"] = ["Fever", "Productive cough", "Shortness of breath", "Chills"]
                st.session_state["mm_trajectory"] = "Worsening gradually"
                st.session_state["mm_history"] = ["Asthma / COPD"]
                st.session_state["mm_smoking"] = "Former smoker"
                st.session_state["mm_meds"] = "Albuterol inhaler"
            elif pt["id"] == "PT-103":
                st.session_state["mm_symptoms"] = ["Dry cough"]
                st.session_state["mm_trajectory"] = "Improving"
                st.session_state["mm_history"] = []
                st.session_state["mm_smoking"] = "Non-smoker"
                st.session_state["mm_meds"] = ""
            else:
                st.session_state["mm_symptoms"] = []
                st.session_state["mm_trajectory"] = "Unchanged / Static"
                st.session_state["mm_history"] = []
                st.session_state["mm_smoking"] = "Non-smoker"
                st.session_state["mm_meds"] = ""

if "mm_age" not in st.session_state:
    set_multimodal_case("Custom")

# Quick Demo Preset Buttons
st.markdown("##### ⚡ Quick Clinical Multimodal Scenarios")
b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("🫁 Pneumonia Cohort (PT-102)", use_container_width=True):
        set_multimodal_case("PT-102")
        st.rerun()
with b2:
    if st.button("❤️ Cardiomegaly / ACS (PT-101)", use_container_width=True):
        set_multimodal_case("PT-101")
        st.rerun()
with b3:
    if st.button("🟢 Routine Ambulatory (PT-103)", use_container_width=True):
        set_multimodal_case("PT-103")
        st.rerun()
with b4:
    if st.button("✍️ Blank / Custom Input", use_container_width=True):
        set_multimodal_case("Custom")
        st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ONE: PROFESSIONAL PATIENT INFORMATION FORM
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 1. Patient Demographics, History & Clinical Presentation")

with st.expander("👤 Patient Demographics & Medical History", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        pt_age = st.number_input("Patient Age (Years):", min_value=0, max_value=120, value=int(st.session_state.get("mm_age", 45)))
    with col_p2:
        pt_sex = st.selectbox(
            "Biological Sex:",
            ["Male", "Female", "Other", "Prefer not to disclose"],
            index=["Male", "Female", "Other", "Prefer not to disclose"].index(st.session_state.get("mm_sex", "Male"))
        )
    with col_p3:
        pt_smoking = st.selectbox(
            "Smoking History:",
            ["Non-smoker", "Current smoker", "Former smoker", "Passive / Secondary exposure"],
            index=["Non-smoker", "Current smoker", "Former smoker", "Passive / Secondary exposure"].index(st.session_state.get("mm_smoking", "Non-smoker"))
        )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        pt_history = st.multiselect(
            "Medical History / Pre-existing Respiratory & Cardiac Conditions:",
            [
                "Asthma / COPD", "Previous pneumonia", "Tuberculosis history",
                "Coronary Artery Disease (CAD)", "Congestive Heart Failure",
                "Hypertension", "Type 2 Diabetes", "Immunocompromised / Cancer",
                "Recent long-haul travel / surgery", "None"
            ],
            default=st.session_state.get("mm_history", [])
        )
    with col_h2:
        pt_meds = st.text_input("Current Medications (Voluntary):", value=st.session_state.get("mm_meds", ""), placeholder="e.g. Inhalers, Beta-blockers, ACE-inhibitors")

with st.expander("📝 Reported Symptoms & Temporal Chronology", expanded=True):
    pt_narrative = st.text_area(
        "Symptom Narrative (Patient's own words or clinical intake):",
        value=st.session_state.get("mm_narrative", ""),
        height=100,
        placeholder="e.g. 'I have had productive cough with fever for 4 days and progressive shortness of breath, but no chest pain.'"
    )

    pt_structured_syms = st.multiselect(
        "Select Specific Presenting Symptoms:",
        [
            "Cough", "Dry cough", "Productive cough", "Fever", "Chest pain",
            "Shortness of breath", "Difficulty breathing", "Wheezing",
            "Fatigue", "Chills", "Sputum production", "Blood in sputum (Hemoptysis)",
            "Pleuritic pain (pain on breathing)", "Orthopnea (breathless lying flat)", "Ankle swelling"
        ],
        default=st.session_state.get("mm_symptoms", [])
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        pt_duration = st.number_input("Symptom Duration (Days):", min_value=0.01, max_value=365.0, value=float(st.session_state.get("mm_duration", 2.0)), step=0.5)
    with col_s2:
        pt_severity = st.slider("Reported Symptom Severity (0 to 10):", min_value=0, max_value=10, value=int(st.session_state.get("mm_severity", 4)))
    with col_s3:
        pt_trajectory = st.selectbox(
            "Symptom Progression / Trajectory:",
            ["Worsening rapidly", "Worsening gradually", "Unchanged / Static", "Improving"],
            index=["Worsening rapidly", "Worsening gradually", "Unchanged / Static", "Improving"].index(st.session_state.get("mm_trajectory", "Unchanged / Static"))
        )

with st.expander("🩺 Physiological Vital Signs (Optional — Enable if measured)", expanded=st.session_state.get("mm_enable_vitals", False)):
    enable_vitals = st.checkbox("Vital signs recorded at evaluation", value=st.session_state.get("mm_enable_vitals", False))
    vitals_dict = None
    if enable_vitals:
        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        with vc1:
            v_temp = st.number_input("Body Temp (°C):", min_value=30.0, max_value=44.0, value=float(st.session_state.get("mm_temp", 37.0)), step=0.1)
        with vc2:
            v_spo2 = st.number_input("SpO2 (%):", min_value=40, max_value=100, value=int(st.session_state.get("mm_spo2", 98)), step=1)
        with vc3:
            v_hr = st.number_input("Heart Rate (bpm):", min_value=20, max_value=260, value=int(st.session_state.get("mm_hr", 75)), step=1)
        with vc4:
            v_rr = st.number_input("Resp Rate (/min):", min_value=4, max_value=70, value=int(st.session_state.get("mm_rr", 16)), step=1)
        with vc5:
            v_sys = st.number_input("Systolic BP (mmHg):", min_value=40, max_value=300, value=int(st.session_state.get("mm_bp_sys", 120)), step=1)

        vitals_dict = {
            "Temperature": v_temp,
            "SpO2": v_spo2,
            "HeartRate": v_hr,
            "RespiratoryRate": v_rr,
            "SystolicBP": v_sys
        }
        v_ok, v_errs = validate_vital_signs(vitals_dict)
        if not v_ok:
            for ve in v_errs:
                st.warning(f"⚠️ {ve}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE TWO: CHEST X-RAY UPLOAD AND IMAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🖼️ 2. Chest Radiograph Upload & Projection Settings")

col_img_mode, col_proj, col_thresh = st.columns([2, 1, 1])
with col_img_mode:
    source_choice = st.radio("Radiograph Input Source:", ["Select from Curated Radiograph Cohort", "Upload Medical Image (PNG, JPG, TIFF)"], horizontal=True)
with col_proj:
    projection_view = st.selectbox(
        "Image Projection:",
        ["PA view", "AP view", "Lateral view", "Unknown"],
        index=["PA view", "AP view", "Lateral view", "Unknown"].index(st.session_state.get("mm_projection", "PA view")),
        help="Posteroanterior (PA) is standard for cardiomegaly assessment. AP magnifies the cardiac silhouette."
    )
with col_thresh:
    threshold = st.slider("Screening Threshold:", min_value=0.20, max_value=0.80, value=0.40, step=0.05)

image_bytes = None
image_filename = "radiograph"

if source_choice == "Select from Curated Radiograph Cohort":
    sample_files = list(SAMPLE_XRAYS_DIR.glob("*.png"))
    file_map = {f.name: f for f in sample_files}
    if file_map:
        default_file = st.session_state.get("mm_sample_img", "normal_sample.png")
        file_keys = list(file_map.keys())
        default_idx = file_keys.index(default_file) if default_file in file_keys else 0
        selected_file_name = st.selectbox("Curated Radiograph Sample:", file_keys, index=default_idx)
        with open(file_map[selected_file_name], "rb") as f:
            image_bytes = f.read()
        image_filename = selected_file_name
else:
    uploaded = st.file_uploader("Upload Radiograph Image:", type=["png", "jpg", "jpeg", "tiff"])
    if uploaded is not None:
        image_bytes = uploaded.read()
        image_filename = uploaded.name

# Image Pre-validation & Quality Inspection
validation_passed = False
ood_metadata = {}

if image_bytes is not None:
    is_valid, err_msg, meta = validate_radiograph_image(image_bytes)
    ood_metadata = meta
    col_prev, col_status = st.columns([1, 2])
    with col_prev:
        try:
            preview_img = Image.open(io.BytesIO(image_bytes))
            st.image(preview_img, caption=f"Preview: {image_filename}", use_column_width=True)
        except Exception:
            st.error("Cannot decode preview image.")
    with col_status:
        st.markdown("#### 🔍 Image Quality & OOD Validation Status")
        if is_valid:
            validation_passed = True
            st.success("✅ **Image Quality Status: Suitable for Processing**")
            st.write(f"• Resolution: `{meta.get('width')} x {meta.get('height')} px` | Format: `{meta.get('format')}`")
            st.write(f"• Aspect Ratio: `{meta.get('aspect_ratio')}` (Within standard PA/AP boundary 0.5–2.0)")
            st.write(f"• Chromatic Divergence: `{meta.get('chromatic_divergence', 0.0)}` (Monochrome transmission verified)")
            st.caption("ℹ️ *Quality validation verifies digital file integrity and transmission contrast; it is not proof that the film is clinically diagnostic.*")
        else:
            validation_passed = False
            st.error(f"❌ **Image Processing Rejected: {err_msg}**")
            if meta:
                st.write(f"• Metrics: `{meta}`")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE THREE, FOUR & FIVE: MULTIMODAL INFERENCE & INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
analyze_btn = st.button("Execute Multimodal Clinical Analysis 🚀", type="primary", disabled=(image_bytes is None or not validation_passed))

if analyze_btn and image_bytes is not None and validation_passed:
    with st.spinner("Executing MobileNetV2 inference, Grad-CAM extraction, and clinical concordance rules..."):
        # 1. Image Screening & Grad-CAM
        xray_res = xray_screener.analyze_radiograph(image_bytes, threshold=threshold)

        # 2. Contextual Symptom Parsing
        parsed_narrative = ClinicalNarrativeParser.parse(pt_narrative)
        # Combine explicitly checked symptoms with narrative-extracted symptoms
        all_current = list(set([s.lower() for s in pt_structured_syms] + parsed_narrative["current_symptoms"]))

        symptom_payload = {
            "narrative": pt_narrative,
            "current_symptoms": all_current,
            "negated_symptoms": parsed_narrative["negated_symptoms"],
            "duration_days": pt_duration,
            "severity": pt_severity,
            "trajectory": pt_trajectory
        }

        patient_payload = {
            "age": pt_age,
            "sex": pt_sex,
            "medical_history": pt_history,
            "smoking_history": pt_smoking,
            "medications": pt_meds
        }

        # 3. Multimodal Integration Engine
        mm_report = MultimodalClinicalIntegrator.synthesize_report(
            patient_info=patient_payload,
            symptom_data=symptom_payload,
            vital_signs=vitals_dict,
            xray_result=xray_res,
            projection_view=projection_view
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE SEVEN: PROFESSIONAL COMBINED ANALYSIS REPORT
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📑 Multimodal Clinical Decision Support Report")

    # Status Banner
    st.markdown(f"""
    <div class="status-card" style="background-color: {mm_report['status_color']}15; border: 1.5px solid {mm_report['status_color']};">
        <h3 style="margin:0; color: {mm_report['status_color']};">{mm_report['status_badge']}</h3>
        <p style="margin:6px 0 0 0; font-weight: 500;">{mm_report['action_advisory']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Emergency Warning Triggers (if any)
    if mm_report["emergency_triggers"]:
        st.error("🚨 **CRITICAL LIFE-SAFETY WARNING TRIGGERS IDENTIFIED:**")
        for et in mm_report["emergency_triggers"]:
            st.error(f"• **{et}**")
        st.error(f"📞 **Immediate National Emergency Services: Dial {EMERGENCY_CONTACT_INDIA} in India.**")

    # Report Tabs
    tab_report, tab_gradcam, tab_eval, tab_export = st.tabs([
        "📋 Combined Analysis Report",
        "🔬 Grad-CAM Saliency Maps",
        "📊 Model Evaluation & Metrics",
        "📄 Export Clinical PDF"
    ])

    # ── TAB 1: COMBINED ANALYSIS REPORT ──────────────────────────────────────
    with tab_report:
        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            st.markdown("#### A. Patient & Symptom Analysis")
            st.markdown(f"""
            <div class="clinical-callout">
                <strong>Demographics:</strong> {pt_age} years | {pt_sex} | {pt_smoking}<br>
                <strong>Duration & Severity:</strong> {pt_duration} day(s) | Severity {pt_severity}/10 ({pt_trajectory})<br>
                <strong>Current Symptoms:</strong> {', '.join(all_current) if all_current else 'None specified'}<br>
                <strong>Negated / Denied:</strong> {', '.join(parsed_narrative['negated_symptoms']) if parsed_narrative['negated_symptoms'] else 'None reported'}<br>
                <strong>Known History:</strong> {', '.join(pt_history) if pt_history else 'None reported'}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### B. Radiograph Screening Summary")
            flagged = mm_report["flagged_pathologies"]
            probs = mm_report["radiology_probabilities"]
            top_cls = mm_report["top_radiology_class"]

            st.markdown(f"""
            <div class="clinical-callout">
                <strong>Primary Finding:</strong> {mm_report['primary_radiology_finding']}<br>
                <strong>Top Pathology Class:</strong> {top_cls} ({probs.get(top_cls, 0.0)*100:.1f}%)<br>
                <strong>Screening Threshold:</strong> {threshold*100:.0f}%<br>
                <strong>Projection:</strong> {projection_view}
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown("#### C. Multimodal Concordance & Discordance Analysis")

            # Concordance Insights
            if mm_report["concordant_insights"]:
                st.markdown("**🟢 Concordant Clinical Correlations:**")
                for c in mm_report["concordant_insights"]:
                    st.markdown(f"""
                    <div class="concordant-box">
                        <strong>{c['condition']}</strong> (Pathology: <em>{c['pathology']}</em>)<br>
                        {c['synthesis']}
                    </div>
                    """, unsafe_allow_html=True)

            # Discordance Insights
            if mm_report["discordant_insights"]:
                st.markdown("**🔴 Critical Clinical Discordances / Discrepancies:**")
                for d in mm_report["discordant_insights"]:
                    st.markdown(f"""
                    <div class="discordant-box">
                        <strong>{d['clinical_concern']}</strong> (<em>{d['type']}</em>)<br>
                        {d['explanation']}
                    </div>
                    """, unsafe_allow_html=True)

            if not mm_report["concordant_insights"] and not mm_report["discordant_insights"]:
                st.info("No significant acute concordance or critical discordance flagged between clinical symptoms and radiograph screening.")

        st.markdown("---")
        col_next, col_miss = st.columns([1, 1])

        with col_next:
            st.markdown("#### D. Suggested Non-Prescriptive Next Steps")
            for step in mm_report["suggested_next_steps"]:
                st.write(f"• {step}")

        with col_miss:
            st.markdown("#### E. Missing Clinical Information")
            for m in mm_report["missing_information"]:
                st.write(f"• {m}")

        st.markdown("---")
        st.caption(f"⚠️ **Clinical Decision Support Notice:** {mm_report['limitations_notice']}")

    # ── TAB 2: GRAD-CAM SALIENCY EXPLAINABILITY ──────────────────────────────
    with tab_gradcam:
        st.markdown("### 🔬 Local Convolutional Feature Attention (Grad-CAM)")
        st.caption("Gradient-weighted Class Activation Mapping displays feature map regions in the final convolutional layer influencing model prediction.")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### Original Input Radiograph")
            if "original_image" in xray_res and xray_res["original_image"] is not None:
                st.image(xray_res["original_image"], use_column_width=True, caption=f"Input: {image_filename}")
            else:
                st.image(image_bytes, use_column_width=True, caption=f"Input: {image_filename}")

        with col_g2:
            st.markdown(f"#### Grad-CAM Saliency Overlay ({top_cls})")
            if xray_res.get("gradcam_available") and xray_res.get("gradcam_image") is not None:
                st.image(xray_res["gradcam_image"], use_column_width=True, caption=f"Attention heatmap for top class: {top_cls}")
            else:
                st.info("Grad-CAM overlay available when model weights file is present and gradient signals exceed baseline.")

        # Multi-label probability distribution chart
        st.markdown("#### Multi-Label Pathology Screening Distribution")
        p_list = [{"Pathology": k, "Probability": v, "Flagged": v >= threshold} for k, v in probs.items()]
        fig_prob = px.bar(
            p_list,
            x="Probability",
            y="Pathology",
            orientation="h",
            color="Flagged",
            color_discrete_map={True: "#ef4444", False: "#64748b"},
            text_auto=".1%",
            range_x=[0, 1]
        )
        fig_prob.add_vline(x=threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"Screening Cutoff ({threshold:.0%})")
        fig_prob.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_prob, use_container_width=True)

        st.caption("ℹ️ *Important Explainability Disclaimer: Grad-CAM highlights mathematical neural network gradient activations. It does not demarcate exact histological pathology or confirm anatomical disease.*")

    # ── TAB 3: MODEL EVALUATION & METRICS DASHBOARD ──────────────────────────
    with tab_eval:
        st.markdown("### 📊 MobileNetV2 Radiograph Screener: Empirical Evaluation")
        st.caption("Held-out multi-label test set evaluation metrics computed on benchmark cohort:")

        eval_path = MODELS_DIR / "xray_evaluation.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            eval_rows = []
            for cls_name, m_dict in eval_data.items():
                eval_rows.append({
                    "Pathology Class": cls_name,
                    "Sensitivity (Recall)": f"{m_dict.get('Sensitivity', 0.0)*100:.1f}%",
                    "Specificity": f"{m_dict.get('Specificity', 0.0)*100:.1f}%",
                    "Precision": f"{m_dict.get('Precision', 0.0)*100:.1f}%",
                    "F1-Score": f"{m_dict.get('F1_Score', 0.0):.3f}",
                    "ROC-AUC": f"{m_dict.get('ROC_AUC', 0.0):.3f}"
                })

            st.dataframe(eval_rows, use_container_width=True)

            # ROC-AUC Bar Comparison
            auc_df = [{"Class": k, "ROC_AUC": float(v.get("ROC_AUC", 0.0))} for k, v in eval_data.items()]
            fig_auc = px.bar(
                auc_df,
                x="Class",
                y="ROC_AUC",
                range_y=[0, 1.05],
                title="Per-Class Area Under ROC Curve (ROC-AUC)",
                color="ROC_AUC",
                color_continuous_scale="Blues",
                text_auto=".3f"
            )
            fig_auc.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_auc, use_container_width=True)
        else:
            st.info("Evaluation metrics file pending. Run `python training/train_xray.py`.")

        st.warning("""
        **Evaluation & External Validation Limitations:**  
        • Metrics reflect test performance on the curated evaluation cohort; external domain shifts (differing X-ray tube voltages, patient positioning, motion artifacts) can affect generalization.  
        • False-positive and false-negative rates require in-person clinical auscultation and physician oversight.
        """)

    # ── TAB 4: EXPORT CLINICAL PDF ───────────────────────────────────────────
    with tab_export:
        st.markdown("### 📄 Export Multimodal Clinical Consultation PDF")
        st.caption("Generate a professional, downloadable consultation summary report using ReportLab:")

        pdf_bytes = generate_multimodal_xray_pdf(mm_report)

        st.download_button(
            label="📥 Download Multimodal Consultation PDF Report",
            data=pdf_bytes,
            file_name=f"MediAssist_Multimodal_Report_{pt_age}yo_{top_cls}.pdf",
            mime="application/pdf",
            help="Generates an in-memory PDF with zero permanent disk retention."
        )

        st.markdown("---")
        # Text version fallback
        text_summary = f"""MEDIASSIST MULTIMODAL CHEST X-RAY & SYMPTOM SUMMARY
Decision Status: {mm_report['decision_status']}
Clinical Advisory: {mm_report['action_advisory']}
Patient: {pt_age}yo {pt_sex} | Duration: {pt_duration}d | Severity: {pt_severity}/10
Current Symptoms: {', '.join(all_current)}
Negated Symptoms: {', '.join(parsed_narrative['negated_symptoms'])}
Radiograph Finding: {mm_report['primary_radiology_finding']} (Top class: {top_cls})
Concordant Insights: {len(mm_report['concordant_insights'])}
Discordant Insights: {len(mm_report['discordant_insights'])}
======================================================================
"""
        st.download_button(
            label="📄 Download Text Summary (.txt)",
            data=text_summary,
            file_name="MediAssist_Multimodal_Summary.txt",
            mime="text/plain"
        )
