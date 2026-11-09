"""
MediAssist — AI Clinical Decision Support System 🩺
Main Application Entry Point & Multi-Step Clinical Wizard
Zelvora Technologies Pvt. Ltd. / Zelvora Industry Labs AI Capstone
"""
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# Core Modules & Utilities
from models import (
    get_triage_model,
    get_xray_model,
    get_lab_model,
    get_risk_engine,
    get_rag_engine
)
from utils.data_processing import (
    validate_symptom_narrative,
    validate_radiograph_image,
    sanitize_in_memory
)
from utils.xai_visuals import (
    generate_lime_text_explanation,
    render_xai_comparison_columns
)
from utils.pdf_generator import generate_clinical_summary_pdf
from src.utils import DEMO_PATIENTS
from src.config import LAB_REFERENCE_RANGES, EMERGENCY_CONTACT_INDIA

# Page Configuration
st.set_page_config(
    page_title="MediAssist — Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Clinical Styling & Seamless CSS
st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .clinical-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.90rem;
    }
    .security-badge {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .explanation-box {
        background-color: #ffffff;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 0.8rem 0;
    }
    .step-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION (WIZARD & ZERO DATA RETENTION)
# ==============================================================================
if "wizard_step" not in st.session_state:
    st.session_state["wizard_step"] = 1  # 1: Data Entry, 2: AI Processing, 3: Results, 4: Export

if "disclaimer_acknowledged" not in st.session_state:
    st.session_state["disclaimer_acknowledged"] = False

if "patient_name" not in st.session_state:
    st.session_state["patient_name"] = "Harish Sharma (PT-001)"

if "chief_complaint" not in st.session_state:
    st.session_state["chief_complaint"] = DEMO_PATIENTS[0]["chief_complaint"]

if "duration_days" not in st.session_state:
    st.session_state["duration_days"] = float(DEMO_PATIENTS[0]["duration_days"])

if "severity_score" not in st.session_state:
    st.session_state["severity_score"] = int(DEMO_PATIENTS[0]["severity"])

if "patient_age" not in st.session_state:
    st.session_state["patient_age"] = int(DEMO_PATIENTS[0]["age"])

if "patient_sex" not in st.session_state:
    st.session_state["patient_sex"] = DEMO_PATIENTS[0]["sex"]

if "enable_vitals" not in st.session_state:
    st.session_state["enable_vitals"] = True

if "v_hr" not in st.session_state:
    st.session_state["v_hr"] = 98

if "v_spo2" not in st.session_state:
    st.session_state["v_spo2"] = 96

if "v_sys" not in st.session_state:
    st.session_state["v_sys"] = 145

if "xray_bytes" not in st.session_state:
    # Preload default curated sample radiograph
    sample_path = Path(__file__).resolve().parent / "data" / "sample_xrays" / "cardiomegaly_sample.png"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.session_state["xray_bytes"] = f.read()
    else:
        st.session_state["xray_bytes"] = None

if "xray_source_name" not in st.session_state:
    st.session_state["xray_source_name"] = "cardiomegaly_sample.png (Curated Cohort)"

if "lab_values" not in st.session_state:
    st.session_state["lab_values"] = DEMO_PATIENTS[0].get("labs", {})

if "triage_result" not in st.session_state:
    st.session_state["triage_result"] = None

if "xray_result" not in st.session_state:
    st.session_state["xray_result"] = None

if "lab_result" not in st.session_state:
    st.session_state["lab_result"] = None

if "risk_result" not in st.session_state:
    st.session_state["risk_result"] = None

if "rag_result" not in st.session_state:
    st.session_state["rag_result"] = None


# Helper to load preset patient into session state
def load_preset_patient(patient_id: str):
    if patient_id == "Custom":
        st.session_state["patient_name"] = "Custom Consultation"
        st.session_state["chief_complaint"] = ""
        st.session_state["duration_days"] = 1.0
        st.session_state["severity_score"] = 4
        st.session_state["patient_age"] = 35
        st.session_state["patient_sex"] = "Male"
        st.session_state["enable_vitals"] = False
        st.session_state["v_hr"] = 75
        st.session_state["v_spo2"] = 98
        st.session_state["v_sys"] = 120
        st.session_state["lab_values"] = {}
        st.session_state["xray_bytes"] = None
        st.session_state["xray_source_name"] = "None"
    else:
        pt = next((p for p in DEMO_PATIENTS if p["id"] == patient_id), None)
        if pt:
            st.session_state["patient_name"] = f"{pt['name']} ({pt['id']})"
            st.session_state["chief_complaint"] = pt["chief_complaint"]
            st.session_state["duration_days"] = float(pt["duration_days"])
            st.session_state["severity_score"] = int(pt["severity"])
            st.session_state["patient_age"] = int(pt["age"])
            st.session_state["patient_sex"] = pt["sex"]
            
            v = pt.get("vitals", {})
            st.session_state["enable_vitals"] = bool(v)
            st.session_state["v_hr"] = int(v.get("HeartRate", 75))
            st.session_state["v_spo2"] = int(v.get("SpO2", 98))
            st.session_state["v_sys"] = int(v.get("SystolicBP", 120))
            st.session_state["lab_values"] = pt.get("labs", {})
            
            # Match sample X-ray if available
            xray_file = pt.get("xray_file")
            if xray_file:
                xray_path = Path(__file__).resolve().parent / "data" / "sample_xrays" / xray_file
                if xray_path.exists():
                    with open(xray_path, "rb") as f:
                        st.session_state["xray_bytes"] = f.read()
                    st.session_state["xray_source_name"] = f"{xray_file} (Pre-loaded)"


# Zero Data Retention Memory Flush
def flush_session_data():
    sanitize_in_memory(st.session_state.get("xray_bytes"))
    st.session_state["chief_complaint"] = ""
    st.session_state["xray_bytes"] = None
    st.session_state["lab_values"] = {}
    st.session_state["triage_result"] = None
    st.session_state["xray_result"] = None
    st.session_state["lab_result"] = None
    st.session_state["risk_result"] = None
    st.session_state["rag_result"] = None
    st.session_state["wizard_step"] = 1
    st.rerun()


# ==============================================================================
# SIDEBAR CONTROLS & REPOSITORY ARCHITECTURE
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=54)
    st.title("MediAssist")
    st.markdown("---")
    
    # Wizard Navigation
    st.markdown("### 🧭 Clinical Workflow Wizard")
    steps_map = {
        1: "1. 📋 Clinical Data Entry",
        2: "2. ⚡ AI Multi-Model Processing",
        3: "3. 📊 Results, XAI & Review",
        4: "4. 📄 Export Consultation PDF"
    }
    selected_step_name = st.radio(
        "Current Workflow Stage:",
        list(steps_map.values()),
        index=st.session_state["wizard_step"] - 1,
        key="wizard_radio"
    )
    # Sync radio back to wizard step number
    st.session_state["wizard_step"] = list(steps_map.values()).index(selected_step_name) + 1
    
    st.markdown("---")
    
    # Confidence Thresholding Slider (Task 2 Requirement)
    st.markdown("### 🎚️ Anomaly Confidence Threshold")
    conf_threshold = st.slider(
        "Filter Low-Confidence Flags:",
        min_value=10,
        max_value=90,
        value=40,
        step=5,
        help="Findings and abnormalities with confidence above this threshold will trigger actionable flags."
    )
    threshold_decimal = conf_threshold / 100.0
    
    st.markdown("---")
    
    # Zero Data Retention Status & Flush Button (Task 3 Requirement)
    st.markdown("### 🔒 Security & Privacy")
    st.caption("🛡️ **Zero Data Retention:** In-memory `io.BytesIO` streams only. Uploaded files are never persisted to disk or external databases.")
    if st.button("🗑️ Flush Memory & Reset Session", use_container_width=True):
        flush_session_data()
        
    st.markdown("---")


# ==============================================================================
# HEADER & MANDATORY DISMISSIBLE DISCLAIMER BANNER (Task 3 Requirement)
# ==============================================================================
st.markdown("""
<div class="clinical-banner">
    <h2 style="margin:0; font-size: 2.0rem;">🩺 MediAssist — AI Clinical Decision Support System</h2>
    <p style="margin: 0.3rem 0 0 0; font-size: 1.05rem; opacity: 0.95;">
        Multi-Model Healthcare AI: NLP Triage, MobileNetV2 Chest X-Ray, Lab Interpretation & RAG Guideline Grounding
    </p>
</div>
""", unsafe_allow_html=True)

# Zero Retention Informational Pill
st.markdown("""
<div class="security-badge">
    <span>🔒 <b>Zero Data Retention Compliant:</b> Patient data, symptom narratives, and radiographs are processed strictly in volatile ephemeral memory. No data is stored, cached to disk, or logged.</span>
</div>
""", unsafe_allow_html=True)

# Dismissible Disclaimer Banner
if not st.session_state["disclaimer_acknowledged"]:
    col_disc, col_btn = st.columns([5, 1])
    with col_disc:
        st.warning(
            "⚠️ **MANDATORY CLINICAL NOTICE & REGULATORY DISCLAIMER:** "
            "MediAssist is an educational and research prototype developed for demonstration purposes. "
            "It is **NOT** a certified medical device (FDA, CE, or CDSCO), does not independently formulate clinical diagnoses, "
            "and does not prescribe treatments. All decisions must be reviewed by a licensed physician. "
            "**In an acute emergency, dial 112 (India) or 911 (US) immediately.**"
        )
    with col_btn:
        if st.button("Acknowledge & Dismiss", use_container_width=True):
            st.session_state["disclaimer_acknowledged"] = True
            st.rerun()


# ==============================================================================
# STEP 1: CLINICAL DATA ENTRY (WIZARD STAGE 1)
# ==============================================================================
if st.session_state["wizard_step"] == 1:
    st.markdown("### 📋 Step 1: Patient Clinical Data Entry")
    st.write("Load a curated demonstration patient profile or enter custom clinical observations:")
    
    # Quick Preset Scenarios
    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns(5)
    with c_btn1:
        if st.button("🚨 STEMI Alert (PT-001)", use_container_width=True):
            load_preset_patient("PT-001")
            st.rerun()
    with c_btn2:
        if st.button("🧠 Acute Stroke (PT-005)", use_container_width=True):
            load_preset_patient("PT-005")
            st.rerun()
    with c_btn3:
        if st.button("🫁 Sepsis/Pneumonia (PT-002)", use_container_width=True):
            load_preset_patient("PT-002")
            st.rerun()
    with c_btn4:
        if st.button("🧪 Critical DKA (PT-003)", use_container_width=True):
            load_preset_patient("PT-003")
            st.rerun()
    with c_btn5:
        if st.button("✍️ Custom / Clear", use_container_width=True):
            load_preset_patient("Custom")
            st.rerun()
            
    st.markdown("---")
    
    # Demographics & Presentation Details
    col_demo1, col_demo2, col_demo3, col_demo4 = st.columns([2, 1, 1, 1])
    with col_demo1:
        st.session_state["patient_name"] = st.text_input("Patient Identifier / Name:", value=st.session_state["patient_name"])
    with col_demo2:
        st.session_state["patient_age"] = st.number_input("Age (Years):", min_value=0, max_value=120, value=st.session_state["patient_age"])
    with col_demo3:
        st.session_state["patient_sex"] = st.selectbox("Sex:", ["Male", "Female"], index=0 if st.session_state["patient_sex"] == "Male" else 1)
    with col_demo4:
        st.session_state["duration_days"] = st.number_input("Duration (Days):", min_value=0.01, max_value=365.0, value=st.session_state["duration_days"], step=0.5)

    # Narrative Complaint & Severity
    col_narrative, col_sev = st.columns([3, 1])
    with col_narrative:
        st.session_state["chief_complaint"] = st.text_area(
            "Chief Complaint / Symptom Narrative:",
            value=st.session_state["chief_complaint"],
            height=130,
            placeholder="Describe clinical symptoms (e.g. acute crushing chest pressure radiating to left arm with diaphoresis)..."
        )
    with col_sev:
        st.session_state["severity_score"] = st.slider("Reported Severity (1-10):", min_value=1, max_value=10, value=st.session_state["severity_score"])
        st.session_state["enable_vitals"] = st.checkbox("Include Vital Signs", value=st.session_state["enable_vitals"])

    # Optional Physiological Vital Signs
    if st.session_state["enable_vitals"]:
        with st.expander("➕ Physiological Vital Signs (Recorded at Triage)", expanded=True):
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.session_state["v_hr"] = st.number_input("Heart Rate (bpm):", min_value=30, max_value=250, value=st.session_state["v_hr"])
            with cv2:
                st.session_state["v_spo2"] = st.number_input("SpO2 (%):", min_value=50, max_value=100, value=st.session_state["v_spo2"])
            with cv3:
                st.session_state["v_sys"] = st.number_input("Systolic BP (mmHg):", min_value=50, max_value=260, value=st.session_state["v_sys"])

    st.markdown("---")
    
    # Multi-Modal Diagnostic Inputs: X-Ray & Labs
    col_xray_input, col_lab_input = st.columns(2)
    
    with col_xray_input:
        st.subheader("🩻 Chest Radiograph Input")
        xray_mode = st.radio("Radiograph Source:", ["Curated Demonstration Cohort", "Upload Medical Image (In-Memory)"], horizontal=True)
        if xray_mode == "Curated Demonstration Cohort":
            sample_dir = Path(__file__).resolve().parent / "data" / "sample_xrays"
            if sample_dir.exists():
                files = list(sample_dir.glob("*.png"))
                file_names = [f.name for f in files]
                chosen_sample = st.selectbox("Select Cohort Radiograph:", file_names, index=0 if file_names else 0)
                if chosen_sample:
                    with open(sample_dir / chosen_sample, "rb") as f:
                        st.session_state["xray_bytes"] = f.read()
                    st.session_state["xray_source_name"] = chosen_sample
            else:
                st.info("Sample radiographs directory is empty.")
        else:
            uploaded_file = st.file_uploader("Upload Radiograph (PNG/JPEG/TIFF):", type=["png", "jpg", "jpeg", "tiff"])
            if uploaded_file is not None:
                st.session_state["xray_bytes"] = uploaded_file.read()
                st.session_state["xray_source_name"] = uploaded_file.name
                st.success(f"Loaded '{uploaded_file.name}' into volatile memory.")

        if st.session_state.get("xray_bytes"):
            try:
                preview_img = Image.open(io.BytesIO(st.session_state["xray_bytes"]))
                st.image(preview_img, width=220, caption=f"Active Radiograph: {st.session_state.get('xray_source_name')}")
            except Exception:
                st.caption("Unable to preview image.")

    with col_lab_input:
        st.subheader("🧪 Laboratory Diagnostic Panels")
        st.caption("Enter measured values or leave blank if test was not ordered:")
        
        lab_panel = st.session_state.get("lab_values", {})
        col_l1, col_l2 = st.columns(2)
        test_keys = list(LAB_REFERENCE_RANGES.keys())
        half = len(test_keys) // 2
        
        updated_labs = {}
        with col_l1:
            for t_name in test_keys[:half]:
                unit = LAB_REFERENCE_RANGES[t_name]["unit"]
                cur_val = lab_panel.get(t_name, {}).get("value", "")
                val_str = st.text_input(f"{t_name} ({unit}):", value=str(cur_val) if cur_val != "" else "", key=f"inp_lab_{t_name}")
                try:
                    val_float = float(val_str.strip()) if val_str and val_str.strip() else None
                except ValueError:
                    val_float = None
                updated_labs[t_name] = {"value": val_float, "unit": unit}
                
        with col_l2:
            for t_name in test_keys[half:]:
                unit = LAB_REFERENCE_RANGES[t_name]["unit"]
                cur_val = lab_panel.get(t_name, {}).get("value", "")
                val_str = st.text_input(f"{t_name} ({unit}):", value=str(cur_val) if cur_val != "" else "", key=f"inp_lab_{t_name}")
                try:
                    val_float = float(val_str.strip()) if val_str and val_str.strip() else None
                except ValueError:
                    val_float = None
                updated_labs[t_name] = {"value": val_float, "unit": unit}
                
        st.session_state["lab_values"] = updated_labs

    st.markdown("---")
    
    # Navigation to Step 2
    if st.button("Proceed to AI Multi-Model Processing ⚡ ➡️", type="primary", use_container_width=True):
        st.session_state["wizard_step"] = 2
        st.rerun()


# ==============================================================================
# STEP 2: AI MULTI-MODEL PROCESSING & ASYNC PIPELINE (WIZARD STAGE 2)
# ==============================================================================
elif st.session_state["wizard_step"] == 2:
    st.markdown("### ⚡ Step 2: Multi-Model AI Inference & Guideline RAG")
    st.write("Executing neural network inference, rule engines, explainability calculations, and clinical guideline retrieval...")
    
    progress_bar = st.progress(0, text="Initializing cached neural network models...")
    
    with st.spinner("Processing multi-modal clinical pipelines..."):
        # 1. Load cached models via @st.cache_resource
        progress_bar.progress(15, text="1/5 Loading cached NLP & Vision models via @st.cache_resource...")
        triage_engine = get_triage_model()
        xray_engine = get_xray_model()
        lab_engine = get_lab_model()
        risk_engine = get_risk_engine()
        rag_engine = get_rag_engine()
        time.sleep(0.15)
        
        # 2. NLP Symptom Triage Execution
        progress_bar.progress(35, text="2/5 Executing Calibrated NLP Symptom Triage & Red-Flag Safety scan...")
        vitals_payload = None
        if st.session_state["enable_vitals"]:
            vitals_payload = {
                "HeartRate": st.session_state["v_hr"],
                "SpO2": st.session_state["v_spo2"],
                "SystolicBP": st.session_state["v_sys"]
            }
            
        triage_res = triage_engine.predict(
            chief_complaint=st.session_state["chief_complaint"],
            duration_days=st.session_state["duration_days"],
            severity_score=st.session_state["severity_score"],
            patient_age=st.session_state["patient_age"],
            vital_signs=vitals_payload,
            confidence_threshold=threshold_decimal
        )
        st.session_state["triage_result"] = triage_res
        time.sleep(0.15)
        
        # 3. Chest Radiograph Analysis & OOD Validation
        progress_bar.progress(60, text="3/5 Performing Out-of-Distribution validation & MobileNetV2 Chest X-Ray screening...")
        xray_res = None
        if st.session_state.get("xray_bytes"):
            xray_res = xray_engine.analyze(
                image_bytes=st.session_state["xray_bytes"],
                threshold=threshold_decimal
            )
        st.session_state["xray_result"] = xray_res
        time.sleep(0.15)
        
        # 4. Laboratory Result Interpretation
        progress_bar.progress(80, text="4/5 Comparing laboratory values against Tietz & Harrison Reference Intervals...")
        lab_res = lab_engine.interpret_panel(
            panel_dict=st.session_state.get("lab_values", {}),
            patient_sex=st.session_state["patient_sex"],
            patient_age=st.session_state["patient_age"]
        )
        st.session_state["lab_result"] = lab_res
        time.sleep(0.15)
        
        # 5. Guideline RAG Retrieval & Composite Risk Engine
        progress_bar.progress(95, text="5/5 Retrieving grounded clinical practice guidelines and computing composite risk...")
        detected_xray = xray_res.get("detected_abnormalities", []) if xray_res else []
        critical_labs = lab_res.get("critical_alerts", []) if lab_res else []
        
        rag_res = rag_engine.synthesize_guideline_evidence(
            presentation_text=st.session_state["chief_complaint"],
            urgency_category=triage_res["urgency_category"],
            detected_pathologies=detected_xray,
            critical_labs=critical_labs
        )
        st.session_state["rag_result"] = rag_res
        
        risk_res = risk_engine.calculate_risk(
            triage_output=triage_res,
            lab_output=lab_res,
            xray_output=xray_res,
            vital_signs=vitals_payload,
            patient_age=st.session_state["patient_age"]
        )
        st.session_state["risk_result"] = risk_res
        
        progress_bar.progress(100, text="Multi-modal AI inference complete! All findings synthesized.")
        time.sleep(0.2)
        
    st.success("✅ Multi-modal clinical synthesis completed successfully.")
    
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Back to Data Entry", use_container_width=True):
            st.session_state["wizard_step"] = 1
            st.rerun()
    with col_next:
        if st.button("View Diagnostic Results & Explainable AI 📊 ➡️", type="primary", use_container_width=True):
            st.session_state["wizard_step"] = 3
            st.rerun()


# ==============================================================================
# STEP 3: RESULTS, EXPLAINABLE AI & REVIEW (WIZARD STAGE 3)
# ==============================================================================
elif st.session_state["wizard_step"] == 3:
    st.markdown("### 📊 Step 3: Diagnostic Results & Explainable AI (XAI) Review")
    st.caption(f"Filtering anomalies using active Confidence Threshold: **{conf_threshold}%** (Adjustable in sidebar)")
    
    triage_res = st.session_state.get("triage_result")
    xray_res = st.session_state.get("xray_result")
    lab_res = st.session_state.get("lab_result")
    risk_res = st.session_state.get("risk_result")
    rag_res = st.session_state.get("rag_result")
    
    if not triage_res:
        st.warning("No processed results in session. Please run Step 2 first.")
        if st.button("Run AI Processing Now"):
            st.session_state["wizard_step"] = 2
            st.rerun()
    else:
        # 1. Executive Urgency Alert Callout
        urgency = triage_res["urgency_category"]
        if urgency == "Emergency Warning":
            st.error(f"### 🚨 Urgency Category: {urgency.upper()}")
            st.error(f"**CRITICAL ACTION ADVISORY:** {triage_res['action_advisory']}")
            if triage_res.get("emergency_contact"):
                st.error(f"📞 Immediate National Emergency Hotline: Dial **{triage_res['emergency_contact']}**")
        elif urgency == "Urgent Assessment":
            st.warning(f"### ⚠️ Urgency Category: {urgency}")
            st.write(f"**Action Advisory:** {triage_res['action_advisory']}")
        elif urgency == "Routine Assessment":
            st.success(f"### 🟢 Urgency Category: {urgency}")
            st.write(f"**Action Advisory:** {triage_res['action_advisory']}")
        elif urgency in ["Requires More Information", "Insufficient Information"]:
            st.info(f"### ❓ Urgency Category: {urgency.upper()}")
            st.warning(f"⚠️ **Clinical Advisory (Low Model Confidence):** {triage_res['action_advisory']}")
            st.caption("💡 The calibrated model confidence is below the diagnostic threshold. Rather than forcing an inaccurate classification, MediAssist prompts for additional clinical inquiry, vital signs, or symptom progression details.")
        else:
            st.info(f"### ℹ️ Urgency Category: {urgency}")
            
        if triage_res.get("safety_override"):
            st.error(f"🛡️ **Deterministic Safety Override Engaged:** Life-threat detected: `{', '.join(triage_res.get('red_flags', []))}`. Statistical probability override enforced to protect patient life.")

        # 2. Mandatory 10-Second Clinician Explanation Card
        st.markdown("#### ⏱️ Clinician Explanation (10-Second Summary)")
        exp = triage_res["explanation"]
        st.markdown(f"""
        > **Finding:** {exp['finding']}  
        > **Reason:** {exp['reason']}  
        > **Confidence:** {exp['confidence']}  
        > **Limitation:** {exp['limitation']}
        """)

        # 3. Composite Patient Risk Prioritization Metric Card
        if risk_res:
            c_score, c_tier, c_drivers = st.columns([1, 1, 2])
            with c_score:
                st.metric("Composite Risk Score", f"{risk_res['composite_risk_score']} / 100", delta="Emergency Override" if risk_res['safety_override_active'] else None, delta_color="inverse")
            with c_tier:
                st.metric("Priority Tier", risk_res['priority_tier'])
            with c_drivers:
                factors_str = "; ".join(risk_res.get("contributing_factors", [])) or "Non-acute baseline parameters."
                st.markdown(f"**Primary Risk Drivers:**\n*{factors_str}*")

        st.markdown("---")

        # 4. EXPLAINABLE AI (XAI) SECTION (LIME & Grad-CAM)
        st.subheader("🔍 Explainable AI (XAI) Visualizations")
        
        tab_vision_xai, tab_text_xai = st.tabs([
            "🩻 Chest Radiograph & Grad-CAM Anatomical Saliency",
            "📝 Symptom Narrative & LIME Word Attributions"
        ])
        
        with tab_vision_xai:
            if xray_res and xray_res.get("success"):
                gc = xray_res.get("gradcam", {})
                render_xai_comparison_columns(
                    orig_img=gc.get("original"),
                    blended_img=gc.get("blend"),
                    probabilities=xray_res.get("probabilities", {}),
                    threshold=threshold_decimal
                )
                
                # Flagged anomalies above threshold
                detected = [k for k, v in xray_res.get("probabilities", {}).items() if v >= threshold_decimal and k != "Normal"]
                if detected:
                    st.warning(f"⚠️ **Pathologies Exceeding {conf_threshold}% Threshold:** {', '.join(detected)}")
                else:
                    st.info(f"ℹ️ No thoracic pathologies exceeded the active {conf_threshold}% threshold.")
                    
                st.caption(f"**Vision Model Explanation:** {xray_res['explanation']['reason']}")
            elif xray_res and xray_res.get("ood_rejected"):
                st.error(f"🚫 **Out-of-Distribution Validation Rejection:** {xray_res.get('error')}")
            else:
                st.info("No chest radiograph submitted in Step 1.")

        with tab_text_xai:
            triage_model_obj = get_triage_model()
            if triage_model_obj.model is not None and triage_model_obj.vectorizer is not None and st.session_state["chief_complaint"]:
                with st.spinner("Computing LIME local token attributions..."):
                    lime_res = generate_lime_text_explanation(
                        st.session_state["chief_complaint"],
                        triage_model_obj.model,
                        triage_model_obj.vectorizer
                    )
                if lime_res.get("plot") is not None:
                    st.plotly_chart(lime_res["plot"], use_container_width=True)
                else:
                    st.caption("Insufficient vocabulary weights to render LIME plot.")
            else:
                st.caption("NLP vectorizer model waiting for input narrative.")

        st.markdown("---")

        # 5. LABORATORY & CRITICAL PANIC VALUES
        st.subheader("🧪 Laboratory Result Interpretation")
        if lab_res and lab_res.get("tests"):
            if lab_res.get("has_critical"):
                st.error(f"🚨 **CRITICAL VALUE PANIC ALERT:** {len(lab_res['critical_alerts'])} test(s) require STAT physician notification.")
            
            lab_rows = []
            for t in lab_res["tests"]:
                lab_rows.append({
                    "Test": t["test_name"],
                    "Measured Value": f"{t['value']} {t['unit']}",
                    "Status": t["status"],
                    "Reference Range": f"{t['reference_interval'][0]} - {t['reference_interval'][1]} {t['unit']}",
                    "Clinical Implication": t["clinical_implication"],
                    "Action": t["recommendation"]
                })
            df_labs = pd.DataFrame(lab_rows)
            st.dataframe(df_labs, use_container_width=True, hide_index=True)
        else:
            st.info("No laboratory panel entered for this consultation.")

        st.markdown("---")

        # 6. RAG EVIDENCE (RETRIEVAL-AUGMENTED GENERATION GUIDELINE GROUNDING)
        st.subheader("📚 Grounded Clinical Guidelines (RAG Evidence)")
        if rag_res and rag_res.get("has_evidence"):
            st.markdown(f"**Synthesized Clinical Insight:** {rag_res['synthesized_summary']}")
            for gl in rag_res.get("guidelines", []):
                with st.expander(f"📖 {gl.get('topic')} — {gl.get('source')} ({gl.get('year')}) [Relevance: {gl.get('relevance_score')}]"):
                    st.write(f"**Guideline Summary:** {gl.get('summary')}")
                    st.info(f"**Standard of Care Recommendation:** {gl.get('recommendation')}")
        else:
            st.caption("No specific indexed medical guideline matched the query parameters.")

        st.markdown("---")
        
        # Navigation
        col_b, col_e = st.columns(2)
        with col_b:
            if st.button("⬅️ Modify Data Entry", use_container_width=True):
                st.session_state["wizard_step"] = 1
                st.rerun()
        with col_e:
            if st.button("Proceed to Export PDF Report 📄 ➡️", type="primary", use_container_width=True):
                st.session_state["wizard_step"] = 4
                st.rerun()


# ==============================================================================
# STEP 4: EXPORT CONSULTATION REPORT (WIZARD STAGE 4)
# ==============================================================================
elif st.session_state["wizard_step"] == 4:
    st.markdown("### 📄 Step 4: Export Formatted Clinical Consultation PDF")
    st.write("Compile this decision-support session into a clean, downloadable clinical report ready for physician review.")
    
    triage_res = st.session_state.get("triage_result")
    xray_res = st.session_state.get("xray_result")
    lab_res = st.session_state.get("lab_result")
    risk_res = st.session_state.get("risk_result")
    rag_res = st.session_state.get("rag_result")
    
    if not triage_res:
        st.warning("No diagnostic session data available. Please complete Steps 1 and 2 first.")
        if st.button("Start New Consultation"):
            st.session_state["wizard_step"] = 1
            st.rerun()
    else:
        # Package session metadata
        session_meta = {
            "patient_name": st.session_state.get("patient_name"),
            "age": st.session_state.get("patient_age"),
            "sex": st.session_state.get("patient_sex"),
            "duration_days": st.session_state.get("duration_days"),
            "severity_score": st.session_state.get("severity_score"),
            "chief_complaint": st.session_state.get("chief_complaint")
        }
        
        # Generate PDF in volatile memory
        with st.spinner("Compiling structured ReportLab PDF consultation report..."):
            pdf_bytes = generate_clinical_summary_pdf(
                session_data=session_meta,
                triage_result=triage_res,
                xray_result=xray_res,
                lab_result=lab_res,
                risk_result=risk_res,
                rag_guidelines=rag_res.get("guidelines") if rag_res else None
            )
            
        st.success("✅ Clinical Consultation PDF generated successfully in volatile memory!")
        
        col_down, col_flush = st.columns(2)
        with col_down:
            st.download_button(
                label="📥 Download Clinical Consultation Summary (PDF)",
                data=pdf_bytes,
                file_name=f"MediAssist_Consultation_{st.session_state.get('patient_name', 'Patient').replace(' ', '_')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        with col_flush:
            if st.button("🗑️ Finish Consultation & Flush Memory", use_container_width=True):
                flush_session_data()
                st.success("Memory flushed! Zero Data Retention enforced.")
                st.rerun()

        st.markdown("---")
        if st.button("⬅️ Return to Review & XAI Visuals"):
            st.session_state["wizard_step"] = 3
            st.rerun()
