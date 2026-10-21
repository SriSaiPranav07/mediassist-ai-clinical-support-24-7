"""
MediAssist — Page 2: Symptom Triage Module
Natural Language Processing with Deterministic Safety Overrides & 10-Second Explanations
"""
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_triage_engine, get_explainability_engine
from src.utils import DEMO_PATIENTS
from src.config import EMERGENCY_CONTACT_INDIA

st.set_page_config(page_title="MediAssist — Symptom Triage", page_icon="🩺", layout="wide")

st.title("🩺 NLP Symptom Triage & Safety Screening")
st.caption("Module A: Calibrated Symptom Urgency Classification with Red-Flag Safety Overrides")

triage_engine = get_triage_engine()
exp_engine = get_explainability_engine()

# Demonstration Case Pre-filler
st.markdown("#### ⚡ Quick Demonstration Case Selector")
demo_options = ["Custom Input"] + [f"{p['id']} — {p['clinical_scenario']}" for p in DEMO_PATIENTS]
selected_case = st.selectbox("Load sample patient presentation:", demo_options)

prefill_text = ""
prefill_duration = 2.0
prefill_severity = 5
prefill_age = 45

if selected_case != "Custom Input":
    case_id = selected_case.split(" — ")[0]
    matched_pt = next((p for p in DEMO_PATIENTS if p["id"] == case_id), None)
    if matched_pt:
        prefill_text = matched_pt["chief_complaint"]
        prefill_duration = float(matched_pt["duration_days"])
        prefill_severity = int(matched_pt["severity"])
        prefill_age = int(matched_pt["age"])

# Input Form
with st.form("triage_form"):
    col_input, col_meta = st.columns([3, 1])

    with col_input:
        chief_complaint = st.text_area(
            "Patient Chief Complaint / Symptom Narrative:",
            value=prefill_text,
            height=130,
            placeholder="e.g., Patient describes severe chest pressure radiating to left arm with sweating for 1 hour..."
        )

    with col_meta:
        duration_days = st.number_input("Duration (Days):", min_value=0.05, max_value=365.0, value=prefill_duration, step=0.5)
        severity_score = st.slider("Severity Scale (1-10):", min_value=1, max_value=10, value=prefill_severity)
        patient_age = st.number_input("Patient Age:", min_value=0, max_value=120, value=prefill_age)

    # Optional Vitals
    with st.expander("➕ Optional Vital Signs (for physiological red-flag check)", expanded=False):
        c_hr, c_spo2, c_bp = st.columns(3)
        with c_hr:
            v_hr = st.number_input("Heart Rate (bpm):", min_value=30, max_value=250, value=80)
        with c_spo2:
            v_spo2 = st.number_input("SpO2 (%):", min_value=50, max_value=100, value=98)
        with c_bp:
            v_sys = st.number_input("Systolic BP (mmHg):", min_value=50, max_value=260, value=120)

    submit_triage = st.form_submit_button("Run Triage Screening 🔍", type="primary")

if submit_triage or prefill_text != "":
    vitals_dict = {"HeartRate": v_hr, "SpO2": v_spo2, "SystolicBP": v_sys}
    result = triage_engine.classify_symptoms(
        chief_complaint=chief_complaint,
        duration_days=duration_days,
        severity_score=severity_score,
        patient_age=patient_age,
        vital_signs=vitals_dict
    )

    st.markdown("---")

    # Display Priority Results
    urgency = result["urgency_category"]
    badge_color = {
        "Emergency Warning": "error",
        "Urgent Assessment": "warning",
        "Routine Assessment": "success",
        "Insufficient Information": "info"
    }.get(urgency, "info")

    if urgency == "Emergency Warning":
        st.error(f"### 🚨 Urgency Category: {urgency.upper()}")
        st.error(f"**CRITICAL ADVISORY:** {result['action_advisory']}")
    elif urgency == "Urgent Assessment":
        st.warning(f"### ⚠️ Urgency Category: {urgency}")
        st.write(f"**Advisory:** {result['action_advisory']}")
    elif urgency == "Routine Assessment":
        st.success(f"### 🟢 Urgency Category: {urgency}")
        st.write(f"**Advisory:** {result['action_advisory']}")
    else:
        st.info(f"### ℹ️ Urgency Category: {urgency}")

    # Mandatory 10-Second Clinician Explanation
    st.markdown("### ⏱️ Clinician Explanation (10-Second Summary)")
    exp = result["explanation"]
    st.markdown(f"""
    > **Finding:** {exp['finding']}  
    > **Reason:** {exp['reason']}  
    > **Confidence:** {exp['confidence']}  
    > **Limitation:** {exp['limitation']}
    """)

    # Supporting Diagnostics & Explainability
    c_probs, c_lime = st.columns([1, 1])

    with c_probs:
        st.subheader("📊 Class Probability Distribution")
        probs = result.get("probabilities", {})
        if probs:
            prob_df = [{"Category": k, "Probability": v} for k, v in probs.items()]
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Category",
                orientation="h",
                text_auto=".2%",
                range_x=[0, 1],
                color="Category",
                color_discrete_map={
                    "Emergency Warning": "#ef4444",
                    "Urgent Assessment": "#f59e0b",
                    "Routine Assessment": "#10b981"
                }
            )
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with c_lime:
        st.subheader("🔍 Local Keyword Importance (LIME)")
        if triage_engine.model is not None and triage_engine.vectorizer is not None and chief_complaint:
            lime_res = exp_engine.explain_text_with_lime(chief_complaint, triage_engine.model, triage_engine.vectorizer)
            if lime_res and lime_res.get("weights"):
                lime_df = [{"Token": w[0], "Weight": round(w[1], 4)} for w in lime_res["weights"]]
                fig_lime = px.bar(
                    lime_df,
                    x="Weight",
                    y="Token",
                    orientation="h",
                    color="Weight",
                    color_continuous_scale="RdBu_r"
                )
                fig_lime.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_lime, use_container_width=True)
            else:
                st.caption("No significant vocabulary weights detected in current input.")
        else:
            st.caption("Model explanation engine waiting for input.")
