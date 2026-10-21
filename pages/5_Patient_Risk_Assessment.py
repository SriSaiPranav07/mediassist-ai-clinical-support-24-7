"""
MediAssist — Page 5: Composite Patient Risk Assessment
Transparent Multi-Criteria Prioritization Engine with Absolute Safety Guarantees
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import (
    get_triage_engine,
    get_lab_interpreter,
    get_xray_screener,
    get_risk_engine,
    get_explainability_engine
)
from src.utils import DEMO_PATIENTS
from src.config import SAMPLE_XRAYS_DIR

st.set_page_config(page_title="MediAssist — Patient Risk Assessment", page_icon="⚖️", layout="wide")

st.title("⚖️ Composite Patient Risk Scoring & Prioritization")
st.caption("Module D: Multi-Modal Prioritization Engine with Deterministic Emergency Overrides")

triage_engine = get_triage_engine()
lab_interpreter = get_lab_interpreter()
xray_screener = get_xray_screener()
risk_engine = get_risk_engine()
exp_engine = get_explainability_engine()

# Patient Selector
st.markdown("#### 👤 Select Demonstration Patient Profile")
patient_options = [f"{p['id']} — {p['name']} ({p['clinical_scenario']})" for p in DEMO_PATIENTS]
selected_pt_label = st.selectbox("Clinical Demonstration Record:", patient_options)

pt_id = selected_pt_label.split(" — ")[0]
patient = next(p for p in DEMO_PATIENTS if p["id"] == pt_id)

# Execute Multi-Modal Pipeline
with st.spinner(f"Synthesizing multi-modal diagnostics for {patient['name']}..."):
    # 1. Symptom Triage
    triage_res = triage_engine.classify_symptoms(
        chief_complaint=patient["chief_complaint"],
        duration_days=patient["duration_days"],
        severity_score=patient["severity"],
        patient_age=patient["age"],
        vital_signs=patient.get("vitals")
    )

    # 2. Lab Interpretation
    lab_res = lab_interpreter.interpret_panel(
        panel_dict=patient.get("labs", {}),
        patient_sex=patient["sex"],
        patient_age=patient["age"]
    )

    # 3. Chest X-Ray
    xray_filename = patient.get("xray_sample", "normal_sample.png")
    xray_path = SAMPLE_XRAYS_DIR / xray_filename
    xray_res = None
    if xray_path.exists():
        with open(xray_path, "rb") as f:
            xray_res = xray_screener.analyze_radiograph(f.read())

    # 4. Composite Risk Aggregation
    risk_res = risk_engine.calculate_patient_risk(
        patient_id=patient["id"],
        triage_output=triage_res,
        lab_output=lab_res,
        xray_output=xray_res,
        vital_signs=patient.get("vitals")
    )

st.markdown("---")

# Priority Banner
priority = risk_res["priority_level"]
score = risk_res["composite_score"]

if priority == "Emergency Review":
    st.error(f"## 🚨 PRIORITY LEVEL: {priority.upper()} (Risk Score: {score}/100)")
    st.error(f"**Action Required:** {risk_res['recommended_action']}")
elif priority == "High Priority":
    st.warning(f"## ⚠️ PRIORITY LEVEL: {priority} (Risk Score: {score}/100)")
    st.write(f"**Action Required:** {risk_res['recommended_action']}")
elif priority == "Standard Review":
    st.success(f"## 🟢 PRIORITY LEVEL: {priority} (Risk Score: {score}/100)")
    st.write(f"**Action Required:** {risk_res['recommended_action']}")
else:
    st.info(f"## ℹ️ PRIORITY LEVEL: {priority}")

# Mandatory 10-Second Clinician Explanation Card
st.subheader("⏱️ Clinician Multi-Modal Prioritization Summary (10-Second Summary)")
exp = risk_res["explanation"]
st.markdown(f"""
> **Finding:** {exp['finding']}  
> **Reason:** {exp['reason']}  
> **Confidence:** {exp['confidence']}  
> **Limitation:** {exp['limitation']}
""")

st.markdown("---")

# Multi-Modal Sub-Scores & Waterfall Attribution
col_gauge, col_waterfall = st.columns([1, 1])

with col_gauge:
    st.subheader("📊 Sub-Score Point Attribution")
    sub = risk_res["sub_scores"]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Composite Risk Index (0-100)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1e3a8a"},
            'steps': [
                {'range': [0, 44], 'color': "#dcfce7"},
                {'range': [45, 84], 'color': "#fef3c7"},
                {'range': [85, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_waterfall:
    st.subheader("💧 Risk Contribution by Clinical Dimension")
    sub_df = [
        {"Dimension": "Symptom Triage (Max 30)", "Score": sub.get("symptoms", 0)},
        {"Dimension": "Vital Signs (Max 25)", "Score": sub.get("vitals", 0)},
        {"Dimension": "Laboratory (Max 25)", "Score": sub.get("labs", 0)},
        {"Dimension": "Radiology (Max 20)", "Score": sub.get("radiology", 0)},
    ]
    fig_bar = px.bar(
        sub_df,
        x="Score",
        y="Dimension",
        orientation="h",
        text="Score",
        color="Dimension",
        color_discrete_sequence=["#0284c7", "#f59e0b", "#8b5cf6", "#ec4899"]
    )
    fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# Full Audit Trail of Contributing Clinical Findings
st.markdown("---")
st.subheader("🔍 Complete Clinical Evidence Audit Trail")
col_trail1, col_trail2 = st.columns(2)

with col_trail1:
    st.markdown("**Contributing Clinical Findings Identified:**")
    for factor in risk_res["contributing_factors"]:
        st.markdown(f"- {factor}")

with col_trail2:
    if risk_res["emergency_reasons"]:
        st.markdown("**🚨 Deterministic Life-Threat Triggers:**")
        for reason in risk_res["emergency_reasons"]:
            st.markdown(f"- 🔴 **{reason}**")
    elif risk_res["high_priority_reasons"]:
        st.markdown("**⚠️ Elevated Clinical Factors:**")
        for reason in risk_res["high_priority_reasons"]:
            st.markdown(f"- 🟡 {reason}")
    else:
        st.markdown("**🟢 Standard Priority Factors:**")
        st.markdown("- Patient vitals and parameters remain within expected non-emergent bounds.")
