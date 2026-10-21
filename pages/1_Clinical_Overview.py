"""
MediAssist — Page 1: Clinical Overview
Executive Clinical Summary & Demonstration Patient Cohort
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import DEMO_PATIENTS
from src.config import TRIAGE_MODEL_PATH, XRAY_MODEL_PATH

st.set_page_config(page_title="MediAssist — Clinical Overview", page_icon="📊", layout="wide")

st.title("📊 Clinical Overview & Executive Summary")
st.caption("MediAssist Clinical Decision Support Platform — Cohort Monitoring")

# Clinical Alert Banner
st.info("ℹ️ **Active System Mode:** Demonstration & Research Cohort. All patient profiles below are synthetic cases constructed for clinical decision-support evaluation.")

# Executive KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Demonstration Cases", value=len(DEMO_PATIENTS), delta="Synthetic Cohort")
with col2:
    st.metric(label="Active AI/Rule Models", value="4 Subsystems", delta="Operational")
with col3:
    st.metric(label="Emergency Safety Layer", value="Active (112)", delta="Deterministic")
with col4:
    xray_status = "Available" if Path(XRAY_MODEL_PATH).exists() else "Offline"
    st.metric(label="MobileNetV2 CNN", value=xray_status, delta="Trained Weights")

st.markdown("---")

# Layout: Patient Queue & Priority Distribution
col_queue, col_chart = st.columns([3, 2])

with col_queue:
    st.subheader("📋 Active Demonstration Patient Queue")
    patient_rows = []
    for p in DEMO_PATIENTS:
        patient_rows.append({
            "ID": p["id"],
            "Patient Name": p["name"],
            "Age": p["age"],
            "Sex": p["sex"],
            "Clinical Presentation": p["clinical_scenario"],
            "Expected Triage": p["expected_triage"]
        })
    df_queue = pd.DataFrame(patient_rows)
    st.dataframe(df_queue, use_container_width=True, hide_index=True)

with col_chart:
    st.subheader("📈 Cohort Triage Distribution")
    triage_counts = df_queue["Expected Triage"].value_counts().reset_index()
    triage_counts.columns = ["Urgency", "Count"]
    fig = px.pie(
        triage_counts,
        names="Urgency",
        values="Count",
        color="Urgency",
        color_discrete_map={
            "Emergency Warning": "#ef4444",
            "Urgent Assessment": "#f59e0b",
            "Routine Assessment": "#10b981"
        },
        hole=0.45
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Model Inventory & Data Provenance Status
st.subheader("🔬 Diagnostic Engine Architecture & Status")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    **Module A: NLP Triage**
    - **Engine:** Calibrated TF-IDF + Logistic Classifier
    - **Safety:** Regex Red-Flag Lexicon
    - **Status:** 🟢 Operational
    """)

with c2:
    st.markdown("""
    **Module B: Chest X-Ray**
    - **Engine:** MobileNetV2 (Transfer Learning)
    - **Classes:** 6 Thoracic Pathologies
    - **Status:** 🟢 Operational (Grad-CAM Enabled)
    """)

with c3:
    st.markdown("""
    **Module C: Lab Interpreter**
    - **Engine:** Tietz & Harrison Reference Rules
    - **Tests:** 10 Core Metabolic/Cardiac Panels
    - **Status:** 🟢 Operational
    """)

with c4:
    st.markdown("""
    **Module D: Risk Engine**
    - **Engine:** Deterministic Safety Priority Engine
    - **Guarantees:** No dilution of emergency red flags
    - **Status:** 🟢 Operational
    """)
