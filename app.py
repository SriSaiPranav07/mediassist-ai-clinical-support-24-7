"""
MediAssist — AI Clinical Decision Support System
Main Application Entry Point
Zelvora Technologies / Zelvora Industry Labs AI Capstone
"""
import streamlit as st
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="MediAssist — AI Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished clinical theme
st.markdown("""
<style>
    /* Hide unconfigured app entry from Streamlit sidebar navigation */
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
    /* Professional Clinical Styling */
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .clinical-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
    }
    .explanation-card {
        background-color: #ffffff;
        border-left: 4px solid #0284c7;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    .emergency-badge {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/caduceus.png", width=64)
        st.title("MediAssist")
        st.caption("AI Clinical Decision Support System\nZelvora Industry Labs Capstone")
        st.markdown("---")
        
        st.markdown("### 🚨 Emergency Safety Layer")
        st.info("**Emergency Hotlines:**\n- **India:** Dial **112**\n- **US/Global:** Dial **911 / 112**\n\n*Emergency warning rules automatically override statistical scores.*")
        
        st.markdown("---")
        st.markdown("### 🤖 Active AI Models")
        st.markdown("- **NLP Triage:** Calibrated TF-IDF Classifier\n- **Chest X-Ray:** Fine-Tuned MobileNetV2 CNN\n- **Lab Engine:** Tietz & Harrison Reference Rules\n- **Risk Engine:** Multi-Modal Safety Matrix")
        
        st.markdown("---")
        st.caption("© 2026 Zelvora Technologies. Educational & Research Demonstration Only.")

render_sidebar()

# Main Landing Page Content
st.markdown("""
<div class="clinical-banner">
    <h1 style="margin:0; font-size: 2.2rem;">🩺 MediAssist</h1>
    <p style="margin: 0.4rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
        Multi-Model AI Clinical Decision Support System with Plain-English Clinician Explanations
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Educational & Research Demonstration Disclaimer:</strong><br>
    MediAssist is developed strictly for educational, capstone, and research demonstration. It is <strong>NOT a certified medical device</strong> (FDA, CE, or CDSCO) and does <strong>not provide definitive clinical diagnoses or prescribe treatments</strong>. Every output must be reviewed and verified by a licensed healthcare professional.
</div>
""", unsafe_allow_html=True)

st.markdown("### 🎯 Quick Module Launcher")
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
with col_nav1:
    if st.button("📊 Clinical Overview", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Clinical_Overview.py")
with col_nav2:
    if st.button("🩺 Symptom Triage", use_container_width=True):
        st.switch_page("pages/2_Symptom_Triage.py")
with col_nav3:
    if st.button("🩻 Chest X-Ray", use_container_width=True):
        st.switch_page("pages/3_Chest_Xray_Analysis.py")
with col_nav4:
    if st.button("🧪 Lab Interpreter", use_container_width=True):
        st.switch_page("pages/4_Lab_Interpreter.py")

st.markdown("### 🎯 System Navigation & Capabilities")
st.markdown("""
Welcome to **MediAssist**. Select a clinical module from the sidebar navigation:

1. **[1_Clinical_Overview](/Clinical_Overview)** — Executive triage summary, key performance indicators, active models, and demonstration patient cohorts.
2. **[2_Symptom_Triage](/Symptom_Triage)** — Natural Language Processing for symptom urgency screening with deterministic life-threat red-flag safety overrides.
3. **[3_Chest_Xray_Analysis](/Chest_Xray_Analysis)** — MobileNetV2 Deep Learning transfer screening for 6 thoracic pathologies with **Grad-CAM** visual saliency.
4. **[4_Lab_Interpreter](/Lab_Interpreter)** — Expert clinical laboratory engine flagging abnormal, borderline, and potentially critical laboratory values with Tietz/Harrison reference intervals.
5. **[5_Patient_Risk_Assessment](/Patient_Risk_Assessment)** — Transparent multi-criteria patient prioritization synthesizing symptoms, vital signs, labs, and imaging without risk-diluting averages.
6. **[6_Explainability_Center](/Explainability_Center)** — LIME text explainability, Grad-CAM visual heatmaps, SHAP-style waterfall attributions, and **10-Second Clinician Explanations**.
7. **[7_Model_Evaluation](/Model_Evaluation)** — Comprehensive held-out evaluation reports, ROC-AUC, sensitivity/specificity curves, and error analysis.
8. **[8_Model_Cards](/Model_Cards)** — Comprehensive model documentation, intended usage, ethical guidelines, and bias assessments.
""")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    #### 🛡️ Safety-First Architecture
    - Red-flag detection for stroke, STEMI, and acute respiratory failure.
    - Deterministic override guarantees emergency priority.
    """)
with col2:
    st.markdown("""
    #### ⏱️ Under-10-Second Clinician Explanations
    - Structured **Finding**, **Reason**, **Confidence**, and **Limitation** for every single AI output.
    """)
with col3:
    st.markdown("""
    #### 🔬 Reproducible AI & Evaluation
    - Real trained weights for MobileNetV2 CNN and calibrated NLP classifier.
    - Zero fake metrics or hardcoded hallucinations.
    """)
