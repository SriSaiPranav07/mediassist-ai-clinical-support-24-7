"""
MediAssist — Page 8: Detailed Model Cards & Governance
Standardized Transparency Cards for Clinical AI Systems
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="MediAssist — Model Cards", page_icon="📜", layout="wide")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📜 Model Cards & Transparency Documentation")
st.caption("Module G: Standardized Healthcare Model Documentation following Mitchell et al. Guidelines")

mc_tabs = st.tabs([
    "🏥 System Architecture Card",
    "📝 NLP Triage Model Card",
    "🩻 Chest X-Ray Model Card",
    "🧪 Lab Interpreter Reference Card"
])

# 1. SYSTEM MODEL CARD
with mc_tabs[0]:
    st.markdown("""
    ## MediAssist System-Level Model Card (v1.0.0)

    ### 1. System Summary
    **MediAssist** is a multi-model clinical decision support demonstration platform developed for the Zelvora Industry Labs AI Capstone. It integrates natural language symptom triage, deep learning chest radiograph screening, expert laboratory interpretation, and transparent patient risk prioritization.

    ### 2. Intended Use
    - **Primary Purpose:** Educational training, clinical research workflow demonstration, and simulated triage queue prioritization.
    - **Intended Users:** Medical students, clinical researchers, and healthcare AI engineers.
    - **Deployment Setting:** Research workstations and academic clinical simulation environments.

    ### 3. Explicitly Prohibited Uses
    - ❌ **Autonomous Clinical Diagnosis:** MediAssist must NOT be used to replace a physician or formulate standalone medical diagnoses.
    - ❌ **Medication Prescription:** MediAssist does NOT prescribe drugs or clinical treatments.
    - ❌ **Real-World Triage Without Human Oversight:** Under no circumstances should patients be directed or delayed without licensed human clinical assessment.

    ### 4. Safety & Ethical Safeguards
    - **Emergency Priority Guarantee:** Deterministic clinical red flags (crushing chest pain, FAST stroke symptoms, severe hypoxemia) trigger immediate referral to emergency services (112 in India) and cannot be overridden by statistical models.
    - **Under-10-Second Explanations:** Every output is structured into Finding, Reason, Confidence, and Limitation.
    - **Privacy:** No real identifiable patient health information (PHI) is ingested, retained, or logged.
    """)

# 2. NLP TRIAGE MODEL CARD
with mc_tabs[1]:
    st.markdown("""
    ## NLP Symptom Triage Classifier Model Card

    ### Model Details
    - **Architecture:** Calibrated Classifier (Platt Sigmoid Scaling over Balanced Logistic Regression) with TF-IDF unigram/bigram feature extraction.
    - **Framework:** Scikit-learn 1.5.2
    - **Input:** Patient chief complaint text, severity scale (1-10), symptom duration (days), age.
    - **Output:** Urgency category (`Emergency Warning`, `Urgent Assessment`, `Routine Assessment`, `Insufficient Information`) and class probabilities.

    ### Training Data & Provenance
    - **Dataset:** Curated clinical presentation corpus across Emergency, Urgent, and Routine categories based on Emergency Severity Index (ESI) and Manchester Triage criteria.
    - **Split:** 75% stratified training, 25% held-out test split.

    ### Performance Metrics (Held-Out Test Set)
    - **Accuracy:** 66.7%
    - **Macro Sensitivity (Recall):** 66.7%
    - **Macro Specificity:** 83.3%
    - **Macro ROC-AUC:** 0.773

    ### Limitations & Demographic Biases
    - Training data is limited to English clinical text; may exhibit reduced accuracy on idiomatic expressions, regional dialects, or non-English narratives.
    - Requires clinician confirmation.
    """)

# 3. CHEST X-RAY MODEL CARD
with mc_tabs[2]:
    st.markdown("""
    ## MobileNetV2 Chest X-Ray Screener Model Card

    ### Model Details
    - **Backbone:** MobileNetV2 (ImageNet pretrained weights)
    - **Classification Head:** Global Average Pooling, Batch Normalization, Dropout (0.3), Dense (128, ReLU), Dropout (0.2), Dense (6, Sigmoid).
    - **Target Pathologies:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Pneumothorax, Normal.
    - **Framework:** TensorFlow 2.16+ / Keras 3.x
    - **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the `out_relu` convolutional feature map.

    ### Performance Metrics (Held-Out Test Set)
    - **Cardiomegaly ROC-AUC:** 1.000 (Sensitivity: 100.0%, Specificity: 100.0%)
    - **Effusion ROC-AUC:** 0.944 (Sensitivity: 66.7%, Specificity: 91.7%)
    - **Infiltration ROC-AUC:** 1.000 (Sensitivity: 100.0%, Specificity: 46.2%)
    - **Atelectasis ROC-AUC:** 0.944 (Sensitivity: 100.0%, Specificity: 66.7%)
    - **Pneumothorax ROC-AUC:** 0.731 (Sensitivity: 100.0%, Specificity: 69.2%)
    - **Normal Screening ROC-AUC:** 0.909 (Sensitivity: 100.0%, Specificity: 81.8%)

    ### Limitations
    - Screenings are image-level probabilistic assessments, not diagnostic readings.
    - Grad-CAM highlights CNN attention regions; it does not indicate histopathologic tissue borders.
    """)

# 4. LAB INTERPRETER CARD
with mc_tabs[3]:
    st.markdown("""
    ## Clinical Laboratory Interpreter Reference Card

    ### Source Authorities
    - **Tietz Clinical Guide to Laboratory Tests, 4th Edition** (Burtis CA, Ashwood ER, Bruns DE)
    - **Harrison's Principles of Internal Medicine, 21st Edition**
    - **College of American Pathologists (CAP)** Critical Value Guidelines

    ### Evaluated Panels
    - Hemoglobin (Sex-stratified reference intervals)
    - White Blood Cell (WBC) Count & Platelets
    - Serum Glucose (Fasting vs Random protocols)
    - Serum Creatinine, Sodium, and Potassium
    - Liver Enzymes (ALT, AST)
    - High-Sensitivity Cardiac Troponin I

    ### Critical Alerts
    - Values crossing critical thresholds (e.g. Potassium < 2.8 or > 6.2 mEq/L, Platelets < 20 x 10^3/uL) trigger immediate stat physician alerts and cannot be muted.
    """)
