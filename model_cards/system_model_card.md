# System Model Card: MediAssist — AI Clinical Decision Support System

## 1. System Details
- **System Name:** MediAssist
- **Version:** 1.0.0
- **Organization:** MediAssist Clinical AI Team
- **Release Date:** October 2026
- **License:** MIT License

## 2. Intended Use & Domain
MediAssist is an educational, research-oriented clinical decision support system designed to assist healthcare professionals by demonstrating multi-modal AI integration across preliminary symptom triage, chest radiograph screening, laboratory result interpretation, and patient risk prioritization.

### Primary Intended Users
- Healthcare practitioners and clinicians in training
- Medical researchers and biomedical informaticians
- AI/ML engineering teams evaluating clinical decision-support architectures

### Out-of-Scope / Prohibited Uses
- **Autonomous Clinical Diagnosis:** MediAssist must NOT be deployed as an autonomous diagnostic device.
- **Direct Prescriptive Therapy:** MediAssist does NOT prescribe medications, clinical dosing, or surgical procedures.
- **Emergency Delay:** MediAssist must never delay immediate patient access to emergency medical services.

## 3. High-Level Multi-Model Architecture
1. **Module A — NLP Symptom Triage Classifier:** Combines a deterministic safety layer (clinical red flags) with a calibrated TF-IDF text classifier.
2. **Module B — MobileNetV2 Chest X-Ray Screener:** Transfer learning CNN producing multi-label probabilities for 6 thoracic pathologies with Grad-CAM visual heatmaps.
3. **Module C — Clinical Laboratory Interpreter:** Authoritative clinical reference interval matching (Tietz & Harrison compendiums) with stat critical value alerts.
4. **Module D — Composite Patient Risk Aggregator:** Multi-criteria priority engine that synthesizes all modules without allowing low-risk scores to dilute emergency red flags.
5. **Module E — Interpretability & 10-Second Explanations:** Local interpretable model explanations (LIME), Grad-CAM, and standardized clinician summaries.

## 4. Safety & Ethical Safeguards
- **Deterministic Red-Flag Override:** Any acute life-threat trigger (e.g. crushing chest pain, stroke signs, severe hypoxemia SpO2 < 90%) forces an immediate `Emergency Warning` / `Emergency Review` priority, advising direct contact with emergency services (112 in India).
- **Mandatory 10-Second Explanations:** Every prediction is paired with a concise clinician summary structured into: Finding, Reason, Confidence, and Limitation.
- **Privacy Preservation:** All clinical profiles used in this project are synthetic demonstration cases. No protected health information (PHI) is ingested or retained.

## 5. Deployment Environment
- Python 3.11+
- TensorFlow / Keras 3.x
- Scikit-learn, Pandas, NumPy, Pillow, Plotly, LIME, Pytest
- Streamlit Community Cloud or local workstation
