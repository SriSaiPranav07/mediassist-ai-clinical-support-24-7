# MediAssist System Architecture & Technical Specifications

## 1. System Philosophy
MediAssist is architected around three foundational healthcare AI principles:
1. **Separation of Concerns:** Each diagnostic modality (NLP symptoms, Computer Vision radiograph screening, Laboratory rules) executes independently and maintains transparent intermediate states.
2. **Deterministic Safety Precedence:** Statistical machine learning models must NEVER override explicit clinical red flags. High-risk life threats automatically force emergency triage.
3. **Clinician-Centric Interpretability:** Every model decision generates a plain-English explanation digestible in under 10 seconds (Finding, Reason, Confidence, Limitation).

---

## 2. Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph DataInputs["1. Patient Data Ingestion"]
        In1["Symptom Narrative (Chief Complaint)"]
        In2["Patient Vitals (HR, BP, SpO2, Temp, RR)"]
        In3["Chest Radiograph (PNG / JPEG / TIFF)"]
        In4["Laboratory Panels (Metabolic, Hepatic, Cardiac)"]
    end

    subgraph ValidationLayer["2. Validation & Safety Pre-filter"]
        Val1["Data Type & Physiological Boundary Validation"]
        Val2{"Deterministic Red-Flag Check?"}
    end

    subgraph Modalities["3. Multi-Modal Inference Modules"]
        ModNLP["Module A: Calibrated TF-IDF Triage Classifier"]
        ModCNN["Module B: MobileNetV2 Chest X-Ray Screener"]
        ModLab["Module C: Tietz/Harrison Laboratory Engine"]
    end

    subgraph Explainability["4. Model Interpretability"]
        ExpNLP["LIME Token Attribution"]
        ExpCNN["Grad-CAM Saliency Maps"]
        ExpLab["Reference Boundary Delta Analysis"]
        Exp10s["Standardized 10-Second Explanations"]
    end

    subgraph Aggregation["5. Composite Patient Risk Engine"]
        EngPriority["Transparent Multi-Criteria Scoring (0-100)"]
        EngSafety{"Any Life-Threats or Critical Labs?"}
        ActionEmerg["Emergency Review (Floor Score 85.0)"]
        ActionHigh["High Priority Review"]
        ActionStd["Standard Clinical Queuing"]
    end

    subgraph Presentation["6. Streamlit Clinical Dashboard"]
        UI1["Clinical Overview & Cohort Monitoring"]
        UI2["Symptom Triage & Red-Flag Alerts"]
        UI3["Radiograph Screening with Saliency"]
        UI4["Laboratory Abnormal & Critical Flags"]
        UI5["Patient Risk Prioritization & Audit Trail"]
        UI6["Explainability Center & Model Cards"]
    end

    DataInputs --> ValidationLayer
    Val2 -- "Red-Flag Detected" --> ActionEmerg
    Val2 -- "No Immediate Red-Flag" --> Modalities
    
    In1 --> ModNLP
    In3 --> ModCNN
    In4 --> ModLab

    ModNLP --> ExpNLP --> Exp10s
    ModCNN --> ExpCNN --> Exp10s
    ModLab --> ExpLab --> Exp10s

    ModNLP & In2 & ModLab & ModCNN --> Aggregation
    EngSafety -- "Yes" --> ActionEmerg
    EngSafety -- "No" --> EngPriority
    EngPriority --> ActionHigh
    EngPriority --> ActionStd

    ActionEmerg & ActionHigh & ActionStd --> Presentation
```

---

## 3. Data Flow & Contracts

| Module | Input Contract | Output Contract | Model / Technique |
| :--- | :--- | :--- | :--- |
| **Symptom Triage** | Text narrative, duration, severity, age | Urgency class, calibrated probabilities, keywords | Calibrated Logistic Regression + TF-IDF |
| **Chest X-Ray Screener** | 224x224 RGB Radiograph tensor | 6-class pathology probabilities, Grad-CAM heatmap | MobileNetV2 Transfer Learning CNN |
| **Laboratory Interpreter** | Test name, numeric value, unit, sex, age | Status (Normal, Low, High, Critical), Implication | Deterministic Tietz & Harrison Rule Base |
| **Risk Prioritization** | Triage, Vitals, Labs, X-Ray outputs | Priority level, composite score (0-100), audit trail | Weighted Multi-Criteria Priority Matrix |
