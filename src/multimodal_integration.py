"""
MediAssist - Multimodal Chest X-ray & Symptom Integration Engine
Author: MediAssist Clinical AI Team

Synthesizes:
  1. Patient Demographics & Medical History
  2. Structured Symptoms & Contextual NLP Narrative (Negation, Temporality)
  3. Physiological Vital Signs
  4. Chest Radiograph Vision Model Findings & Grad-CAM Attention
  5. Clinical Decision Rules (Concordance, Discordance, Safety Net)

Provides:
  - Concordance Analysis (Findings supported by symptoms)
  - Discordance Analysis (Symptoms present despite normal X-ray, or unexpected incidental findings)
  - Independent Clinical Safety Screening (Emergency Warning, Prompt Assessment, Routine Review)
  - Transparent Rationale & Suggested Next Steps (Non-prescriptive, clinician-ready)
"""
from typing import Dict, Any, List, Optional, Tuple
from .config import EMERGENCY_CONTACT_INDIA


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLINICAL MULTIMODAL CONCORDANCE RULES
# ─────────────────────────────────────────────────────────────────────────────

# Known clinical correlations between symptoms, vital signs, and radiograph findings
CONCORDANCE_RULES = [
    {
        "id": "CONC-PNEUMONIA-001",
        "condition": "Lower Respiratory Infection / Pneumonia",
        "radiology_target": "Infiltration",
        "symptoms_required": ["fever", "cough"],
        "vital_clues": ["Temperature", "RespiratoryRate"],
        "description": "Patient reports fever and cough alongside model-flagged pulmonary infiltration.",
        "clinical_synthesis": (
            "Concordance High: The clinical constellation of cough, fever, and alveolar infiltration "
            "is consistent with community-acquired or acute infectious pneumonia. Clinical correlation "
            "with auscultatory findings (crackles/bronchial breath sounds) and inflammatory markers is indicated."
        )
    },
    {
        "id": "CONC-EFFUSION-001",
        "condition": "Pleural Effusion / Fluid Accumulation",
        "radiology_target": "Effusion",
        "symptoms_required": ["shortness of breath", "chest pain"],
        "vital_clues": ["SpO2"],
        "description": "Shortness of breath or pleuritic discomfort alongside model-flagged pleural effusion.",
        "clinical_synthesis": (
            "Concordance High: Model-flagged blunting of costophrenic angles or pleural opacity aligns "
            "with reported breathlessness or pleuritic chest discomfort. Consideration of decubitus views, "
            "thoracic ultrasound, or diagnostic thoracentesis may be warranted based on clinical context."
        )
    },
    {
        "id": "CONC-CARDIAC-001",
        "condition": "Cardiomegaly / Congestive Heart Failure",
        "radiology_target": "Cardiomegaly",
        "symptoms_required": ["shortness of breath", "fatigue"],
        "vital_clues": ["SystolicBP", "HeartRate"],
        "description": "Dyspnea on exertion or fatigue alongside an enlarged cardiac silhouette.",
        "clinical_synthesis": (
            "Concordance Moderate-High: An elevated cardiothoracic ratio (> 0.5 on PA projection) aligns "
            "with patient-reported exertional dyspnea or fatigue. Clinical correlation with jugular venous pressure, "
            "peripheral edema, and echocardiography is recommended."
        )
    },
    {
        "id": "CONC-PTX-001",
        "condition": "Pneumothorax / Pleural Air",
        "radiology_target": "Pneumothorax",
        "symptoms_required": ["chest pain", "shortness of breath"],
        "vital_clues": ["SpO2", "RespiratoryRate"],
        "description": "Acute chest pain or breathlessness with model-detected visceral pleural line / pneumothorax.",
        "clinical_synthesis": (
            "CRITICAL CONCORDANCE: Sudden-onset chest pain and dyspnea coinciding with model-flagged pneumothorax. "
            "Requires immediate professional physical evaluation for tension physiology (tracheal deviation, "
            "hemodynamic instability) and urgent thoracic imaging verification."
        )
    },
    {
        "id": "CONC-ATELECTASIS-001",
        "condition": "Atelectasis / Volume Loss",
        "radiology_target": "Atelectasis",
        "symptoms_required": ["cough", "shortness of breath"],
        "vital_clues": ["SpO2"],
        "description": "Mild dyspnea or cough with model-detected lung volume reduction / linear opacities.",
        "clinical_synthesis": (
            "Concordance Moderate: Sub-segmental volume loss or linear atelectasis correlates with reported cough "
            "or mild breathing limitation, common in post-operative, shallow-breathing, or mucus-plugging states."
        )
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISCORDANCE & CLINICAL DISCREPANCY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit_multimodal_discordance(
    symptoms: List[str],
    flagged_xray: List[str],
    vitals: Optional[Dict[str, Any]],
    narrative: str
) -> List[Dict[str, str]]:
    """
    Identifies clinically critical discrepancies where symptoms suggest acute pathology
    that an X-ray cannot reliably rule out, or unexpected incidental findings.
    """
    discrepancies: List[Dict[str, str]] = []
    sym_set = set(s.lower() for s in symptoms)

    # 1. Acute Shortness of Breath with "Normal" or Unremarkable X-ray
    if ("shortness of breath" in sym_set or "difficulty breathing" in sym_set) and (not flagged_xray or flagged_xray == ["Normal"]):
        discrepancies.append({
            "type": "Critical Negative-Image Discordance",
            "clinical_concern": "Dyspnea with Normal Radiograph",
            "explanation": (
                "The patient reports significant shortness of breath, but the chest radiograph model detected no overt acute opacity. "
                "CRITICAL CLINICAL NOTE: A normal chest X-ray does NOT exclude acute Pulmonary Embolism (PE), early bronchial asthma / bronchospasm, "
                "metabolic acidosis (e.g., DKA), or early-stage viral pneumonia. Further evaluation with pulse oximetry, arterial blood gas, or CT pulmonary angiogram (CTPA) may be required."
            )
        })

    # 2. Acute Severe Chest Pain with "Normal" X-ray
    if "chest pain" in sym_set and (not flagged_xray or flagged_xray == ["Normal"]):
        discrepancies.append({
            "type": "Critical Negative-Image Discordance",
            "clinical_concern": "Chest Pain with Normal Radiograph",
            "explanation": (
                "Patient reports chest pain, yet no acute pulmonary abnormality was flagged on the radiograph. "
                "CRITICAL CLINICAL NOTE: A normal chest X-ray does NOT rule out Acute Coronary Syndrome (STEMI / NSTEMI), "
                "acute pericarditis, or early aortic pathology. An urgent 12-lead ECG and serial cardiac troponin measurements are essential."
            )
        })

    # 3. High Fever with Productive Cough but "Normal" X-ray
    if "fever" in sym_set and "cough" in sym_set and (not flagged_xray or flagged_xray == ["Normal"]):
        discrepancies.append({
            "type": "Early Infection Discordance",
            "clinical_concern": "Febrile Respiratory Symptoms with Clear Radiograph",
            "explanation": (
                "Patient presents with fever and cough, but no focal consolidation is flagged. "
                "Early pneumonia (within the first 24-48 hours) or dehydration can result in a false-negative chest X-ray. "
                "Clinical auscultation, hydration reassessment, and re-imaging if symptoms persist are advised."
            )
        })

    # 4. Incidental Finding: Infiltration or Effusion flagged with NO reported respiratory symptoms
    if flagged_xray and flagged_xray != ["Normal"]:
        respiratory_syms = {"cough", "fever", "shortness of breath", "chest pain", "wheezing", "sputum"}
        if not sym_set.intersection(respiratory_syms):
            discrepancies.append({
                "type": "Incidental / Asymptomatic Radiographic Abnormality",
                "clinical_concern": f"Radiographic {', '.join(flagged_xray)} without acute respiratory complaint",
                "explanation": (
                    f"The vision model detected potential {', '.join(flagged_xray)}, but the patient reported no acute cough, fever, or dyspnea. "
                    "This may represent prior chronic scarring, baseline pleural thickening, atelectasis from shallow inspiration, or a false-positive model activation. "
                    "Comparison with prior historical radiographs is strongly recommended."
                )
            })

    return discrepancies


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTIMODAL INTEGRATION & CLINICAL SAFETY SYNTHESIZER
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalClinicalIntegrator:
    """
    Integrates clinical history, symptoms, vital signs, and radiograph model outputs.
    """

    @classmethod
    def synthesize_report(
        cls,
        patient_info: Dict[str, Any],
        symptom_data: Dict[str, Any],
        vital_signs: Optional[Dict[str, Any]],
        xray_result: Dict[str, Any],
        projection_view: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal clinical integration.
        """
        # Extract inputs
        age = patient_info.get("age", 35)
        sex = patient_info.get("sex", "Unknown")
        history = patient_info.get("medical_history", [])
        smoking = patient_info.get("smoking_history", "Non-smoker")
        medications = patient_info.get("medications", "")

        narrative = symptom_data.get("narrative", "")
        current_syms = symptom_data.get("current_symptoms", [])
        negated_syms = symptom_data.get("negated_symptoms", [])
        duration_days = float(symptom_data.get("duration_days", 2.0))
        severity_score = int(symptom_data.get("severity", 4))
        trajectory = symptom_data.get("trajectory", "Unchanged")

        flagged_xray = xray_result.get("flagged_findings", [])
        xray_probs = xray_result.get("probabilities", {})
        primary_xray_finding = xray_result.get("primary_finding", "No overt acute abnormality")
        top_xray_class = xray_result.get("top_class", "Normal")
        xray_ood = xray_result.get("ood_rejected", False)

        # ── Step 1: Safety & Emergency Screen ────────────────────────────────
        emergency_triggers: List[str] = []
        urgent_triggers: List[str] = []

        # Vital sign emergency triggers
        if vital_signs and isinstance(vital_signs, dict):
            spo2 = vital_signs.get("SpO2")
            if spo2 is not None and float(spo2) < 90.0:
                emergency_triggers.append(f"Critical Hypoxemia (SpO2 {spo2}% < 90%)")
            elif spo2 is not None and float(spo2) < 94.0:
                urgent_triggers.append(f"Moderate Hypoxemia (SpO2 {spo2}%)")

            hr = vital_signs.get("HeartRate")
            if hr is not None and (float(hr) > 150.0 or float(hr) < 40.0):
                emergency_triggers.append(f"Critical Heart Rate ({hr} bpm)")
            elif hr is not None and float(hr) > 115.0:
                urgent_triggers.append(f"Significant Tachycardia ({hr} bpm)")

            sys_bp = vital_signs.get("SystolicBP")
            if sys_bp is not None and float(sys_bp) < 80.0:
                emergency_triggers.append(f"Profound Hypotension / Shock ({sys_bp} mmHg)")
            elif sys_bp is not None and float(sys_bp) > 200.0:
                emergency_triggers.append(f"Severe Hypertensive Crisis ({sys_bp} mmHg)")

            temp = vital_signs.get("Temperature")
            if temp is not None and float(temp) >= 39.0:
                urgent_triggers.append(f"High Pyrexia (Body Temp {temp} °C)")

        # Narrative emergency triggers (severe breathing difficulty, stroke, syncope)
        if any(w in narrative.lower() for w in ["cannot breathe", "gasping for air", "stridor", "turning blue", "cyanosis"]):
            emergency_triggers.append("Severe Acute Respiratory Distress (Airway / Oxygenation compromise)")

        if any(w in narrative.lower() for w in ["crushing chest pain", "radiating to left arm", "chest pressure with sweating"]):
            emergency_triggers.append("Suspected Acute Coronary Syndrome (Severe retrosternal pain radiating to arm)")

        if any(w in narrative.lower() for w in ["facial droop", "slurred speech", "sudden arm weakness"]):
            emergency_triggers.append("Positive FAST Stroke Signs (Sudden focal neurological deficit)")

        # Radiograph acute findings: Tension Pneumothorax
        if "Pneumothorax" in flagged_xray and xray_probs.get("Pneumothorax", 0.0) >= 0.65:
            emergency_triggers.append(f"High-Probability Pneumothorax on Radiograph ({xray_probs.get('Pneumothorax')*100:.1f}%)")

        # ── Step 2: Concordance & Discordance Analysis ───────────────────────
        concordant_insights: List[Dict[str, str]] = []
        sym_set = set(s.lower() for s in current_syms)

        for rule in CONCORDANCE_RULES:
            target = rule["radiology_target"]
            # Check if pathology was flagged on X-ray
            if target in flagged_xray:
                # Check how many matching symptoms are present
                matching_syms = [s for s in rule["symptoms_required"] if s in sym_set]
                if matching_syms:
                    concordant_insights.append({
                        "condition": rule["condition"],
                        "pathology": target,
                        "matched_symptoms": matching_syms,
                        "synthesis": rule["clinical_synthesis"]
                    })

        # Discordance Analysis
        discordant_insights = audit_multimodal_discordance(
            symptoms=current_syms,
            flagged_xray=flagged_xray,
            vitals=vital_signs,
            narrative=narrative
        )

        # ── Step 3: Determine Multimodal Decision Status ─────────────────────
        if emergency_triggers:
            decision_status = "EMERGENCY WARNING"
            status_color = "#ef4444"
            status_badge = "🚨 Emergency Warning"
            action_advisory = (
                f"IMMEDIATE EMERGENCY MEDICAL CARE REQUIRED. Call National Emergency Services "
                f"(Dial {EMERGENCY_CONTACT_INDIA} in India) or go to the nearest emergency room immediately. Do not delay care."
            )
        elif urgent_triggers or concordant_insights or (flagged_xray and flagged_xray != ["Normal"]) or severity_score >= 7:
            decision_status = "PROMPT CLINICAL ASSESSMENT RECOMMENDED"
            status_color = "#f59e0b"
            status_badge = "⚠️ Prompt Clinical Assessment"
            action_advisory = (
                "Evaluation by a qualified physician or urgent care clinician recommended within 4 to 12 hours. "
                "Professional clinical auscultation and formal radiologist image interpretation are warranted."
            )
        elif not current_syms and (not flagged_xray or flagged_xray == ["Normal"]):
            decision_status = "INSUFFICIENT CLINICAL INFORMATION"
            status_color = "#64748b"
            status_badge = "❓ Insufficient Information"
            action_advisory = (
                "Limited clinical parameters provided. Please supply detailed symptom history, onset duration, "
                "or measured vital signs for a comprehensive evaluation."
            )
        else:
            decision_status = "ROUTINE CLINICIAN REVIEW"
            status_color = "#0d9488"
            status_badge = "🟢 Routine Clinician Review"
            action_advisory = (
                "Sub-acute presentation with no overt critical signs on screening. Schedule a routine primary care follow-up. "
                "NOTE: Routine status is not a guarantee of safety; if breathing difficulty or chest pain develops, seek emergency care."
            )

        # ── Step 4: Missing Information & Recommended Investigations ────────
        missing_items: List[str] = []
        if not vital_signs or not vital_signs.get("SpO2"):
            missing_items.append("Oxygen Saturation (SpO2 %): critical to assess subclinical hypoxemia.")
        if not vital_signs or not vital_signs.get("Temperature"):
            missing_items.append("Body Temperature (°C): essential to distinguish infectious consolidation from atelectasis.")
        if projection_view == "Unknown":
            missing_items.append("Radiographic Projection (PA vs AP): cardiothoracic ratio assessment is reliable only on PA projection.")
        if not history:
            missing_items.append("Prior Pulmonary History: previous chest imaging needed to distinguish acute opacity from chronic scarring.")

        # Suggested clinical next steps (non-prescriptive)
        suggested_steps: List[str] = [
            "Formal radiologic interpretation by a certified radiologist alongside comparison with historical radiographs.",
            "Bedside clinical correlation: thoracic percussion, vocal resonance, and stethoscopic auscultation."
        ]
        if "Infiltration" in flagged_xray or "fever" in sym_set:
            suggested_steps.append("Complete Blood Count (CBC) with differential and inflammatory markers (CRP/ESR) if bacterial infection is suspected.")
        if "Cardiomegaly" in flagged_xray or "chest pain" in sym_set:
            suggested_steps.append("12-Lead Electrocardiogram (ECG) and serum cardiac biomarkers (Troponin I / NT-proBNP).")
        if "Effusion" in flagged_xray:
            suggested_steps.append("Thoracic point-of-care ultrasound (POCUS) to quantify pleural fluid depth and loculation.")

        # ── Step 5: Synthesize Final Report Structure ────────────────────────
        return {
            "decision_status": decision_status,
            "status_badge": status_badge,
            "status_color": status_color,
            "action_advisory": action_advisory,
            "emergency_triggers": emergency_triggers,
            "urgent_triggers": urgent_triggers,
            "concordant_insights": concordant_insights,
            "discordant_insights": discordant_insights,
            "primary_radiology_finding": primary_xray_finding,
            "top_radiology_class": top_xray_class,
            "flagged_pathologies": flagged_xray,
            "radiology_probabilities": xray_probs,
            "projection_view": projection_view,
            "patient_summary": {
                "age": age,
                "sex": sex,
                "duration_days": duration_days,
                "severity_score": severity_score,
                "trajectory": trajectory,
                "smoking": smoking,
                "history": history,
                "current_symptoms": current_syms,
                "negated_symptoms": negated_syms,
                "vital_signs": vital_signs or {}
            },
            "missing_information": missing_items,
            "suggested_next_steps": suggested_steps,
            "limitations_notice": (
                "MediAssist AI Multimodal CDSS is an educational and clinical research demonstration. "
                "It does NOT generate definitive diagnoses, replace formal radiologist reports, or substitute "
                "for direct physician examination. MobileNetV2 features represent statistical pattern associations. "
                "Always seek licensed medical evaluation for acute complaints."
            )
        }
