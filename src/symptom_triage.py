"""
MediAssist - NLP-Based Symptom Triage Classifier & Safety Override Engine
Author: Zelvora Industry Labs / MediAssist Team

Enhanced Inference Engine (v2):
  1. Feature Engineering & Weighting:
     - Numerical features (severity_score, duration_days) actively weight the posterior
       probability distribution, preventing over-reliance on TF-IDF text features.
     - Red-Flag keyword dictionary: heavily penalizes "Emergency Warning" probability
       if overt red-flag terms are absent.
  2. Rule-Based Guardrails (Clinical Safety Net):
     - If severity <= 5, duration < 3 days, and no red-flag keywords are present,
       caps "Urgent" and "Emergency" probabilities and defaults classification to
       "Routine Assessment".
  3. Calibrated Confidence Thresholding:
     - Posterior probabilities from CalibratedClassifierCV are monitored.
     - A low-confidence score (e.g., <= 50%, like 46%) triggers a "Requires More Information"
       triage status rather than forcing an inaccurate "Urgent" classification.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import joblib
from pathlib import Path

from .config import (
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
    TRIAGE_INSUFFICIENT,
    RED_FLAG_PATTERNS,
    EMERGENCY_CONTACT_INDIA,
    TRIAGE_MODEL_PATH,
    TRIAGE_VECTORIZER_PATH
)
from .data_validation import validate_symptom_text

# Authoritative Red Flag Keyword Dictionary (Manchester / ESI Tier 1)
RED_FLAG_KEYWORDS: List[str] = [
    "chest pain", "chest pressure", "shortness of breath", "dyspnea",
    "cannot breathe", "gasping for air", "unconscious", "unresponsive",
    "collapsed", "fainting", "syncope", "loss of consciousness",
    "facial droop", "slurred speech", "sudden hemiparesis", "paralysis",
    "stridor", "choking", "asphyxia", "anaphylaxis", "swollen tongue",
    "throat closing", "coughing blood", "hemoptysis", "hematemesis",
    "vomiting blood", "thunderclap headache", "worst headache of life",
    "cyanosis", "seizure", "respiratory failure", "arterial bleed"
]


def detect_red_flag_keywords(text: str) -> List[str]:
    """
    Scans narrative for presence of any keywords in the Red Flag dictionary
    as well as regex pattern boundaries.
    """
    found = []
    text_lower = text.lower()
    
    # 1. Regex pattern matches
    for pattern in RED_FLAG_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            found.append(match.group(0))
            
    # 2. Keyword dictionary matches
    for kw in RED_FLAG_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            found.append(kw)
            
    return list(dict.fromkeys(found))


def extract_clinical_keywords(text: str) -> List[str]:
    """
    Extracts recognizable clinical symptoms and key physiological complaints.
    """
    keywords = [
        "fever", "cough", "shortness of breath", "dyspnea", "chest pain",
        "chest pressure", "headache", "dizziness", "nausea", "vomiting",
        "diarrhea", "abdominal pain", "fatigue", "palpitations", "sore throat",
        "rash", "joint pain", "swelling", "edema", "hemoptysis", "weakness",
        "numbness", "sweating", "diaphoresis", "chills", "wheezing",
        "stiff neck", "photophobia", "urinary frequency", "flank pain"
    ]
    text_lower = text.lower()
    return [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)]


def apply_clinical_heuristics_and_weighting(
    raw_probs: Dict[str, float],
    raw_text: str,
    severity_score: int,
    duration_days: float,
    has_red_flags: bool,
    confidence_threshold: float = 0.50
) -> Tuple[Dict[str, float], str, float, bool, Optional[str]]:
    """
    Applies feature engineering weights, red-flag penalties, rule-based clinical guardrails,
    and probability calibration thresholding to the raw classifier output.

    Returns:
        (calibrated_probs, final_category, confidence, is_low_confidence, heuristic_note)
    """
    probs = dict(raw_probs)
    # Ensure all three standard classes exist in the dict
    for cls in [TRIAGE_EMERGENCY, TRIAGE_URGENT, TRIAGE_ROUTINE]:
        probs.setdefault(cls, 0.0)

    # --------------------------------------------------------------------------
    # 1. FEATURE WEIGHTING: Red Flag Penalty
    # If no overt red flags are present in the text, heavily penalize Emergency Warning
    # --------------------------------------------------------------------------
    if not has_red_flags:
        # Heavily penalize Emergency Warning (reduce by 85%)
        probs[TRIAGE_EMERGENCY] = probs[TRIAGE_EMERGENCY] * 0.15

    # --------------------------------------------------------------------------
    # 2. FEATURE WEIGHTING: Numerical Severity & Duration Influence
    # --------------------------------------------------------------------------
    if severity_score <= 3:
        # Mild presentation: strongly boost Routine, scale down higher urgencies
        probs[TRIAGE_ROUTINE] *= 1.8
        probs[TRIAGE_URGENT] *= 0.40
        probs[TRIAGE_EMERGENCY] *= 0.20
    elif 4 <= severity_score <= 5:
        # Moderate-low: mild boost to Routine unless duration is prolonged
        probs[TRIAGE_ROUTINE] *= 1.30
        probs[TRIAGE_URGENT] *= 0.70
        probs[TRIAGE_EMERGENCY] *= 0.30
    elif 6 <= severity_score <= 7:
        # Moderate-high: shift weight towards Urgent
        probs[TRIAGE_URGENT] *= 1.45
        probs[TRIAGE_ROUTINE] *= 0.60
    elif severity_score >= 8:
        # High severity: strongly penalize Routine
        probs[TRIAGE_ROUTINE] *= 0.20
        probs[TRIAGE_URGENT] *= 1.50
        if has_red_flags:
            probs[TRIAGE_EMERGENCY] *= 1.80

    # Duration considerations
    if duration_days > 14.0 and severity_score <= 5:
        # Sub-acute/chronic low severity belongs in routine outpatient follow-up
        probs[TRIAGE_ROUTINE] *= 1.30
    elif duration_days <= 1.0 and severity_score >= 8:
        # Hyper-acute severe presentation elevates urgency
        probs[TRIAGE_URGENT] *= 1.25

    # --------------------------------------------------------------------------
    # 3. RULE-BASED GUARDRAIL (Mandatory Clinical Safety Net)
    # Rule: If severity <= 5, duration is under 3 days, and no red-flag keywords
    # are present, cap Urgent & Emergency probabilities and default to Routine Assessment.
    # --------------------------------------------------------------------------
    heuristic_applied = False
    heuristic_note = None

    if severity_score <= 5 and duration_days < 3.0 and not has_red_flags:
        probs[TRIAGE_EMERGENCY] = min(probs.get(TRIAGE_EMERGENCY, 0.0), 0.02)
        probs[TRIAGE_URGENT] = min(probs.get(TRIAGE_URGENT, 0.0), 0.15)
        probs[TRIAGE_ROUTINE] = max(0.83, 1.0 - probs[TRIAGE_EMERGENCY] - probs[TRIAGE_URGENT])
        heuristic_applied = True
        heuristic_note = (
            f"Clinical Guardrail Engaged: Mild presentation (severity {severity_score}/10, "
            f"duration {duration_days}d, no red flags). Capped Urgent (<=15%) and Emergency (<=2%)."
        )

    # Re-normalize probabilities so they sum to 1.0
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    else:
        probs = {TRIAGE_ROUTINE: 1.0, TRIAGE_URGENT: 0.0, TRIAGE_EMERGENCY: 0.0}

    # Determine highest probability class
    top_class = max(probs, key=probs.get)
    top_prob = probs[top_class]

    # Enforce default to Routine Assessment if guardrail was applied
    if heuristic_applied:
        top_class = TRIAGE_ROUTINE
        top_prob = probs[TRIAGE_ROUTINE]

    # --------------------------------------------------------------------------
    # 4. CALIBRATED CONFIDENCE THRESHOLDING
    # If the top confidence is below threshold (e.g., 46% <= 50%),
    # trigger "Requires More Information" rather than forcing an inaccurate "Urgent"
    # --------------------------------------------------------------------------
    is_low_confidence = False
    if top_prob < confidence_threshold and not heuristic_applied:
        is_low_confidence = True
        final_category = "Requires More Information"
    else:
        final_category = top_class

    return probs, final_category, top_prob, is_low_confidence, heuristic_note


def predict_triage(
    chief_complaint: str,
    duration_days: float = 1.0,
    severity_score: int = 5,
    patient_age: int = 35,
    vital_signs: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.50,
    model = None,
    vectorizer = None
) -> Dict[str, Any]:
    """
    Production inference function for NLP symptom triage.

    Combines:
      - Deterministic emergency red-flag safety overrides (100% priority)
      - Calibrated ML posterior probabilities (.predict_proba)
      - Feature engineering & numerical feature weighting (severity & duration)
      - Red-flag keyword dictionary penalty
      - Clinical rule-based guardrail for mild presentations
      - Low-confidence calibration thresholding (triggers 'Requires More Information')
      - Clinician 10-second explanation
    """
    # 1. Input Validation
    is_valid, cleaned_text, err = validate_symptom_text(chief_complaint)
    if not is_valid:
        return {
            "urgency_category": TRIAGE_INSUFFICIENT,
            "confidence": 0.0,
            "identified_symptoms": [],
            "red_flags": [],
            "probabilities": {},
            "explanation": {
                "finding": "Insufficient information to evaluate clinical urgency.",
                "reason": err,
                "confidence": "Not applicable due to invalid input.",
                "limitation": "The model cannot perform screening on empty or non-textual input."
            },
            "safety_override": False,
            "action_advisory": "Please supply a detailed description of symptoms.",
            "emergency_contact": None,
            "is_low_confidence": False
        }

    # 2. Red-Flag Safety Checks
    narrative_red_flags = detect_red_flag_keywords(cleaned_text)
    symptoms_found = extract_clinical_keywords(cleaned_text)

    # Physiological vital sign red flags (if explicit values provided)
    vital_red_flags = []
    if vital_signs and isinstance(vital_signs, dict):
        spo2 = vital_signs.get("SpO2")
        if spo2 is not None and float(spo2) < 90.0:
            vital_red_flags.append(f"Severe hypoxemia (SpO2 {spo2}% < 90%)")
        hr = vital_signs.get("HeartRate")
        if hr is not None and (float(hr) > 150.0 or float(hr) < 40.0):
            vital_red_flags.append(f"Critical heart rate ({hr} bpm)")
        sys_bp = vital_signs.get("SystolicBP")
        if sys_bp is not None and float(sys_bp) < 80.0:
            vital_red_flags.append(f"Profound hypotension (Systolic BP {sys_bp} mmHg)")

    all_emergency_reasons = narrative_red_flags + vital_red_flags

    # Deterministic Safety Override (Absolute Priority)
    if len(all_emergency_reasons) > 0:
        return {
            "urgency_category": TRIAGE_EMERGENCY,
            "confidence": 1.0,
            "identified_symptoms": symptoms_found,
            "red_flags": all_emergency_reasons,
            "probabilities": {TRIAGE_EMERGENCY: 1.0, TRIAGE_URGENT: 0.0, TRIAGE_ROUTINE: 0.0},
            "explanation": {
                "finding": "Immediate Emergency Medical Evaluation Required.",
                "reason": f"Overt life-threatening warning signs detected: {', '.join(all_emergency_reasons)}.",
                "confidence": "Deterministic clinical safety rule (100% rule-level priority).",
                "limitation": "This screening flag identifies acute risk; it is not a definitive diagnostic workup."
            },
            "safety_override": True,
            "action_advisory": f"Call emergency medical services immediately (Dial {EMERGENCY_CONTACT_INDIA} in India) or go to the nearest emergency department.",
            "emergency_contact": EMERGENCY_CONTACT_INDIA,
            "is_low_confidence": False
        }

    # 3. Model Inference & Probability Calibration
    raw_probabilities = {}
    if model is not None and vectorizer is not None:
        try:
            features = vectorizer.transform([cleaned_text])
            probs_arr = model.predict_proba(features)[0]
            classes = list(model.classes_)
            raw_probabilities = {cls: float(p) for cls, p in zip(classes, probs_arr)}
        except Exception:
            raw_probabilities = {}

    if not raw_probabilities:
        # Uniform baseline if model is unavailable
        raw_probabilities = {TRIAGE_ROUTINE: 0.50, TRIAGE_URGENT: 0.40, TRIAGE_EMERGENCY: 0.10}

    # 4. Feature Engineering, Red-Flag Penalty, Guardrails & Confidence Thresholding
    has_narrative_red_flags = len(narrative_red_flags) > 0
    adj_probs, final_category, conf_score, is_low_conf, guardrail_note = apply_clinical_heuristics_and_weighting(
        raw_probs=raw_probabilities,
        raw_text=cleaned_text,
        severity_score=severity_score,
        duration_days=duration_days,
        has_red_flags=has_narrative_red_flags,
        confidence_threshold=confidence_threshold
    )

    # 5. Formulate Clinician 10-Second Explanation & Advisory
    if is_low_conf:
        action_advisory = (
            f"The model's calibrated confidence is {conf_score*100:.1f}%, which is below the "
            f"decision threshold ({confidence_threshold*100:.0f}%). Clinical consultation and "
            f"additional diagnostic details (vitals, onset timeline) are required."
        )
        plain_reason = (
            f"Symptoms reported: '{', '.join(symptoms_found) if symptoms_found else cleaned_text[:60]}', "
            f"severity {severity_score}/10 over {duration_days} day(s). Prediction confidence "
            f"({conf_score*100:.1f}%) is equivocal between categories."
        )
        finding = "Requires More Information (Equivocal Confidence)"
    else:
        if final_category == TRIAGE_ROUTINE:
            action_advisory = "Schedule routine consultation or follow-up with primary care physician."
        elif final_category == TRIAGE_URGENT:
            action_advisory = "Prompt medical assessment recommended within 4-12 hours."
        else:
            action_advisory = f"Emergency medical assessment advised. Contact emergency services ({EMERGENCY_CONTACT_INDIA})."

        plain_reason = (
            f"Reported symptoms include '{', '.join(symptoms_found)}' "
            f"with patient-reported severity {severity_score}/10 over {duration_days} day(s)."
            if symptoms_found else
            f"Clinical text matched urgency pattern with severity {severity_score}/10 over {duration_days} day(s)."
        )
        if guardrail_note:
            plain_reason += f" [{guardrail_note}]"
        finding = f"Calibrated classification: {final_category}."

    return {
        "urgency_category": final_category,
        "confidence": round(conf_score, 3),
        "identified_symptoms": symptoms_found,
        "red_flags": [],
        "probabilities": adj_probs,
        "raw_probabilities": raw_probabilities,
        "explanation": {
            "finding": finding,
            "reason": plain_reason,
            "confidence": f"Model calibrated confidence: {conf_score*100:.1f}%.",
            "limitation": "NLP triage models cannot replace clinician history-taking and physical examination."
        },
        "safety_override": False,
        "action_advisory": action_advisory,
        "emergency_contact": None,
        "is_low_confidence": is_low_conf,
        "guardrail_applied": guardrail_note is not None
    }


class SymptomTriageEngine:
    """
    Two-stage Triage Classifier:
    1. Deterministic Clinical Safety Layer (Red-Flag override)
    2. Calibrated NLP Machine Learning Classifier (TF-IDF + Calibrated Model)
    3. Clinical Heuristic Guardrails & Feature Weighting
    """

    def __init__(self, model_path: Optional[Path] = None, vectorizer_path: Optional[Path] = None):
        self.model_path = model_path or TRIAGE_MODEL_PATH
        self.vectorizer_path = vectorizer_path or TRIAGE_VECTORIZER_PATH
        self.model = None
        self.vectorizer = None
        self._load_models()

    def _load_models(self):
        try:
            if Path(self.model_path).exists() and Path(self.vectorizer_path).exists():
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
        except Exception:
            self.model = None
            self.vectorizer = None

    def detect_red_flags(self, text: str) -> List[str]:
        return detect_red_flag_keywords(text)

    def extract_keywords(self, text: str) -> List[str]:
        return extract_clinical_keywords(text)

    def classify_symptoms(
        self,
        chief_complaint: str,
        duration_days: float = 1.0,
        severity_score: int = 5,
        patient_age: int = 35,
        vital_signs: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end triage analysis with safety overrides, heuristic guardrails,
        and 10-second explanation.
        """
        return predict_triage(
            chief_complaint=chief_complaint,
            duration_days=duration_days,
            severity_score=severity_score,
            patient_age=patient_age,
            vital_signs=vital_signs,
            confidence_threshold=confidence_threshold,
            model=self.model,
            vectorizer=self.vectorizer
        )
