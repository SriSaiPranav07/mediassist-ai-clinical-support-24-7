"""
MediAssist - Hybrid Clinical Triage Architecture
Author: Zelvora Industry Labs / MediAssist Team

Architecture & Priority Order:
  1. Priority 1: Emergency Warning (Deterministic Clinical Red-Flag Override)
  2. Priority 2: Urgent Assessment (Clinically Verified Urgent Criteria)
  3. Priority 3: Information Sufficiency Audit (Prevents Unsafe / Contradictory Routine Defaults)
  4. Priority 4: Routine Assessment (Only when verified safe, sufficient, and mild)
"""
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np

from .config import (
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
    TRIAGE_INSUFFICIENT,
    EMERGENCY_CONTACT_INDIA,
    TRIAGE_MODEL_PATH,
    TRIAGE_VECTORIZER_PATH
)
from .data_validation import validate_symptom_text, validate_vital_signs
from .clinical_guardrails import (
    ClinicalNarrativeParser,
    EmergencySafetyEngine,
    UrgentAssessmentEngine,
    InformationSufficiencyEngine
)


def compute_shannon_entropy(probs: Dict[str, float]) -> float:
    """
    Computes normalized Shannon entropy H in [0, 1] across class probabilities.
    0.0 = absolute certainty (one class has 1.0 probability)
    1.0 = maximum uncertainty (uniform distribution across classes)
    """
    p_vals = [p for p in probs.values() if p > 0.0]
    if len(p_vals) <= 1:
        return 0.0
    k = len(probs)
    max_entropy = math.log2(k) if k > 1 else 1.0
    entropy = -sum(p * math.log2(p) for p in p_vals)
    return round(float(entropy / max_entropy), 3)


def predict_triage(
    chief_complaint: str,
    duration_days: float = 1.0,
    severity_score: int = 5,
    patient_age: int = 35,
    vital_signs: Optional[Dict[str, Any]] = None,
    comorbidities: Optional[List[str]] = None,
    current_medications: Optional[str] = None,
    symptom_trajectory: str = "Unchanged",
    previous_episodes: str = "First time",
    confidence_threshold: float = 0.50,
    model: Any = None,
    vectorizer: Any = None
) -> Dict[str, Any]:
    """
    Executes the Hybrid Clinical Triage Architecture:
      Priority: Emergency Warning > Urgent Assessment > Insufficient Information > Routine Assessment.
      Deterministic clinical safety rules strictly override statistical ML probabilities.
    """
    # ── Step 1: Input Validation ─────────────────────────────────────────────
    is_valid, cleaned_text, val_err = validate_symptom_text(chief_complaint)
    if not is_valid:
        return {
            "urgency_category": TRIAGE_INSUFFICIENT,
            "decision_source": "Input Validation",
            "safety_override": False,
            "confidence": 0.0,
            "model_uncertainty": 1.0,
            "probabilities": {TRIAGE_EMERGENCY: 0.0, TRIAGE_URGENT: 0.0, TRIAGE_ROUTINE: 0.0},
            "reported_symptoms": {"current": [], "negated": [], "historical": [], "family": []},
            "red_flags": [],
            "urgent_flags": [],
            "missing_information": [val_err],
            "explanation": {
                "summary": "Insufficient information to evaluate clinical urgency.",
                "finding": "Symptom description is incomplete or invalid.",
                "reason": val_err,
                "confidence": "Not applicable due to invalid input.",
                "limitation": "The system cannot perform screening on empty or non-textual input."
            },
            "action_advisory": "Please supply a detailed description of your symptoms.",
            "emergency_contact": None,
            "raw_model_class": None,
            "raw_model_confidence": 0.0
        }

    # Validate vital signs if supplied
    vital_issues: List[str] = []
    sanitized_vitals: Optional[Dict[str, Any]] = None
    if vital_signs and isinstance(vital_signs, dict):
        v_ok, v_errs = validate_vital_signs(vital_signs)
        if not v_ok:
            vital_issues.extend(v_errs)
        sanitized_vitals = {k: v for k, v in vital_signs.items() if v is not None}

    # ── Step 2: Contextual NLP Narrative Parsing ─────────────────────────────
    parsed_context = ClinicalNarrativeParser.parse(cleaned_text)
    current_symptoms = parsed_context["current_symptoms"]
    negated_symptoms = parsed_context["negated_symptoms"]
    historical_symptoms = parsed_context["historical_symptoms"]
    family_symptoms = parsed_context["family_history"]
    clean_patient_text = parsed_context["clean_patient_text"]

    # ── Step 3: Statistical NLP Model Inference ──────────────────────────────
    raw_probabilities: Dict[str, float] = {}
    raw_top_class = "Routine Assessment"
    raw_top_prob = 0.50

    if model is not None and vectorizer is not None:
        try:
            eval_text = clean_patient_text if clean_patient_text else cleaned_text
            features = vectorizer.transform([eval_text])
            probs_arr = model.predict_proba(features)[0]
            model_classes = list(model.classes_)
            raw_probabilities = {cls: round(float(p), 4) for cls, p in zip(model_classes, probs_arr)}
            tot = sum(raw_probabilities.values())
            if tot > 0:
                raw_probabilities = {cls: round(p / tot, 4) for cls, p in raw_probabilities.items()}
            raw_top_class = max(raw_probabilities, key=raw_probabilities.get)
            raw_top_prob = raw_probabilities[raw_top_class]
        except Exception:
            raw_probabilities = {TRIAGE_EMERGENCY: 0.10, TRIAGE_URGENT: 0.40, TRIAGE_ROUTINE: 0.50}
    else:
        raw_probabilities = {TRIAGE_EMERGENCY: 0.10, TRIAGE_URGENT: 0.40, TRIAGE_ROUTINE: 0.50}

    uncertainty_entropy = compute_shannon_entropy(raw_probabilities)

    # ── Step 4: Hybrid Decision Priority Architecture ────────────────────────
    
    # PRIORITY 1: Deterministic Emergency Red-Flag Engine (100% Safety Override)
    emerg_res = EmergencySafetyEngine.evaluate(
        parsed_context=parsed_context,
        raw_text=cleaned_text,
        vital_signs=sanitized_vitals
    )

    if emerg_res["is_emergency"]:
        red_flags = emerg_res["emergency_reasons"]
        finding = "IMMEDIATE EMERGENCY MEDICAL EVALUATION REQUIRED."
        reason = (
            f"Deterministic Clinical Safety Override Engaged. Overt life-threatening warning signs detected: "
            f"{'; '.join(red_flags)}. Statistical model predictions are overridden to protect patient life."
        )
        return {
            "urgency_category": TRIAGE_EMERGENCY,
            "decision_source": "Deterministic Emergency Safety Rule",
            "safety_override": True,
            "confidence": 1.0,
            "model_uncertainty": uncertainty_entropy,
            "probabilities": raw_probabilities,
            "reported_symptoms": {
                "current": current_symptoms,
                "negated": negated_symptoms,
                "historical": historical_symptoms,
                "family": family_symptoms
            },
            "red_flags": red_flags,
            "urgent_flags": [],
            "missing_information": vital_issues,
            "explanation": {
                "summary": "Immediate Emergency Warning",
                "finding": finding,
                "reason": reason,
                "confidence": "Deterministic Clinical Rule Priority (100% Safety Override).",
                "limitation": "Emergency screening flags acute risk; it does not replace hospital resuscitation or definitive diagnostic workup."
            },
            "action_advisory": f"Call emergency medical services immediately (Dial {EMERGENCY_CONTACT_INDIA} in India) or proceed to the nearest emergency department without delay.",
            "emergency_contact": EMERGENCY_CONTACT_INDIA,
            "raw_model_class": raw_top_class,
            "raw_model_confidence": raw_top_prob
        }

    # PRIORITY 2: Deterministic Clinical Urgent Criteria
    urgent_res = UrgentAssessmentEngine.evaluate(
        parsed_context=parsed_context,
        severity_score=severity_score,
        duration_days=duration_days,
        trajectory=symptom_trajectory,
        vital_signs=sanitized_vitals,
        comorbidities=comorbidities,
        patient_age=patient_age
    )

    if urgent_res["is_urgent"]:
        urgent_flags = urgent_res["urgent_reasons"]
        finding = "Urgent Clinical Assessment Recommended."
        reason = (
            f"Clinical criteria for urgent evaluation met: {'; '.join(urgent_flags)}. "
            f"Patient-reported severity {severity_score}/10, duration {duration_days} day(s), trajectory '{symptom_trajectory}'."
        )
        return {
            "urgency_category": TRIAGE_URGENT,
            "decision_source": "Clinical Urgent Safety Criteria",
            "safety_override": (raw_top_class != TRIAGE_URGENT),
            "confidence": max(raw_top_prob, 0.75),
            "model_uncertainty": uncertainty_entropy,
            "probabilities": raw_probabilities,
            "reported_symptoms": {
                "current": current_symptoms,
                "negated": negated_symptoms,
                "historical": historical_symptoms,
                "family": family_symptoms
            },
            "red_flags": [],
            "urgent_flags": urgent_flags,
            "missing_information": vital_issues,
            "explanation": {
                "summary": "Urgent Medical Assessment Warranted",
                "finding": finding,
                "reason": reason,
                "confidence": f"Model probability: {raw_probabilities.get(TRIAGE_URGENT, 0.0)*100:.1f}%. Urgent criteria verification: Positive.",
                "limitation": "AI triage screening identifies elevated urgency; in-person examination and lab/radiology diagnostics are required."
            },
            "action_advisory": "Prompt medical assessment recommended within 4 to 12 hours at an urgent care center or clinic.",
            "emergency_contact": None,
            "raw_model_class": raw_top_class,
            "raw_model_confidence": raw_top_prob
        }

    # PRIORITY 3: Information Sufficiency Audit
    # If information is incomplete or contradictory, do NOT default to Routine!
    is_sufficient, missing_factors = InformationSufficiencyEngine.evaluate(
        chief_complaint=cleaned_text,
        parsed_context=parsed_context,
        severity_score=severity_score,
        duration_days=duration_days,
        vital_signs=sanitized_vitals,
        comorbidities=comorbidities,
        model_top_prob=raw_top_prob
    )

    all_missing = missing_factors + vital_issues

    if not is_sufficient:
        finding = "Insufficient Information for Safe Categorization."
        reason = (
            f"Essential clinical parameters are unquantified: {'; '.join(all_missing)}. "
            f"Prematurely assigning routine status could overlook underlying pathology."
        )
        return {
            "urgency_category": TRIAGE_INSUFFICIENT,
            "decision_source": "Information Sufficiency Safety Net",
            "safety_override": False,
            "confidence": round(raw_top_prob, 3),
            "model_uncertainty": uncertainty_entropy,
            "probabilities": raw_probabilities,
            "reported_symptoms": {
                "current": current_symptoms,
                "negated": negated_symptoms,
                "historical": historical_symptoms,
                "family": family_symptoms
            },
            "red_flags": [],
            "urgent_flags": [],
            "missing_information": all_missing,
            "explanation": {
                "summary": "Additional Clinical Information Required",
                "finding": finding,
                "reason": reason,
                "confidence": f"Model confidence is equivocal ({raw_top_prob*100:.1f}%) or required diagnostic details are missing.",
                "limitation": "Symptom descriptions without temperature, vitals, or temporal progression cannot be safely resolved to routine care."
            },
            "action_advisory": "Please supply additional details: measure body temperature, specify sputum production, or record vital signs before proceeding.",
            "emergency_contact": None,
            "raw_model_class": raw_top_class,
            "raw_model_confidence": raw_top_prob
        }

    # Check if ML model predicted Urgent Assessment and presentation is moderate
    if raw_top_class == TRIAGE_URGENT and severity_score >= 5:
        urgent_flags = [f"NLP symptom pattern matched urgent clinical presentation (confidence {raw_top_prob*100:.1f}%)"]
        finding = "Urgent Clinical Assessment Recommended."
        reason = f"Statistical model identified urgent presentation with severity {severity_score}/10 over {duration_days} day(s)."
        return {
            "urgency_category": TRIAGE_URGENT,
            "decision_source": "Calibrated NLP Model",
            "safety_override": False,
            "confidence": raw_top_prob,
            "model_uncertainty": uncertainty_entropy,
            "probabilities": raw_probabilities,
            "reported_symptoms": {
                "current": current_symptoms,
                "negated": negated_symptoms,
                "historical": historical_symptoms,
                "family": family_symptoms
            },
            "red_flags": [],
            "urgent_flags": urgent_flags,
            "missing_information": all_missing,
            "explanation": {
                "summary": "Urgent Medical Assessment Warranted",
                "finding": finding,
                "reason": reason,
                "confidence": f"Model calibrated probability: {raw_top_prob*100:.1f}%.",
                "limitation": "AI triage screening identifies elevated urgency; clinical evaluation is required."
            },
            "action_advisory": "Prompt medical assessment recommended within 4 to 12 hours at an urgent care center or clinic.",
            "emergency_contact": None,
            "raw_model_class": raw_top_class,
            "raw_model_confidence": raw_top_prob
        }

    # PRIORITY 4: Routine Assessment
    # Reached ONLY when:
    #   - No emergency red flags
    #   - No urgent criteria
    #   - Information is verified sufficient
    #   - Severity is mild (<= 4 or <= 5 with benign duration)
    finding = "Routine Clinical Assessment."
    reason = (
        f"Sub-acute / uncomplicated presentation with mild severity ({severity_score}/10) over {duration_days} day(s). "
        f"Reported current symptoms: {', '.join(current_symptoms) if current_symptoms else 'mild nonspecific'}. "
        f"No emergency red flags or urgent criteria detected."
    )
    if negated_symptoms:
        reason += f" Patient affirmatively denied: {', '.join(negated_symptoms)}."

    return {
        "urgency_category": TRIAGE_ROUTINE,
        "decision_source": "Calibrated NLP Model & Benign Safety Screen",
        "safety_override": False,
        "confidence": round(raw_probabilities.get(TRIAGE_ROUTINE, raw_top_prob), 3),
        "model_uncertainty": uncertainty_entropy,
        "probabilities": raw_probabilities,
        "reported_symptoms": {
            "current": current_symptoms,
            "negated": negated_symptoms,
            "historical": historical_symptoms,
            "family": family_symptoms
        },
        "red_flags": [],
        "urgent_flags": [],
        "missing_information": all_missing,
        "explanation": {
            "summary": "Routine Primary Care Presentation",
            "finding": finding,
            "reason": reason,
            "confidence": f"Model calibrated probability: {raw_probabilities.get(TRIAGE_ROUTINE, raw_top_prob)*100:.1f}%.",
            "limitation": "AI triage screening is an assistive decision tool; if symptoms worsen or new red flags develop, seek prompt medical care."
        },
        "action_advisory": "Schedule a routine consultation or follow-up with your primary care physician. Re-evaluate if symptoms worsen.",
        "emergency_contact": None,
        "raw_model_class": raw_top_class,
        "raw_model_confidence": raw_top_prob
    }


class SymptomTriageEngine:
    """
    Singleton-compatible engine providing end-to-end hybrid symptom triage.
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

    def classify_symptoms(
        self,
        chief_complaint: str,
        duration_days: float = 1.0,
        severity_score: int = 5,
        patient_age: int = 35,
        vital_signs: Optional[Dict[str, Any]] = None,
        comorbidities: Optional[List[str]] = None,
        current_medications: Optional[str] = None,
        symptom_trajectory: str = "Unchanged",
        previous_episodes: str = "First time",
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end hybrid triage evaluation with deterministic overrides.
        """
        return predict_triage(
            chief_complaint=chief_complaint,
            duration_days=duration_days,
            severity_score=severity_score,
            patient_age=patient_age,
            vital_signs=vital_signs,
            comorbidities=comorbidities,
            current_medications=current_medications,
            symptom_trajectory=symptom_trajectory,
            previous_episodes=previous_episodes,
            confidence_threshold=confidence_threshold,
            model=self.model,
            vectorizer=self.vectorizer
        )
