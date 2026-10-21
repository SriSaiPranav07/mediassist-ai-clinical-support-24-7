"""
MediAssist - NLP-Based Symptom Triage Classifier & Safety Override Engine
Author: Zelvora Industry Labs / MediAssist Team
"""
from typing import Dict, Any, List, Optional
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

class SymptomTriageEngine:
    """
    Two-stage Triage Classifier:
    1. Deterministic Clinical Safety Layer (Red-Flag override)
    2. Calibrated NLP Machine Learning Classifier (TF-IDF + Calibrated Model)
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
        """
        Conservative Safety Layer: Scan text for overt life-threatening symptoms.
        """
        detected = []
        text_lower = text.lower()
        for pattern in RED_FLAG_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                detected.append(match.group(0))
        return list(set(detected))

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract recognizable clinical symptoms and key phrases.
        """
        clinical_keywords = [
            "fever", "cough", "shortness of breath", "dyspnea", "chest pain",
            "headache", "dizziness", "nausea", "vomiting", "diarrhea",
            "abdominal pain", "fatigue", "palpitations", "sore throat",
            "rash", "joint pain", "swelling", "edema", "hemoptysis",
            "weakness", "numbness", "sweating", "diaphoresis", "chills"
        ]
        text_lower = text.lower()
        found = [kw for kw in clinical_keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)]
        return found

    def classify_symptoms(
        self,
        chief_complaint: str,
        duration_days: float = 1.0,
        severity_score: int = 5,
        patient_age: int = 35,
        vital_signs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end triage analysis with safety overrides and 10-second explanation.
        """
        # Step 1: Input validation
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
                "emergency_contact": None
            }

        # Step 2: Deterministic Safety Check (Emergency Warning Signs)
        red_flags = self.detect_red_flags(cleaned_text)
        symptoms_found = self.extract_keywords(cleaned_text)

        # Also check severe vital sign red flags if supplied
        vital_red_flags = []
        if vital_signs:
            spo2 = vital_signs.get("SpO2")
            if spo2 is not None and float(spo2) < 90.0:
                vital_red_flags.append(f"Severe hypoxemia (SpO2 {spo2}% < 90%)")
            hr = vital_signs.get("HeartRate")
            if hr is not None and (float(hr) > 150.0 or float(hr) < 40.0):
                vital_red_flags.append(f"Critical heart rate ({hr} bpm)")
            sys_bp = vital_signs.get("SystolicBP")
            if sys_bp is not None and float(sys_bp) < 80.0:
                vital_red_flags.append(f"Profound hypotension (Systolic BP {sys_bp} mmHg)")

        all_emergency_reasons = red_flags + vital_red_flags

        if len(all_emergency_reasons) > 0:
            # Absolute Safety Override
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
                "emergency_contact": EMERGENCY_CONTACT_INDIA
            }

        # Step 3: Statistical NLP Inference
        if self.model is not None and self.vectorizer is not None:
            try:
                features = self.vectorizer.transform([cleaned_text])
                probs = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
                top_class = classes[probs.argmax()]
                top_prob = float(probs.max())

                # Context-aware adjustments based on severity and duration
                if severity_score >= 8 and top_class == TRIAGE_ROUTINE:
                    top_class = TRIAGE_URGENT
                    prob_dict[TRIAGE_URGENT] = max(prob_dict.get(TRIAGE_URGENT, 0.5), 0.70)
                    top_prob = prob_dict[TRIAGE_URGENT]

                # Formulate 10-second explanation
                plain_reason = (
                    f"Reported symptoms include '{', '.join(symptoms_found)}' "
                    f"with patient-reported severity {severity_score}/10 over {duration_days} day(s)."
                    if symptoms_found else
                    f"Reported text matched clinical urgency pattern with severity {severity_score}/10."
                )

                return {
                    "urgency_category": top_class,
                    "confidence": round(top_prob, 3),
                    "identified_symptoms": symptoms_found,
                    "red_flags": [],
                    "probabilities": prob_dict,
                    "explanation": {
                        "finding": f"Preliminary classification: {top_class}.",
                        "reason": plain_reason,
                        "confidence": f"Model confidence is {top_prob*100:.1f}%.",
                        "limitation": "NLP triage models cannot replace clinician history-taking and physical examination."
                    },
                    "safety_override": False,
                    "action_advisory": (
                        "Prompt medical assessment recommended within 4-12 hours." if top_class == TRIAGE_URGENT
                        else "Schedule routine consultation or follow-up with primary care physician."
                    ),
                    "emergency_contact": None
                }
            except Exception as e:
                pass  # Fall through to heuristic fallback

        # Step 4: Transparent Rule-based Fallback when model is not yet loaded
        if severity_score >= 7 or duration_days > 14:
            predicted = TRIAGE_URGENT
            conf = 0.75
            reason = f"Elevated symptom severity ({severity_score}/10) or prolonged duration ({duration_days} days)."
        else:
            predicted = TRIAGE_ROUTINE
            conf = 0.70
            reason = f"Sub-acute presentation with moderate severity ({severity_score}/10)."

        return {
            "urgency_category": predicted,
            "confidence": conf,
            "identified_symptoms": symptoms_found,
            "red_flags": [],
            "probabilities": {TRIAGE_URGENT: 0.5, TRIAGE_ROUTINE: 0.5},
            "explanation": {
                "finding": f"Baseline heuristic assessment: {predicted}.",
                "reason": reason,
                "confidence": "Heuristic clinical rule baseline (model checkpoint pending).",
                "limitation": "Educational triage rule; requires medical evaluation by a licensed practitioner."
            },
            "safety_override": False,
            "action_advisory": "Follow local medical guidelines for clinical evaluation.",
            "emergency_contact": None
        }
