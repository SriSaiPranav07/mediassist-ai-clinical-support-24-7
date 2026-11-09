"""
MediAssist - Symptom Triage Model & Safety Screening
Author: Zelvora Industry Labs / MediAssist Team

Provides the modular SymptomTriageModel with:
  - Red flag keyword dictionary & regex patterns
  - Feature engineering & numerical feature weighting
  - Clinical heuristic guardrail (capping urgent/emergency for mild symptoms)
  - Calibrated probability thresholding (triggers 'Requires More Information')
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib

from src.config import (
    TRIAGE_MODEL_PATH,
    TRIAGE_VECTORIZER_PATH
)
from src.symptom_triage import (
    predict_triage,
    detect_red_flag_keywords,
    extract_clinical_keywords
)


class SymptomTriageModel:
    """
    Two-stage Triage Classifier:
    1. Deterministic Safety Override Layer (Red-Flag check)
    2. Calibrated NLP Classifier (TF-IDF + Calibrated Logistic Regression)
    3. Clinical Heuristics & Guardrail Net
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

    def predict(
        self,
        chief_complaint: str,
        duration_days: float = 1.0,
        severity_score: int = 5,
        patient_age: int = 35,
        vital_signs: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end triage prediction with feature weighting,
        rule-based guardrails, and calibration thresholding.
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

    def classify_symptoms(
        self,
        chief_complaint: str,
        duration_days: float = 1.0,
        severity_score: int = 5,
        patient_age: int = 35,
        vital_signs: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Alias for compatibility with earlier page callers."""
        return self.predict(
            chief_complaint=chief_complaint,
            duration_days=duration_days,
            severity_score=severity_score,
            patient_age=patient_age,
            vital_signs=vital_signs,
            confidence_threshold=confidence_threshold
        )
