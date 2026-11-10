"""
MediAssist - Symptom Triage Model & Safety Screening
Author: MediAssist Clinical AI Team

Provides the modular SymptomTriageModel wrapping the Hybrid Clinical Triage Architecture.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib

from src.config import (
    TRIAGE_MODEL_PATH,
    TRIAGE_VECTORIZER_PATH
)
from src.symptom_triage import predict_triage


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

    def predict(
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
        Executes end-to-end hybrid triage prediction with clinical safety overrides.
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
        """Alias for compatibility with earlier page callers."""
        return self.predict(
            chief_complaint=chief_complaint,
            duration_days=duration_days,
            severity_score=severity_score,
            patient_age=patient_age,
            vital_signs=vital_signs,
            comorbidities=comorbidities,
            current_medications=current_medications,
            symptom_trajectory=symptom_trajectory,
            previous_episodes=previous_episodes,
            confidence_threshold=confidence_threshold
        )
