"""
Unit Tests for Symptom Triage Classifier & Safety Overrides
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.symptom_triage import SymptomTriageEngine, predict_triage
from src.config import TRIAGE_EMERGENCY, TRIAGE_URGENT, TRIAGE_ROUTINE, TRIAGE_INSUFFICIENT

@pytest.fixture
def triage_engine():
    return SymptomTriageEngine()

def test_emergency_red_flag_detection(triage_engine):
    # Severe chest pain
    res = triage_engine.classify_symptoms("Patient has crushing chest pain radiating to left arm and sweating")
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True
    assert len(res["red_flags"]) > 0
    assert res["emergency_contact"] == "112"

def test_stroke_red_flag(triage_engine):
    # Stroke symptoms
    res = triage_engine.classify_symptoms("Sudden onset of facial droop and right arm weakness")
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True

def test_vital_sign_emergency_override(triage_engine):
    # Severe hypoxemia
    res = triage_engine.classify_symptoms(
        chief_complaint="Patient feeling tired and sluggish",
        vital_signs={"SpO2": 85, "HeartRate": 80}
    )
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True

def test_routine_presentation(triage_engine):
    res = triage_engine.classify_symptoms(
        chief_complaint="Mild intermittent runny nose and sneezing for 3 days with no breathing issues",
        duration_days=3.0,
        severity_score=2
    )
    assert res["urgency_category"] == TRIAGE_ROUTINE
    assert res["safety_override"] is False

def test_insufficient_information(triage_engine):
    res = triage_engine.classify_symptoms("")
    assert res["urgency_category"] == TRIAGE_INSUFFICIENT
    assert res["confidence"] == 0.0

def test_10_second_explanation_structure(triage_engine):
    res = triage_engine.classify_symptoms("Crushing chest pain and severe shortness of breath")
    exp = res["explanation"]
    assert "finding" in exp
    assert "reason" in exp
    assert "confidence" in exp
    assert "limitation" in exp
    assert len(exp["finding"]) > 0

def test_clinical_guardrail_rule(triage_engine):
    """
    Rule: If severity <= 5, duration is under 3 days, and no red-flag keywords
    are present, cap Urgent and Emergency probabilities and default to Routine Assessment.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Feeling a little tired after working on a laptop for several hours with mild eye strain",
        duration_days=1.5,
        severity_score=3
    )
    assert res["urgency_category"] == TRIAGE_ROUTINE
    assert res["safety_override"] is False
    assert res["probabilities"][TRIAGE_EMERGENCY] <= 0.05
    assert res["probabilities"][TRIAGE_URGENT] <= 0.20
    assert res["guardrail_applied"] is True

def test_red_flag_penalty_when_absent(triage_engine):
    """
    When no red flag keywords are present, Emergency Warning probability
    is heavily penalized.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Mild dry skin on elbows and knees with slight itching",
        duration_days=5.0,
        severity_score=2
    )
    assert res["probabilities"].get(TRIAGE_EMERGENCY, 0.0) < 0.10

def test_low_confidence_trigger_requires_more_information(triage_engine):
    """
    When calibrated confidence is below threshold (e.g. 0.85 on vague text),
    triggers 'Requires More Information'.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Patient presents with general malaise and vague nonspecific discomfort",
        duration_days=6.0,
        severity_score=6,
        confidence_threshold=0.85
    )
    assert res["urgency_category"] == "Requires More Information"
    assert res["is_low_confidence"] is True
