"""
Unit Tests for Symptom Triage Classifier & Safety Overrides
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.symptom_triage import SymptomTriageEngine
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
