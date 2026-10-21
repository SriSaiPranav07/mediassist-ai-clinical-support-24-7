"""
Unit Tests for Composite Patient Risk Aggregation & Safety Guarantees
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.risk_engine import CompositeRiskEngine
from src.config import (
    RISK_LEVEL_EMERGENCY,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_STANDARD,
    RISK_LEVEL_INSUFFICIENT,
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE
)

@pytest.fixture
def risk_engine():
    return CompositeRiskEngine()

def test_emergency_override_never_downgraded(risk_engine):
    """
    Critical Safety Requirement: An explicit emergency warning must NEVER be downgraded
    by averaging it with low-risk model outputs!
    """
    # Emergency triage with completely normal labs and normal X-ray
    triage_out = {
        "urgency_category": TRIAGE_EMERGENCY,
        "safety_override": True,
        "red_flags": ["Crushing chest pain radiating to left arm"]
    }
    lab_out = {
        "total_tests": 4,
        "critical_count": 0,
        "abnormal_count": 0,
        "normal_count": 4,
        "tests": []
    }
    xray_out = {
        "success": True,
        "flagged_findings": [],
        "probabilities": {"Normal": 0.95}
    }

    result = risk_engine.calculate_patient_risk(
        patient_id="TEST-EMERGENCY-01",
        triage_output=triage_out,
        lab_output=lab_out,
        xray_output=xray_out
    )

    # Must be EMERGENCY REVIEW despite normal labs and normal X-ray
    assert result["priority_level"] == RISK_LEVEL_EMERGENCY
    assert result["composite_score"] >= 85.0
    assert len(result["emergency_reasons"]) > 0

def test_lab_critical_triggers_emergency_review(risk_engine):
    # Critical hyperkalemia triggers emergency review even if patient reports mild symptoms
    triage_out = {"urgency_category": TRIAGE_ROUTINE, "safety_override": False}
    lab_out = {
        "total_tests": 1,
        "critical_count": 1,
        "abnormal_count": 0,
        "normal_count": 0,
        "tests": [{"test_name": "Potassium", "value": 7.2, "unit": "mEq/L", "status": "Critical High", "is_critical": True}]
    }
    result = risk_engine.calculate_patient_risk(
        patient_id="TEST-CRIT-LAB",
        triage_output=triage_out,
        lab_output=lab_out
    )
    assert result["priority_level"] == RISK_LEVEL_EMERGENCY

def test_standard_review(risk_engine):
    triage_out = {"urgency_category": TRIAGE_ROUTINE, "safety_override": False}
    lab_out = {"total_tests": 2, "critical_count": 0, "abnormal_count": 0, "normal_count": 2, "tests": []}
    result = risk_engine.calculate_patient_risk(
        patient_id="TEST-STANDARD",
        triage_output=triage_out,
        lab_output=lab_out
    )
    assert result["priority_level"] == RISK_LEVEL_STANDARD
    assert result["composite_score"] < 45.0

def test_insufficient_information_handling(risk_engine):
    result = risk_engine.calculate_patient_risk(patient_id="TEST-EMPTY")
    assert result["priority_level"] == RISK_LEVEL_INSUFFICIENT
    assert result["composite_score"] == 0.0
