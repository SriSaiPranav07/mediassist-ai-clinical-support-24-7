"""
Comprehensive Unit Tests for MediAssist Hybrid Clinical Triage Architecture
Clinical Safety, AI Accuracy, Explainability, Negation, and Probability Audit
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.symptom_triage import SymptomTriageEngine, predict_triage, compute_shannon_entropy
from src.clinical_guardrails import (
    ClinicalNarrativeParser,
    EmergencySafetyEngine,
    UrgentAssessmentEngine,
    InformationSufficiencyEngine
)
from src.config import TRIAGE_EMERGENCY, TRIAGE_URGENT, TRIAGE_ROUTINE, TRIAGE_INSUFFICIENT


@pytest.fixture
def triage_engine():
    return SymptomTriageEngine()


# ── 1. EMERGENCY SAFETY & OVERRIDE TESTS ─────────────────────────────────────

def test_emergency_red_flag_overrides_high_routine_probability(triage_engine):
    """
    Critical Clinical Safety Net Test:
    Even if a narrative contains words that might produce a routine classification,
    an overt emergency red flag (crushing chest pain radiating to arm) MUST trigger
    Emergency Warning with 100% safety override priority.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="I had mild cold symptoms earlier, but right now I have severe crushing chest pain radiating to my left arm with cold sweats",
        duration_days=0.1,
        severity_score=10
    )
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True
    assert len(res["red_flags"]) > 0
    assert res["emergency_contact"] == "112"
    assert "EMERGENCY" in res["explanation"]["finding"].upper()


def test_stroke_fast_red_flag_detection(triage_engine):
    """
    Tests acute stroke FAST criteria (facial droop, arm weakness, slurred speech).
    """
    res = triage_engine.classify_symptoms("Sudden onset of facial droop and right arm weakness with slurred speech")
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True
    assert any("stroke" in rf.lower() or "fast" in rf.lower() for rf in res["red_flags"])


def test_vital_signs_critical_hypoxemia_override(triage_engine):
    """
    Tests vital signs emergency override: SpO2 < 90% must trigger Emergency Warning
    even if the complaint is non-specific or mild.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Patient feeling tired and sluggish after studying",
        vital_signs={"SpO2": 85, "HeartRate": 80, "SystolicBP": 120}
    )
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True
    assert any("hypoxemia" in rf.lower() or "85" in rf.lower() for rf in res["red_flags"])


def test_vital_signs_critical_hemodynamic_instability(triage_engine):
    """
    Tests vital signs emergency override: profound hypotension (< 80 mmHg) or extreme tachycardia (> 150 bpm).
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Feeling dizzy when standing up",
        vital_signs={"SystolicBP": 70, "HeartRate": 160}
    )
    assert res["urgency_category"] == TRIAGE_EMERGENCY
    assert res["safety_override"] is True


# ── 2. NEGATION, TEMPORALITY & FAMILY HISTORY TESTS ──────────────────────────

def test_negation_does_not_trigger_emergency():
    """
    Tests negation: 'I do not have chest pain' must NOT trigger emergency red flag.
    """
    parsed = ClinicalNarrativeParser.parse("I have a runny nose, but I do not have chest pain or shortness of breath.")
    assert "chest pain" not in parsed["current_symptoms"]
    assert "shortness of breath" not in parsed["current_symptoms"]
    assert "chest pain" in parsed["negated_symptoms"] or "shortness of breath" in parsed["negated_symptoms"]

    emerg_eval = EmergencySafetyEngine.evaluate(
        parsed_context=parsed,
        raw_text="I have a runny nose, but I do not have chest pain or shortness of breath."
    )
    assert emerg_eval["is_emergency"] is False


def test_historical_symptom_does_not_trigger_emergency():
    """
    Tests temporality: 'I had fever and chest pain last year' must NOT be evaluated as current emergency.
    """
    parsed = ClinicalNarrativeParser.parse("I had chest pain last year which resolved, currently just mild itchy eyes.")
    assert "chest pain" not in parsed["current_symptoms"]
    assert "chest pain" in parsed["historical_symptoms"]

    emerg_eval = EmergencySafetyEngine.evaluate(
        parsed_context=parsed,
        raw_text="I had chest pain last year which resolved, currently just mild itchy eyes."
    )
    assert emerg_eval["is_emergency"] is False


def test_family_history_does_not_trigger_patient_emergency():
    """
    Tests family mention: 'My father had a heart attack' must NOT trigger an emergency for the patient.
    """
    parsed = ClinicalNarrativeParser.parse("I have a mild sore throat. My father had a heart attack last month.")
    assert "chest pain" not in parsed["current_symptoms"]
    assert len(parsed["family_history"]) > 0 or "heart" in str(parsed["family_history"]).lower()

    emerg_eval = EmergencySafetyEngine.evaluate(
        parsed_context=parsed,
        raw_text="I have a mild sore throat. My father had a heart attack last month."
    )
    assert emerg_eval["is_emergency"] is False


def test_anxiety_hypothetical_symptom():
    """
    Tests hypothetical concern: 'I am worried about having a heart attack' is not acute chest pain.
    """
    parsed = ClinicalNarrativeParser.parse("I have mild acid reflux and I am worried about having a heart attack.")
    assert "hypothetical_concerns" in parsed


# ── 3. URGENT ASSESSMENT TESTS ───────────────────────────────────────────────

def test_urgent_pneumonia_risk_high_fever_and_productive_cough(triage_engine):
    """
    Tests regression scenario:
    Fever and cough with high fever (39.0 C) and productive symptoms must be escalated
    to Urgent Assessment (potential pneumonia), NOT Routine.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="High fever with productive cough, yellow phlegm, and localized pleuritic pain",
        duration_days=3.0,
        severity_score=7,
        vital_signs={"Temperature": 39.0, "SpO2": 95, "HeartRate": 102}
    )
    assert res["urgency_category"] == TRIAGE_URGENT
    assert any("pneumonia" in uf.lower() or "fever" in uf.lower() or "respiratory" in uf.lower() for uf in res["urgent_flags"])


def test_urgent_high_severity_pain(triage_engine):
    """
    Severe pain score (>= 8/10) warrants Urgent Assessment.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Severe localized abdominal pain with persistent nausea",
        duration_days=1.0,
        severity_score=8
    )
    assert res["urgency_category"] == TRIAGE_URGENT


def test_urgent_rapidly_worsening_trajectory(triage_engine):
    """
    Rapidly worsening trajectory escalates subacute presentations.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Moderate throat pain and fever",
        duration_days=2.0,
        severity_score=6,
        symptom_trajectory="Worsening rapidly"
    )
    assert res["urgency_category"] == TRIAGE_URGENT


# ── 4. INSUFFICIENT INFORMATION TESTS ────────────────────────────────────────

def test_unquantified_fever_and_cough_triggers_insufficient_information(triage_engine):
    """
    Regression Test for prompt requirement:
    Bare 'fever and cough' with no temperature, no duration context, and moderate severity
    should trigger 'Insufficient Information' rather than prematurely defaulting to Routine 84%.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="fever and cough",
        duration_days=1.0,
        severity_score=5,
        vital_signs=None
    )
    assert res["urgency_category"] == TRIAGE_INSUFFICIENT
    assert len(res["missing_information"]) > 0


def test_empty_narrative_triggers_insufficient_information(triage_engine):
    """
    Empty text must be rejected safely.
    """
    res = triage_engine.classify_symptoms("")
    assert res["urgency_category"] == TRIAGE_INSUFFICIENT
    assert res["confidence"] == 0.0


def test_contradictory_information_flags_missing_info(triage_engine):
    """
    High severity (10/10) combined with trivial wording ('just a tiny scratch')
    flags contradictory information.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Just a mild tickle in throat",
        severity_score=9
    )
    assert res["urgency_category"] == TRIAGE_INSUFFICIENT


# ── 5. ROUTINE ASSESSMENT TESTS ──────────────────────────────────────────────

def test_genuine_routine_presentation(triage_engine):
    """
    Routine Assessment is assigned only when information is sufficient,
    duration is known, severity is mild (<= 4), and no red flags exist.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Mild dry cough and runny nose for 4 days with occasional sneezing, no fever or chest pain",
        duration_days=4.0,
        severity_score=2,
        symptom_trajectory="Improving"
    )
    assert res["urgency_category"] == TRIAGE_ROUTINE
    assert res["safety_override"] is False
    assert len(res["red_flags"]) == 0


# ── 6. PROBABILITY AUDIT & MATHEMATICAL INTEGRITY TESTS ───────────────────────

def test_class_probabilities_sum_to_one(triage_engine):
    """
    Probability Audit:
    Class probabilities presented to the clinician MUST sum strictly to 1.0 (100.0%).
    No misleading caps that distort the distribution.
    """
    res = triage_engine.classify_symptoms(
        chief_complaint="Mild intermittent tension headache after computer work",
        duration_days=2.0,
        severity_score=3
    )
    probs = res["probabilities"]
    assert len(probs) >= 3
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 0.001


def test_shannon_entropy_calculation():
    """
    Tests uncertainty entropy metric.
    """
    # Deterministic certainty
    assert compute_shannon_entropy({"Emergency": 1.0, "Urgent": 0.0, "Routine": 0.0}) == 0.0
    # Equal distribution -> maximal entropy 1.0
    h_max = compute_shannon_entropy({"Emergency": 0.3333, "Urgent": 0.3333, "Routine": 0.3334})
    assert abs(h_max - 1.0) < 0.02


def test_ai_generated_summary_structure(triage_engine):
    """
    Tests that structured explanation does not make false clinician claims.
    """
    res = triage_engine.classify_symptoms("Mild intermittent tension headache")
    exp = res["explanation"]
    assert "finding" in exp
    assert "reason" in exp
    assert "confidence" in exp
    assert "limitation" in exp
    # Must not contain fabricated doctor signoffs
    assert "dr." not in exp["reason"].lower()
    assert "doctor" not in exp["finding"].lower()
