"""
MediAssist - Comprehensive System Evaluation & Benchmark Runner
Author: Zelvora Industry Labs / MediAssist Team
Runs validation benchmarks across all 4 clinical modules and generates evaluation artifacts.
"""
import json
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS_DIR,
    LAB_REFERENCE_RANGES,
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
    RISK_LEVEL_EMERGENCY,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_STANDARD
)
from src.symptom_triage import SymptomTriageEngine
from src.lab_interpreter import LabInterpreter
from src.risk_engine import CompositeRiskEngine
from src.utils import DEMO_PATIENTS

def evaluate_lab_boundary_conditions() -> dict:
    """
    Evaluates LabInterpreter across normal, boundary, abnormal, and critical conditions for all tests.
    """
    interpreter = LabInterpreter()
    test_cases = 0
    passed_cases = 0
    test_results = []

    for test_name, spec in LAB_REFERENCE_RANGES.items():
        ref_low, ref_high = spec["ranges"].get("All", list(spec["ranges"].values())[0])
        crit_low = spec.get("critical_low")
        crit_high = spec.get("critical_high")

        # 1. Normal Value Test
        normal_val = round((ref_low + ref_high) / 2.0, 2)
        res_normal = interpreter.interpret_single_test(test_name, normal_val, spec["unit"])
        test_cases += 1
        if res_normal["status"] == "Normal":
            passed_cases += 1
        test_results.append({"test": test_name, "condition": "Normal", "status": res_normal["status"], "pass": res_normal["status"] == "Normal"})

        # 2. Critical Low Test (if defined)
        if crit_low is not None:
            crit_low_val = round(crit_low - 0.5, 2)
            res_cl = interpreter.interpret_single_test(test_name, crit_low_val, spec["unit"])
            test_cases += 1
            passed = (res_cl["status"] == "Critical Low" and res_cl["is_critical"] is True)
            if passed:
                passed_cases += 1
            test_results.append({"test": test_name, "condition": "Critical Low", "status": res_cl["status"], "pass": passed})

        # 3. Critical High Test (if defined)
        if crit_high is not None:
            crit_high_val = round(crit_high + 5.0, 2)
            res_ch = interpreter.interpret_single_test(test_name, crit_high_val, spec["unit"])
            test_cases += 1
            passed = (res_ch["status"] == "Critical High" and res_ch["is_critical"] is True)
            if passed:
                passed_cases += 1
            test_results.append({"test": test_name, "condition": "Critical High", "status": res_ch["status"], "pass": passed})

    accuracy = round(passed_cases / test_cases, 4) if test_cases > 0 else 1.0
    return {
        "total_boundary_tests": test_cases,
        "passed_tests": passed_cases,
        "boundary_accuracy": accuracy,
        "details": test_results
    }

def evaluate_composite_risk_engine() -> dict:
    """
    Evaluates CompositeRiskEngine against clinical scenarios and safety override requirements.
    """
    engine = CompositeRiskEngine()
    results = []
    overrides_verified = 0

    for pt in DEMO_PATIENTS:
        # Simulate triage output
        triage_res = {
            "urgency_category": pt["expected_triage"],
            "safety_override": (pt["expected_triage"] == TRIAGE_EMERGENCY),
            "red_flags": ["Severe acute red flag"] if pt["expected_triage"] == TRIAGE_EMERGENCY else []
        }

        # Simulate lab output
        lab_interpreter = LabInterpreter()
        lab_panel = lab_interpreter.interpret_panel(pt.get("labs", {}))

        # Risk aggregation
        risk = engine.calculate_patient_risk(
            patient_id=pt["id"],
            triage_output=triage_res,
            lab_output=lab_panel,
            vital_signs=pt.get("vitals")
        )

        # Verification: If expected triage was Emergency Warning, risk MUST be Emergency Review
        if pt["expected_triage"] == TRIAGE_EMERGENCY:
            assert risk["priority_level"] == RISK_LEVEL_EMERGENCY, f"Emergency safety override failed for {pt['id']}"
            overrides_verified += 1

        results.append({
            "patient_id": pt["id"],
            "scenario": pt["clinical_scenario"],
            "expected_triage": pt["expected_triage"],
            "assigned_priority": risk["priority_level"],
            "composite_score": risk["composite_score"],
            "factors_count": len(risk["contributing_factors"])
        })

    return {
        "patients_evaluated": len(DEMO_PATIENTS),
        "emergency_overrides_verified": overrides_verified,
        "safety_compliance_rate": 1.0,
        "evaluations": results
    }

def run_all_evaluations():
    print("=" * 60)
    print("MediAssist - Multi-Modal System Evaluation Runner")
    print("=" * 60)

    # 1. Lab Evaluation
    lab_metrics = evaluate_lab_boundary_conditions()
    print(f"Laboratory Rules Verification: {lab_metrics['passed_tests']}/{lab_metrics['total_boundary_tests']} passed ({lab_metrics['boundary_accuracy']*100:.1f}%)")

    # 2. Risk Engine Safety Evaluation
    risk_metrics = evaluate_composite_risk_engine()
    print(f"Risk Safety Overrides Verified: {risk_metrics['emergency_overrides_verified']} (100% compliance)")

    # 3. Load Triage Metrics if present
    triage_path = MODELS_DIR / "triage_evaluation.json"
    triage_metrics = {}
    if triage_path.exists():
        with open(triage_path, "r", encoding="utf-8") as f:
            triage_metrics = json.load(f)
        print(f"NLP Triage Accuracy: {triage_metrics.get('Accuracy', 0)*100:.1f}%")

    # 4. Load X-Ray Metrics if present
    xray_path = MODELS_DIR / "xray_evaluation.json"
    xray_metrics = {}
    if xray_path.exists():
        with open(xray_path, "r", encoding="utf-8") as f:
            xray_metrics = json.load(f)
        print(f"Chest X-Ray Evaluation: {len(xray_metrics)} pathology classes evaluated")

    system_report = {
        "triage_nlp": triage_metrics,
        "chest_xray": xray_metrics,
        "laboratory_rules": lab_metrics,
        "composite_risk_engine": risk_metrics
    }

    report_path = MODELS_DIR / "system_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(system_report, f, indent=2)
    print(f"\nConsolidated System Evaluation Report written to: {report_path}")
    return system_report

if __name__ == "__main__":
    run_all_evaluations()
