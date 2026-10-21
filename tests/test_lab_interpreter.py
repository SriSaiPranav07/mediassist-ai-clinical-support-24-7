"""
Unit Tests for Laboratory Result Interpreter & Critical Value Flags
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.lab_interpreter import LabInterpreter

@pytest.fixture
def lab_interpreter():
    return LabInterpreter()

def test_normal_potassium(lab_interpreter):
    res = lab_interpreter.interpret_single_test("Potassium", 4.2, "mEq/L")
    assert res["status"] == "Normal"
    assert res["is_critical"] is False
    assert res["severity"] == "Normal"

def test_critical_high_potassium(lab_interpreter):
    # Potassium > 6.2 is critical hyperkalemia
    res = lab_interpreter.interpret_single_test("Potassium", 6.8, "mEq/L")
    assert res["status"] == "Critical High"
    assert res["is_critical"] is True
    assert "hyperkalemia" in res["clinical_implication"].lower()

def test_critical_low_hemoglobin(lab_interpreter):
    # Hemoglobin < 7.0 is critical anemia
    res = lab_interpreter.interpret_single_test("Hemoglobin", 6.4, "g/dL", patient_sex="Female")
    assert res["status"] == "Critical Low"
    assert res["is_critical"] is True
    assert "hypoxia" in res["clinical_implication"].lower() or "anemia" in res["clinical_implication"].lower()

def test_sex_specific_reference_interval(lab_interpreter):
    # Hemoglobin 12.5 is normal for Female (12.1-15.1), but Low for Male (13.8-17.2)
    res_female = lab_interpreter.interpret_single_test("Hemoglobin", 12.5, "g/dL", patient_sex="Female")
    res_male = lab_interpreter.interpret_single_test("Hemoglobin", 12.5, "g/dL", patient_sex="Male")
    assert res_female["status"] == "Normal"
    assert res_male["status"] == "Low"

def test_glucose_fasting_vs_random(lab_interpreter):
    # Glucose 120 is normal Random (< 140), but High Fasting (> 99)
    res_random = lab_interpreter.interpret_single_test("Glucose", 120.0, "mg/dL", fasting=False)
    res_fasting = lab_interpreter.interpret_single_test("Glucose", 120.0, "mg/dL", fasting=True)
    assert res_random["status"] == "Normal"
    assert res_fasting["status"] == "High"

def test_panel_interpretation(lab_interpreter):
    panel = {
        "Potassium": {"value": 7.0, "unit": "mEq/L"},   # Critical High
        "Hemoglobin": {"value": 14.0, "unit": "g/dL"},   # Normal
        "Glucose": {"value": 160.0, "unit": "mg/dL"}    # High
    }
    summary = lab_interpreter.interpret_panel(panel)
    assert summary["total_tests"] == 3
    assert summary["critical_count"] == 1
    assert summary["abnormal_count"] == 1
    assert summary["normal_count"] == 1
    assert summary["has_critical"] is True
