"""
Unit Tests for Data Validation & Safety Sanitation
"""
import pytest
from PIL import Image
import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_validation import (
    validate_symptom_text,
    validate_patient_demographics,
    validate_vital_signs,
    validate_lab_input,
    validate_image_file
)
from src.config import LAB_REFERENCE_RANGES

def test_validate_symptom_text():
    # Valid text
    valid, text, err = validate_symptom_text("Experiencing mild headache for 2 days.")
    assert valid is True
    assert err == ""

    # Empty text
    valid, text, err = validate_symptom_text("")
    assert valid is False
    assert "empty" in err.lower()

    # Too short text
    valid, text, err = validate_symptom_text("ok")
    assert valid is False
    assert "short" in err.lower()

    # Non-alphabetic text
    valid, text, err = validate_symptom_text("1234567890 !@#$")
    assert valid is False
    assert "words" in err.lower()

def test_validate_patient_demographics():
    # Valid
    assert validate_patient_demographics(45, "Male")[0] is True
    assert validate_patient_demographics(0, "Female")[0] is True
    assert validate_patient_demographics("72", "Other")[0] is True

    # Out of range age
    assert validate_patient_demographics(-5, "Male")[0] is False
    assert validate_patient_demographics(130, "Female")[0] is False

    # Invalid sex
    assert validate_patient_demographics(30, "InvalidSex")[0] is False

def test_validate_vital_signs():
    # Valid vitals
    vitals = {
        "HeartRate": 72,
        "SpO2": 98,
        "SystolicBP": 120,
        "DiastolicBP": 80,
        "RespiratoryRate": 16,
        "Temperature": 37.0
    }
    valid, issues = validate_vital_signs(vitals)
    assert valid is True
    assert len(issues) == 0

    # Physiologically implausible
    bad_vitals = {
        "HeartRate": 500,  # Impossible
        "SpO2": 15,        # Implausible
        "SystolicBP": 80,
        "DiastolicBP": 95   # Diastolic > Systolic
    }
    valid, issues = validate_vital_signs(bad_vitals)
    assert valid is False
    assert len(issues) >= 2

def test_validate_lab_input():
    # Valid
    valid, msg = validate_lab_input("Hemoglobin", 14.5, "g/dL", LAB_REFERENCE_RANGES)
    assert valid is True

    # Unknown test
    valid, msg = validate_lab_input("UnknownEnzymeX", 10.0, "U/L", LAB_REFERENCE_RANGES)
    assert valid is False

    # Unit mismatch
    valid, msg = validate_lab_input("Hemoglobin", 14.5, "mmol/L", LAB_REFERENCE_RANGES)
    assert valid is False
    assert "mismatch" in msg.lower()

    # Negative value
    valid, msg = validate_lab_input("Platelets", -50, "10^3/uL", LAB_REFERENCE_RANGES)
    assert valid is False
    assert "negative" in msg.lower()

def test_validate_image_file():
    # Create valid synthetic image in memory
    img = Image.new("RGB", (128, 128), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    valid, pil_out, err = validate_image_file(valid_bytes)
    assert valid is True
    assert pil_out is not None

    # Empty bytes
    valid, pil_out, err = validate_image_file(b"")
    assert valid is False

    # Too small image
    small_img = Image.new("RGB", (32, 32))
    buf_small = io.BytesIO()
    small_img.save(buf_small, format="PNG")
    valid, pil_out, err = validate_image_file(buf_small.getvalue())
    assert valid is False
    assert "small" in err.lower()
