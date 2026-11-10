"""
Comprehensive Unit Tests for Multimodal Chest X-Ray & Symptom CDSS
Author: MediAssist Clinical AI Team

Covers:
  - Patient Input & Vital Sign Plausibility
  - Radiograph Validation (PNG, JPEG, Blank, Tiny, Corrupted, OOD Color Photo)
  - Model Inference, Multi-Label Outputs & Supported Classes
  - Multimodal Integration (Concordance, Discordance, Safety Net Overrides)
  - PDF Consultation Report Generation
"""
import pytest
import io
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.xray_model import ChestXRayScreener
from src.utils import generate_anatomical_xray
from src.config import XRAY_CLASSES
from src.multimodal_integration import MultimodalClinicalIntegrator, audit_multimodal_discordance
from src.clinical_guardrails import ClinicalNarrativeParser
from src.data_validation import validate_vital_signs
from utils.data_processing import validate_radiograph_image
from utils.pdf_generator import generate_multimodal_xray_pdf


@pytest.fixture
def xray_screener():
    return ChestXRayScreener()


# ── 1. IMAGE UPLOAD & PREPROCESSING VALIDATION TESTS ─────────────────────────

def test_valid_radiograph_validation():
    """Valid anatomical monochrome radiograph passes validation."""
    img = generate_anatomical_xray("normal", size=(256, 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    is_valid, err, meta = validate_radiograph_image(buf.getvalue())
    assert is_valid is True
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["chromatic_divergence"] < 35.0


def test_corrupted_image_rejection():
    """Corrupted byte payload must be rejected gracefully."""
    is_valid, err, meta = validate_radiograph_image(b"CORRUPTED_NON_IMAGE_DATA_12345")
    assert is_valid is False
    assert "Corrupted" in err or "invalid" in err.lower()


def test_blank_solid_image_rejection():
    """Uniformly white or black solid image must be rejected for lack of contrast."""
    solid = Image.new("L", (200, 200), color=255)
    buf = io.BytesIO()
    solid.save(buf, format="PNG")
    is_valid, err, meta = validate_radiograph_image(buf.getvalue())
    assert is_valid is False
    assert "blank" in err.lower() or "contrast" in err.lower()


def test_very_small_image_rejection():
    """Tiny image (< 64x64) must be rejected."""
    tiny = Image.new("L", (32, 32), color=128)
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    is_valid, err, meta = validate_radiograph_image(buf.getvalue())
    assert is_valid is False
    assert "small" in err.lower()


def test_ood_chromatic_color_photo_rejection():
    """Natural color photograph with high color divergence must be rejected as Out-of-Distribution."""
    color_photo = Image.new("RGB", (200, 200), color=(255, 20, 147))  # Deep pink
    buf = io.BytesIO()
    color_photo.save(buf, format="JPEG")
    is_valid, err, meta = validate_radiograph_image(buf.getvalue())
    assert is_valid is False
    assert "out-of-distribution" in err.lower() or "chromatic" in err.lower()


# ── 2. MODEL INFERENCE & OUTPUT TESTS ────────────────────────────────────────

def test_model_inference_pipeline_and_classes(xray_screener):
    """Verifies output shape, probability keys, and classes."""
    img = generate_anatomical_xray("cardiomegaly")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = xray_screener.analyze_radiograph(buf.getvalue(), threshold=0.40)

    assert res["success"] is True
    assert "probabilities" in res
    assert len(res["probabilities"]) == len(XRAY_CLASSES)
    for cls_name in XRAY_CLASSES:
        assert cls_name in res["probabilities"]
        assert 0.0 <= res["probabilities"][cls_name] <= 1.0


def test_gradcam_overlay_generation(xray_screener):
    """Verifies Grad-CAM heatmap overlay generation."""
    img = generate_anatomical_xray("infiltration")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = xray_screener.analyze_radiograph(buf.getvalue(), threshold=0.30)
    if xray_screener.model is not None:
        assert res["gradcam_available"] is True
        assert res["gradcam_image"] is not None


# ── 3. PATIENT INPUT & VITAL SIGN TESTS ──────────────────────────────────────

def test_vital_signs_plausibility():
    """Tests normal vs non-physiological vitals."""
    # Plausible
    ok, errs = validate_vital_signs({"HeartRate": 72, "SpO2": 98, "SystolicBP": 120, "DiastolicBP": 80})
    assert ok is True
    assert len(errs) == 0

    # Non-physiological: Diastolic >= Systolic
    ok, errs = validate_vital_signs({"SystolicBP": 110, "DiastolicBP": 130})
    assert ok is False
    assert any("Diastolic BP" in e for e in errs)

    # Extreme out of bounds
    ok, errs = validate_vital_signs({"HeartRate": 400})
    assert ok is False
    assert any("Heart Rate" in e for e in errs)


# ── 4. MULTIMODAL CONCORDANCE & DISCORDANCE TESTS ────────────────────────────

def test_multimodal_concordance_pneumonia():
    """
    Fever + Productive Cough alongside Infiltration on X-ray
    triggers High Concordance for Lower Respiratory Infection / Pneumonia.
    """
    patient_info = {"age": 42, "sex": "Female", "medical_history": ["Asthma / COPD"]}
    symptom_data = {
        "narrative": "High fever with productive yellow cough for 4 days",
        "current_symptoms": ["fever", "cough"],
        "negated_symptoms": ["chest pain"],
        "duration_days": 4.0,
        "severity": 7
    }
    xray_result = {
        "flagged_findings": ["Infiltration"],
        "probabilities": {"Infiltration": 0.72, "Normal": 0.15},
        "primary_finding": "Infiltration Detected",
        "top_class": "Infiltration"
    }

    report = MultimodalClinicalIntegrator.synthesize_report(
        patient_info=patient_info,
        symptom_data=symptom_data,
        vital_signs={"Temperature": 38.9, "SpO2": 92},
        xray_result=xray_result,
        projection_view="PA view"
    )

    assert report["decision_status"] in ["PROMPT CLINICAL ASSESSMENT RECOMMENDED", "EMERGENCY WARNING"]
    assert len(report["concordant_insights"]) > 0
    assert any("Pneumonia" in c["condition"] for c in report["concordant_insights"])


def test_multimodal_discordance_severe_dyspnea_normal_xray():
    """
    Severe breathlessness with a Normal X-ray must trigger Critical Discordance
    (warning about Pulmonary Embolism, bronchospasm, or metabolic acidosis).
    """
    patient_info = {"age": 55, "sex": "Male"}
    symptom_data = {
        "narrative": "Sudden onset of severe shortness of breath",
        "current_symptoms": ["shortness of breath"],
        "negated_symptoms": [],
        "duration_days": 0.5,
        "severity": 8
    }
    xray_result = {
        "flagged_findings": ["Normal"],
        "probabilities": {"Normal": 0.85},
        "primary_finding": "No overt acute abnormality",
        "top_class": "Normal"
    }

    report = MultimodalClinicalIntegrator.synthesize_report(
        patient_info=patient_info,
        symptom_data=symptom_data,
        vital_signs={"SpO2": 93},
        xray_result=xray_result,
        projection_view="PA view"
    )

    assert len(report["discordant_insights"]) > 0
    assert any("Dyspnea with Normal Radiograph" in d["clinical_concern"] for d in report["discordant_insights"])


def test_multimodal_discordance_chest_pain_normal_xray():
    """
    Chest pain with a Normal X-ray must trigger Critical Discordance
    (warning that normal X-ray does not rule out ACS / Heart Attack).
    """
    patient_info = {"age": 58, "sex": "Male"}
    symptom_data = {
        "narrative": "Chest pain for 2 hours",
        "current_symptoms": ["chest pain"],
        "negated_symptoms": [],
        "duration_days": 0.1,
        "severity": 7
    }
    xray_result = {
        "flagged_findings": ["Normal"],
        "probabilities": {"Normal": 0.90},
        "primary_finding": "Normal",
        "top_class": "Normal"
    }

    report = MultimodalClinicalIntegrator.synthesize_report(
        patient_info=patient_info,
        symptom_data=symptom_data,
        vital_signs=None,
        xray_result=xray_result,
        projection_view="PA view"
    )

    assert len(report["discordant_insights"]) > 0
    assert any("Chest Pain with Normal Radiograph" in d["clinical_concern"] for d in report["discordant_insights"])


def test_emergency_override_critical_hypoxemia():
    """
    Severe hypoxemia (SpO2 < 90%) overrides radiograph and forces EMERGENCY WARNING.
    """
    patient_info = {"age": 60, "sex": "Female"}
    symptom_data = {
        "narrative": "Feeling weak and tired",
        "current_symptoms": ["fatigue"],
        "negated_symptoms": [],
        "duration_days": 1.0,
        "severity": 4
    }
    xray_result = {
        "flagged_findings": ["Normal"],
        "probabilities": {"Normal": 0.88},
        "primary_finding": "Normal",
        "top_class": "Normal"
    }

    report = MultimodalClinicalIntegrator.synthesize_report(
        patient_info=patient_info,
        symptom_data=symptom_data,
        vital_signs={"SpO2": 84},
        xray_result=xray_result,
        projection_view="PA view"
    )

    assert report["decision_status"] == "EMERGENCY WARNING"
    assert len(report["emergency_triggers"]) > 0


# ── 5. PDF REPORT GENERATION TEST ────────────────────────────────────────────

def test_multimodal_pdf_generation_without_error():
    """Verifies that generate_multimodal_xray_pdf produces valid PDF bytes."""
    patient_info = {"age": 50, "sex": "Male", "medical_history": ["Hypertension"]}
    symptom_data = {
        "narrative": "Mild cough for 2 days",
        "current_symptoms": ["cough"],
        "negated_symptoms": ["fever", "chest pain"],
        "duration_days": 2.0,
        "severity": 3
    }
    xray_result = {
        "flagged_findings": ["Normal"],
        "probabilities": {"Normal": 0.82, "Infiltration": 0.12},
        "primary_finding": "Normal",
        "top_class": "Normal"
    }
    report = MultimodalClinicalIntegrator.synthesize_report(
        patient_info=patient_info,
        symptom_data=symptom_data,
        vital_signs={"HeartRate": 72, "SpO2": 99},
        xray_result=xray_result,
        projection_view="PA view"
    )

    pdf_bytes = generate_multimodal_xray_pdf(report)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
