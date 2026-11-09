"""
MediAssist - Automated Tests for RAG Guideline Engine, OOD Image Validation & PDF Export
"""
import io
from PIL import Image
import numpy as np
import pytest

from models.rag_engine import MedicalGuidelineRAGEngine
from utils.data_processing import validate_radiograph_image, validate_symptom_narrative
from utils.pdf_generator import generate_clinical_summary_pdf


def test_rag_engine_retrieval():
    engine = MedicalGuidelineRAGEngine()
    assert len(engine.guidelines) >= 5, "Guidelines corpus should load successfully"
    
    # Query for heart attack / chest pain
    results = engine.retrieve("Crushing chest pain radiating to left arm with diaphoresis", top_k=2)
    assert len(results) > 0, "Should retrieve relevant cardiac guidelines"
    top_result = results[0]
    assert "STEMI" in top_result["topic"] or "Coronary" in top_result["topic"], f"Unexpected top guideline: {top_result['topic']}"


def test_rag_synthesis():
    engine = MedicalGuidelineRAGEngine()
    synth = engine.synthesize_guideline_evidence(
        presentation_text="Severe sudden facial droop and right arm weakness",
        urgency_category="Emergency Warning",
        detected_pathologies=[],
        critical_labs=[]
    )
    assert synth["has_evidence"] is True
    assert "Stroke" in synth["synthesized_summary"] or "AHA/ASA" in synth["primary_source"]


def test_ood_image_rejection():
    # 1. Create a high-chroma colorful image (e.g., pure red/green cartoon)
    colorful_img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    bio_color = io.BytesIO()
    colorful_img.save(bio_color, format="PNG")
    
    is_valid, err, meta = validate_radiograph_image(bio_color.getvalue())
    assert is_valid is False, "High chromatic divergence non-radiograph must be rejected"
    assert "Out-of-Distribution" in err


def test_valid_grayscale_image_acceptance():
    # 2. Create a simulated monochrome radiograph
    arr = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
    mono_img = Image.fromarray(arr).convert("RGB")
    bio_mono = io.BytesIO()
    mono_img.save(bio_mono, format="PNG")
    
    is_valid, err, meta = validate_radiograph_image(bio_mono.getvalue())
    assert is_valid is True, f"Authentic monochrome radiograph should pass OOD check. Error: {err}"


def test_pdf_clinical_summary_generation():
    session_data = {
        "patient_name": "Test Patient",
        "age": 52,
        "sex": "Female",
        "duration_days": 2.0,
        "severity_score": 7,
        "chief_complaint": "Persistent high fever and productive cough with yellow phlegm."
    }
    triage_result = {
        "urgency_category": "Urgent Assessment",
        "action_advisory": "Prompt clinical evaluation required.",
        "confidence": 0.85,
        "safety_override": False,
        "explanation": {
            "finding": "Urgent Assessment.",
            "reason": "Elevated fever and productive cough.",
            "confidence": "85.0%",
            "limitation": "Requires physician examination."
        }
    }
    
    pdf_bytes = generate_clinical_summary_pdf(session_data, triage_result=triage_result)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000, "PDF size should be substantial"
    assert pdf_bytes.startswith(b"%PDF"), "Output buffer must be a valid PDF document"
