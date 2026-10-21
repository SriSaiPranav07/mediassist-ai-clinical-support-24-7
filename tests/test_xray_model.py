"""
Unit Tests for Chest X-Ray Model & Grad-CAM Visualizations
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

@pytest.fixture
def xray_screener():
    return ChestXRayScreener()

def test_xray_inference_pipeline(xray_screener):
    img = generate_anatomical_xray("cardiomegaly")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = xray_screener.analyze_radiograph(img_bytes)
    assert result["success"] is True
    assert "probabilities" in result
    assert len(result["probabilities"]) == 6
    assert "explanation" in result
    assert "finding" in result["explanation"]
    assert "limitation" in result["explanation"]

def test_gradcam_generation(xray_screener):
    img = generate_anatomical_xray("cardiomegaly")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = xray_screener.analyze_radiograph(img_bytes)
    # Check that probabilities sum or exist and gradcam is computed if model is loaded
    if xray_screener.model is not None:
        assert result["gradcam_available"] is True
        assert result["gradcam_image"] is not None

def test_corrupted_image_rejection(xray_screener):
    result = xray_screener.analyze_radiograph(b"Not an image corrupted payload")
    assert result["success"] is False
    assert "error" in result
