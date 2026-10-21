"""
MediAssist - Helper Utilities, Synthetic Clinical Case Generator, and Image Generators
Author: Zelvora Industry Labs / MediAssist Team
"""
import io
import json
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from pathlib import Path
from .config import SAMPLE_XRAYS_DIR, DATA_DIR

DEMO_PATIENTS: List[Dict[str, Any]] = [
    {
        "id": "PT-101",
        "name": "Rajesh Kumar (Demo)",
        "age": 58,
        "sex": "Male",
        "chief_complaint": "Experiencing severe crushing chest pain radiating to left arm and jaw for the past 45 minutes with cold diaphoresis and nausea.",
        "duration_days": 0.1,
        "severity": 10,
        "vitals": {
            "HeartRate": 118,
            "SpO2": 93,
            "SystolicBP": 165,
            "DiastolicBP": 98,
            "RespiratoryRate": 24,
            "Temperature": 36.8
        },
        "labs": {
            "Troponin_I": {"value": 1.45, "unit": "ng/mL"},
            "Glucose": {"value": 182, "unit": "mg/dL"},
            "Potassium": {"value": 4.1, "unit": "mEq/L"},
            "Creatinine": {"value": 1.2, "unit": "mg/dL"}
        },
        "xray_sample": "cardiomegaly_sample.png",
        "expected_triage": "Emergency Warning",
        "clinical_scenario": "Suspected Acute ST-Elevation Myocardial Infarction (STEMI)"
    },
    {
        "id": "PT-102",
        "name": "Priya Sharma (Demo)",
        "age": 42,
        "sex": "Female",
        "chief_complaint": "High fever, productive cough with yellow sputum, and worsening shortness of breath for 4 days.",
        "duration_days": 4.0,
        "severity": 7,
        "vitals": {
            "HeartRate": 102,
            "SpO2": 92,
            "SystolicBP": 115,
            "DiastolicBP": 75,
            "RespiratoryRate": 26,
            "Temperature": 38.9
        },
        "labs": {
            "WBC": {"value": 16.8, "unit": "10^3/uL"},
            "Hemoglobin": {"value": 13.0, "unit": "g/dL"},
            "Sodium": {"value": 136, "unit": "mEq/L"},
            "Potassium": {"value": 3.9, "unit": "mEq/L"}
        },
        "xray_sample": "infiltration_sample.png",
        "expected_triage": "Urgent Assessment",
        "clinical_scenario": "Community-Acquired Pneumonia with Secondary Hypoxemia"
    },
    {
        "id": "PT-103",
        "name": "Ananya Sen (Demo)",
        "age": 29,
        "sex": "Female",
        "chief_complaint": "Mild intermittent dry cough and nasal congestion for 3 days. No breathing difficulty, chest pain, or high fever.",
        "duration_days": 3.0,
        "severity": 3,
        "vitals": {
            "HeartRate": 74,
            "SpO2": 99,
            "SystolicBP": 118,
            "DiastolicBP": 78,
            "RespiratoryRate": 14,
            "Temperature": 37.0
        },
        "labs": {
            "WBC": {"value": 6.2, "unit": "10^3/uL"},
            "Hemoglobin": {"value": 13.5, "unit": "g/dL"},
            "Platelets": {"value": 240, "unit": "10^3/uL"}
        },
        "xray_sample": "normal_sample.png",
        "expected_triage": "Routine Assessment",
        "clinical_scenario": "Uncomplicated Upper Respiratory Tract Infection (URTI)"
    },
    {
        "id": "PT-104",
        "name": "Vikram Adani (Demo)",
        "age": 64,
        "sex": "Male",
        "chief_complaint": "Sudden onset of right-sided arm weakness and mild facial droop noticed 30 minutes ago by spouse.",
        "duration_days": 0.05,
        "severity": 9,
        "vitals": {
            "HeartRate": 88,
            "SpO2": 97,
            "SystolicBP": 185,
            "DiastolicBP": 105,
            "RespiratoryRate": 18,
            "Temperature": 36.9
        },
        "labs": {
            "Glucose": {"value": 140, "unit": "mg/dL"},
            "Platelets": {"value": 210, "unit": "10^3/uL"},
            "Creatinine": {"value": 1.1, "unit": "mg/dL"}
        },
        "xray_sample": "normal_sample.png",
        "expected_triage": "Emergency Warning",
        "clinical_scenario": "Suspected Acute Ischemic Stroke (FAST Positive)"
    },
    {
        "id": "PT-105",
        "name": "Meera Nair (Demo)",
        "age": 51,
        "sex": "Female",
        "chief_complaint": "Generalized weakness, nausea, polyuria, and severe lethargy for 2 days.",
        "duration_days": 2.0,
        "severity": 8,
        "vitals": {
            "HeartRate": 112,
            "SpO2": 96,
            "SystolicBP": 95,
            "DiastolicBP": 62,
            "RespiratoryRate": 28,
            "Temperature": 37.1
        },
        "labs": {
            "Glucose": {"value": 465, "unit": "mg/dL"},
            "Potassium": {"value": 5.8, "unit": "mEq/L"},
            "Sodium": {"value": 128, "unit": "mEq/L"},
            "Creatinine": {"value": 2.1, "unit": "mg/dL"}
        },
        "xray_sample": "normal_sample.png",
        "expected_triage": "Urgent Assessment",
        "clinical_scenario": "Severe Hyperglycemia / Impending Diabetic Ketoacidosis (DKA)"
    }
]

def generate_anatomical_xray(pathology: str = "normal", size: tuple = (256, 256)) -> Image.Image:
    """
    Generates an anatomically plausible synthetic chest radiograph for demonstration and testing.
    Renders thoracic cage, bilateral lung fields, cardiac silhouette, and pathology artifacts.
    """
    w, h = size
    # Background: dark radiograph canvas
    img = Image.new("L", (w, h), color=25)
    draw = ImageDraw.Draw(img)

    # Chest wall outline / soft tissues (mid-gray)
    draw.rounded_rectangle([15, 20, w - 15, h - 10], radius=35, fill=45)

    # Thoracic spine & mediastinum (central dense column)
    draw.rectangle([w * 0.44, 25, w * 0.56, h - 20], fill=140)

    # Clavicles (horizontal bony structures)
    draw.line([30, 45, w * 0.45, 55], fill=175, width=7)
    draw.line([w * 0.55, 55, w - 30, 45], fill=175, width=7)

    # Rib cage shadows (paired curved lines)
    for y_offset in range(70, h - 50, 28):
        # Left rib
        draw.arc([20, y_offset, w * 0.46, y_offset + 30], start=180, end=360, fill=90, width=4)
        # Right rib
        draw.arc([w * 0.54, y_offset, w - 20, y_offset + 30], start=180, end=360, fill=90, width=4)

    # Lung fields (air-filled = radiolucent / darker)
    # Right lung (anatomical right = viewer left)
    draw.ellipse([35, 60, w * 0.43, h - 50], fill=28)
    # Left lung (anatomical left = viewer right)
    draw.ellipse([w * 0.57, 60, w - 35, h - 50], fill=28)

    # Diaphragmatic domes
    draw.pieslice([25, h - 75, w * 0.47, h + 25], 180, 360, fill=160)
    draw.pieslice([w * 0.53, h - 70, w - 25, h + 25], 180, 360, fill=160)

    # Cardiac silhouette
    if pathology.lower() == "cardiomegaly":
        # Greatly enlarged cardiac shadow extending significantly into left hemithorax
        draw.ellipse([w * 0.30, h * 0.45, w * 0.78, h - 55], fill=175)
    else:
        # Normal cardiac silhouette (cardiothoracic ratio < 0.5)
        draw.ellipse([w * 0.38, h * 0.50, w * 0.65, h - 60], fill=165)

    # Specific pathology visual markers:
    if pathology.lower() == "effusion":
        # Blunting of costophrenic angle and dense basal fluid meniscus
        draw.polygon([(w * 0.55, h - 55), (w - 30, h - 55), (w - 30, h - 85)], fill=190)
    elif pathology.lower() == "infiltration" or pathology.lower() == "pneumonia":
        # Patchy alveolar consolidation (irregular fuzzy opacity in right mid-lower lung)
        for dx, dy in [(0, 0), (12, -8), (-10, 10), (15, 12), (-5, -12)]:
            draw.ellipse([w * 0.18 + dx, h * 0.52 + dy, w * 0.36 + dx, h * 0.72 + dy], fill=120)
    elif pathology.lower() == "atelectasis":
        # Linear/plate-like volume loss opacity
        draw.line([w * 0.15, h * 0.65, w * 0.40, h * 0.62], fill=180, width=8)

    # Apply realistic radiographic gaussian diffusion blur
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    return img_blurred

def initialize_sample_data():
    """
    Saves demo patient database and synthetic chest X-ray samples into the filesystem.
    """
    SAMPLE_XRAYS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save demo patients JSON
    patients_file = DATA_DIR / "demo_patients.json"
    with open(patients_file, "w", encoding="utf-8") as f:
        json.dump(DEMO_PATIENTS, f, indent=2)

    # Generate and save sample X-rays
    samples = {
        "normal_sample.png": "normal",
        "cardiomegaly_sample.png": "cardiomegaly",
        "effusion_sample.png": "effusion",
        "infiltration_sample.png": "infiltration"
    }

    for filename, pathology in samples.items():
        filepath = SAMPLE_XRAYS_DIR / filename
        if not filepath.exists():
            img = generate_anatomical_xray(pathology)
            img.save(filepath, format="PNG")
