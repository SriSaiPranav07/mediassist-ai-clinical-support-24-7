"""
MediAssist - Robust Data Validation and Safety Sanitation
Author: Zelvora Industry Labs / MediAssist Team
"""
from typing import Dict, Any, Tuple, Optional, List
import io
import re
from PIL import Image

def validate_symptom_text(text: Optional[str]) -> Tuple[bool, str, str]:
    """
    Validates patient chief complaint / symptom description.
    Returns: (is_valid, cleaned_text, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "", "Symptom description cannot be empty."
    
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 3:
        return False, cleaned, "Symptom description is too short (minimum 3 characters required)."
    
    if len(cleaned) > 2000:
        return False, cleaned[:2000], "Symptom description exceeds maximum allowed length of 2000 characters."
    
    # Check if string has at least some alphabetic characters
    if not re.search(r"[a-zA-Z]", cleaned):
        return False, cleaned, "Symptom description must contain valid words or text."
    
    return True, cleaned, ""

def validate_patient_demographics(age: Any, sex: str) -> Tuple[bool, str]:
    """
    Validates patient age and sex.
    """
    try:
        age_int = int(age)
        if age_int < 0 or age_int > 125:
            return False, f"Age {age_int} is outside valid physiological boundary (0-125 years)."
    except (ValueError, TypeError):
        return False, "Age must be an integer between 0 and 125."

    if sex not in ["Male", "Female", "Other", "Unknown"]:
        return False, f"Invalid sex '{sex}'. Supported options: Male, Female, Other, Unknown."

    return True, ""

def validate_vital_signs(vitals: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates vital signs against plausible physiological limits.
    Returns: (all_valid, list_of_warning_or_error_messages)
    """
    issues = []
    if not isinstance(vitals, dict):
        return False, ["Vital signs must be provided as a key-value dictionary."]

    checks = {
        "HeartRate": (20, 260, "Heart Rate", "bpm"),
        "SpO2": (40, 100, "Oxygen Saturation", "%"),
        "SystolicBP": (40, 300, "Systolic BP", "mmHg"),
        "DiastolicBP": (20, 200, "Diastolic BP", "mmHg"),
        "RespiratoryRate": (4, 70, "Respiratory Rate", "breaths/min"),
        "Temperature": (30.0, 44.0, "Body Temperature", "°C"),
    }

    for key, (min_v, max_v, name, unit) in checks.items():
        if key in vitals and vitals[key] is not None:
            val = vitals[key]
            try:
                num = float(val)
                if num < min_v or num > max_v:
                    issues.append(f"{name} ({num} {unit}) is outside physiologically plausible range [{min_v}-{max_v}].")
            except (ValueError, TypeError):
                issues.append(f"{name} has non-numeric value: '{val}'.")

    if "SystolicBP" in vitals and "DiastolicBP" in vitals:
        try:
            sys_val = float(vitals["SystolicBP"])
            dia_val = float(vitals["DiastolicBP"])
            if dia_val >= sys_val:
                issues.append(f"Diastolic BP ({dia_val}) cannot be greater than or equal to Systolic BP ({sys_val}).")
        except (ValueError, TypeError):
            pass

    return len(issues) == 0, issues

def validate_lab_input(test_name: str, value: Any, unit: str, supported_ranges: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates laboratory test input value, unit, and test name.
    """
    if test_name not in supported_ranges:
        return False, f"Test '{test_name}' is not in the recognized clinical laboratory compendium."

    expected_unit = supported_ranges[test_name]["unit"]
    if unit.strip().lower() != expected_unit.lower():
        return False, f"Unit mismatch for {test_name}: expected '{expected_unit}', but received '{unit}'."

    try:
        num = float(value)
        if num < 0:
            return False, f"Laboratory value for {test_name} cannot be negative ({num})."
    except (ValueError, TypeError):
        return False, f"Laboratory value for {test_name} must be a valid number, got '{value}'."

    return True, ""

def validate_image_file(image_bytes: bytes, max_mb: float = 10.0) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Validates uploaded medical image bytes (size, format, loadability).
    """
    if not image_bytes:
        return False, None, "Image data is empty."

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        return False, None, f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit ({max_mb} MB)."

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify integrity
        # Re-open after verify because verify exhausts buffer
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.format not in ["JPEG", "PNG", "TIFF", "BMP"]:
            return False, None, f"Unsupported image format: {img.format}. Please provide JPEG, PNG, or TIFF."
        
        width, height = img.size
        if width < 64 or height < 64:
            return False, None, f"Image dimensions ({width}x{height}) are too small for diagnostic feature extraction (min 64x64)."

        return True, img, ""
    except Exception as e:
        return False, None, f"Image parsing error: {str(e)}. File may be corrupted."
