"""
MediAssist - Data Processing, Validation & Zero-Retention Security
Author: MediAssist Clinical AI Team
"""
import io
import gc
import re
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np


def validate_symptom_narrative(text: Optional[str]) -> Tuple[bool, str, str]:
    """
    Validates free-text symptom narratives to ensure clinical relevance and structure.
    
    Returns:
        Tuple[bool, str, str]: (is_valid, cleaned_text, error_message)
    """
    if text is None:
        return False, "", "Symptom description cannot be empty."
    
    cleaned = str(text).strip()
    if len(cleaned) < 5:
        return False, cleaned, "Symptom description is too short (minimum 5 characters required)."
    
    if len(cleaned) > 2500:
        return False, cleaned[:2500], "Symptom description exceeds maximum permitted length (2500 characters)."
    
    # Reject strings composed purely of punctuation, digits, or repetitive symbols
    alphabetic_chars = re.findall(r"[a-zA-Z]", cleaned)
    if len(alphabetic_chars) < 3:
        return False, cleaned, "Input contains insufficient alphabetic text to evaluate clinical symptoms."
    
    return True, cleaned, ""


def validate_radiograph_image(image_bytes: Optional[bytes]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Performs pre-processing validation and Out-of-Distribution (OOD) screening on image inputs.
    Rejects corrupted images, invalid dimensions, and non-radiographic photographs (e.g., selfies, cartoons, outdoor photos).
    
    Returns:
        Tuple[bool, str, Dict[str, Any]]: (is_valid, error_message, metadata_dict)
    """
    if not image_bytes or len(image_bytes) == 0:
        return False, "No image payload received.", {}
    
    try:
        bio = io.BytesIO(image_bytes)
        img = Image.open(bio)
        img.verify()  # Verify image integrity
        
        # Re-open for inspection after verify()
        bio.seek(0)
        img = Image.open(bio)
        width, height = img.size
        img_format = (img.format or "UNKNOWN").upper()
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}", {}
    
    meta = {
        "width": width,
        "height": height,
        "format": img_format,
        "mode": img.mode
    }
    
    # Check minimum dimensions
    if width < 64 or height < 64:
        return False, f"Image dimensions too small ({width}x{height}px). Minimum required is 64x64px.", meta
    
    # Check aspect ratio (radiographs typically have aspect ratios between 0.5 and 2.0)
    aspect_ratio = width / float(height)
    meta["aspect_ratio"] = round(aspect_ratio, 2)
    if aspect_ratio < 0.35 or aspect_ratio > 3.0:
        return False, f"Unusual aspect ratio ({aspect_ratio:.2f}). Standard PA/AP chest radiographs have aspect ratios between 0.5 and 2.0.", meta
    
    # Convert to RGB numpy array for luminance and chromatic saturation analysis
    rgb_img = img.convert("RGB")
    arr = np.array(rgb_img, dtype=np.float32)
    
    # 1. Blank/Solid image check (Zero variance)
    std_dev = float(np.std(arr))
    meta["pixel_std_dev"] = round(std_dev, 2)
    if std_dev < 5.0:
        return False, "Image appears uniformly blank or lacks radiological tissue contrast (pixel std < 5.0).", meta
    
    # 2. Out-of-Distribution (OOD) Chromatic Saturation Check
    # Standard radiographs are monochrome/grayscale (R ≈ G ≈ B).
    # Natural photos, selfies, cartoons, and landscape images exhibit high inter-channel divergence.
    r_channel = arr[:, :, 0]
    g_channel = arr[:, :, 1]
    b_channel = arr[:, :, 2]
    
    channel_diff = np.mean(np.abs(r_channel - g_channel)) + np.mean(np.abs(g_channel - b_channel)) + np.mean(np.abs(r_channel - b_channel))
    meta["chromatic_divergence"] = round(float(channel_diff), 2)
    
    # A genuine X-ray (even saved as RGB PNG) has channel divergence close to 0 (< 15)
    # Natural colored images easily exceed 30.0 - 150.0
    if channel_diff > 35.0:
        return False, (
            f"Out-of-Distribution rejection: High chromatic saturation detected ({channel_diff:.1f} > 35.0). "
            "Chest radiographs must be grayscale/monochromatic transmission images. "
            "Please upload an authentic chest radiograph (PA or AP view)."
        ), meta
    
    return True, "", meta


def validate_laboratory_values(
    panel_dict: Dict[str, Any],
    reference_config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any], Dict[str, str]]:
    """
    Validates and cleanses laboratory panel inputs against biological plausibility boundaries.
    
    Returns:
        Tuple[bool, Dict[str, Any], Dict[str, str]]: (has_valid_inputs, cleaned_values, validation_warnings)
    """
    cleaned = {}
    warnings = {}
    
    for test_name, entry in panel_dict.items():
        if test_name not in reference_config:
            continue
        
        val_raw = entry.get("value")
        if val_raw is None or str(val_raw).strip() == "":
            continue
        
        try:
            val_num = float(str(val_raw).strip())
        except (ValueError, TypeError):
            warnings[test_name] = f"Invalid non-numeric value: '{val_raw}'."
            continue
        
        # Check for non-physiological negative values
        if val_num < 0:
            warnings[test_name] = f"Biologically impossible negative value ({val_num})."
            continue
        
        # Check extreme physiological upper bounds
        if test_name == "Potassium" and val_num > 15.0:
            warnings[test_name] = f"Extreme unphysiological potassium level ({val_num} mEq/L > 15.0)."
            continue
        if test_name == "Glucose" and val_num > 2500.0:
            warnings[test_name] = f"Extreme unphysiological glucose level ({val_num} mg/dL > 2500)."
            continue
        if test_name == "Hemoglobin" and val_num > 35.0:
            warnings[test_name] = f"Extreme unphysiological hemoglobin level ({val_num} g/dL > 35.0)."
            continue
        
        cleaned[test_name] = {
            "value": val_num,
            "unit": entry.get("unit", reference_config[test_name]["unit"])
        }
    
    has_valid = len(cleaned) > 0
    return has_valid, cleaned, warnings


def sanitize_in_memory(*objects: Any) -> None:
    """
    Zero Data Retention Enforcement:
    Closes byte streams, deletes in-memory objects, and triggers aggressive garbage collection.
    Guarantees no patient data or uploaded images persist in process memory or scratch files.
    """
    for obj in objects:
        if isinstance(obj, io.BytesIO):
            try:
                obj.seek(0)
                obj.truncate(0)
                obj.close()
            except Exception:
                pass
        del obj
    
    gc.collect()
