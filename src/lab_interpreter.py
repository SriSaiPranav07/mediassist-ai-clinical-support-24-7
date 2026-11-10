"""
MediAssist - Laboratory Result Interpreter & Critical Alert Engine
Author: MediAssist Clinical AI Team
Reference Sources: Tietz Clinical Guide to Laboratory Tests, 4th Ed; Harrison's Internal Medicine, 21st Ed.
"""
from typing import Dict, Any, List, Optional, Tuple
from .config import LAB_REFERENCE_RANGES
from .data_validation import validate_lab_input

class LabInterpreter:
    """
    Interprets laboratory panels using authoritative reference ranges and critical boundaries.
    """

    def __init__(self, custom_ranges: Optional[Dict[str, Any]] = None):
        self.ranges = custom_ranges or LAB_REFERENCE_RANGES

    def get_reference_interval(self, test_name: str, patient_sex: str = "All", fasting: bool = False) -> Tuple[float, float]:
        """
        Retrieves context-specific reference range for a given test.
        """
        spec = self.ranges.get(test_name)
        if not spec:
            raise ValueError(f"Unknown test: {test_name}")

        ranges = spec["ranges"]
        if test_name == "Glucose":
            key = "Fasting" if fasting else "Random"
            return ranges.get(key, ranges.get("All", (70.0, 140.0)))
        
        if patient_sex in ranges:
            return ranges[patient_sex]
        elif "All" in ranges:
            return ranges["All"]
        else:
            return list(ranges.values())[0]

    def interpret_single_test(
        self,
        test_name: str,
        value: float,
        unit: str,
        patient_sex: str = "All",
        patient_age: int = 40,
        fasting: bool = False
    ) -> Dict[str, Any]:
        """
        Interprets a single laboratory test result against reference intervals and critical limits.
        """
        # Step 1: Input validation
        valid, msg = validate_lab_input(test_name, value, unit, self.ranges)
        if not valid:
            return {
                "test_name": test_name,
                "value": value,
                "unit": unit,
                "status": "Invalid Input",
                "severity": "Unknown",
                "reference_interval": (0, 0),
                "is_critical": False,
                "explanation": {
                    "finding": f"Unable to interpret {test_name}.",
                    "reason": msg,
                    "confidence": "N/A",
                    "limitation": "Inputs failing data validation cannot be evaluated."
                },
                "clinical_implication": msg,
                "recommendation": "Review test requisition and verify entered units and value."
            }

        val = float(value)
        spec = self.ranges[test_name]
        ref_low, ref_high = self.get_reference_interval(test_name, patient_sex=patient_sex, fasting=fasting)
        crit_low = spec.get("critical_low")
        crit_high = spec.get("critical_high")

        # Step 2: Categorization
        status = "Normal"
        is_critical = False
        severity = "Normal"
        implication = "Value is within normal physiological limits."
        recommendation = "Maintain routine follow-up as clinically indicated."

        if crit_low is not None and val < crit_low:
            status = "Critical Low"
            is_critical = True
            severity = "Critical"
            implication = spec.get("low_implication", "Potentially life-threatening deficiency.")
            recommendation = "Stat repeat confirmation; notify attending physician immediately."
        elif crit_high is not None and val > crit_high:
            status = "Critical High"
            is_critical = True
            severity = "Critical"
            implication = spec.get("high_implication", "Potentially life-threatening elevation.")
            recommendation = "Immediate physician escalation and clinical protocol activation."
        elif val < ref_low:
            status = "Low"
            severity = "Abnormal"
            implication = spec.get("low_implication", "Subnormal laboratory level.")
            recommendation = "Correlate with clinical history and investigate underlying etiology."
        elif val > ref_high:
            status = "High"
            severity = "Abnormal"
            implication = spec.get("high_implication", "Supranormal laboratory level.")
            recommendation = "Correlate with patient symptoms, medications, and repeat if indicated."

        # Step 3: Clinician 10-Second Explanation
        finding_desc = f"{test_name} result is {status} ({val} {unit})."
        reason_desc = (
            f"Reference interval is {ref_low} - {ref_high} {unit}."
            if status == "Normal" else
            f"Exceeds physiological boundaries: reference is {ref_low} - {ref_high} {unit}. {implication}"
        )

        return {
            "test_name": test_name,
            "value": val,
            "unit": unit,
            "status": status,
            "severity": severity,
            "reference_interval": (ref_low, ref_high),
            "critical_limits": (crit_low, crit_high),
            "is_critical": is_critical,
            "clinical_implication": implication,
            "recommendation": recommendation,
            "explanation": {
                "finding": finding_desc,
                "reason": reason_desc,
                "confidence": "Authoritative Clinical Reference Interval Rule (Tietz/Harrison).",
                "limitation": "Reference intervals may vary by analyzer, reagent manufacturer, and local hospital policy."
            }
        }

    def interpret_panel(
        self,
        panel_dict: Dict[str, Dict[str, Any]],
        patient_sex: str = "All",
        patient_age: int = 40,
        fasting: bool = False
    ) -> Dict[str, Any]:
        """
        Interprets a full multi-test laboratory panel.
        Input format: { "Hemoglobin": {"value": 11.2, "unit": "g/dL"}, ... }
        """
        results = []
        critical_count = 0
        abnormal_count = 0
        normal_count = 0

        for test_name, payload in panel_dict.items():
            if not isinstance(payload, dict):
                continue
            val = payload.get("value")
            unit = payload.get("unit", "")
            if val is None or val == "":
                continue

            res = self.interpret_single_test(
                test_name=test_name,
                value=val,
                unit=unit,
                patient_sex=patient_sex,
                patient_age=patient_age,
                fasting=fasting
            )
            results.append(res)
            if res["is_critical"]:
                critical_count += 1
            elif res["status"] in ["Low", "High"]:
                abnormal_count += 1
            elif res["status"] == "Normal":
                normal_count += 1

        return {
            "total_tests": len(results),
            "critical_count": critical_count,
            "abnormal_count": abnormal_count,
            "normal_count": normal_count,
            "has_critical": critical_count > 0,
            "tests": results
        }
