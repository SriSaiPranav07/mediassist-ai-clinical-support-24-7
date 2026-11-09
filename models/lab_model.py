"""
MediAssist - Laboratory Result Interpreter & Critical Value Alerts
Author: Zelvora Industry Labs / MediAssist Team
"""
from typing import Dict, Any, List, Optional
from src.config import LAB_REFERENCE_RANGES
from utils.data_processing import validate_laboratory_values


class LabInterpreterModel:
    """
    Expert Clinical Laboratory Interpreter grounded in Tietz and Harrison Compendiums.
    """

    def __init__(self, reference_ranges: Optional[Dict[str, Any]] = None):
        self.reference_ranges = reference_ranges or LAB_REFERENCE_RANGES

    def interpret_panel(
        self,
        panel_dict: Dict[str, Any],
        patient_sex: str = "Male",
        patient_age: int = 40,
        fasting: bool = False
    ) -> Dict[str, Any]:
        """
        Interprets laboratory panel entries against sex- and fasting-stratified reference intervals.
        """
        # Validate inputs
        has_valid, cleaned_labs, validation_warnings = validate_laboratory_values(panel_dict, self.reference_ranges)
        
        evaluated_tests = []
        critical_alerts = []
        abnormal_count = 0
        normal_count = 0

        for test_name, spec in self.reference_ranges.items():
            if test_name not in cleaned_labs:
                continue

            val = cleaned_labs[test_name]["value"]
            unit = cleaned_labs[test_name]["unit"]

            # Select applicable reference interval
            if test_name == "Glucose":
                interval = spec["ranges"]["Fasting"] if fasting else spec["ranges"].get("Random", spec["ranges"]["All"])
            elif patient_sex in spec["ranges"]:
                interval = spec["ranges"][patient_sex]
            else:
                interval = spec["ranges"].get("All", (0.0, 100.0))

            low_lim, high_lim = interval
            crit_low = spec.get("critical_low")
            crit_high = spec.get("critical_high")

            status = "Normal"
            clinical_imp = "Measured concentration falls within established physiological boundaries."
            recommendation = "Continue routine health maintenance."

            if crit_low is not None and val < crit_low:
                status = "Critical Low"
                clinical_imp = spec.get("low_implication", "Critically depressed value.")
                recommendation = "STAT clinical notification; immediate bedside protocol intervention required."
                critical_alerts.append(f"{test_name} ({val} {unit}) is Critically Low (< {crit_low})")
                abnormal_count += 1
            elif crit_high is not None and val > crit_high:
                status = "Critical High"
                clinical_imp = spec.get("high_implication", "Critically elevated value.")
                recommendation = "STAT clinical notification; initiate emergency clinical protocol immediately."
                critical_alerts.append(f"{test_name} ({val} {unit}) is Critically High (> {crit_high})")
                abnormal_count += 1
            elif val < low_lim:
                status = "Low"
                clinical_imp = spec.get("low_implication", "Below reference interval.")
                recommendation = "Correlate with clinical history, repeat specimen, and consider diagnostic follow-up."
                abnormal_count += 1
            elif val > high_lim:
                status = "High"
                clinical_imp = spec.get("high_implication", "Above reference interval.")
                recommendation = "Evaluate underlying etiology and monitor serial laboratory trajectories."
                abnormal_count += 1
            else:
                normal_count += 1

            # 10-Second Explanation per test
            finding_text = f"{test_name}: {status} at {val} {unit}."
            reason_text = (
                f"Result is outside reference interval ({low_lim} - {high_lim} {unit})."
                if status != "Normal" else
                f"Result is within normal reference interval ({low_lim} - {high_lim} {unit})."
            )

            evaluated_tests.append({
                "test_name": test_name,
                "value": val,
                "unit": unit,
                "reference_interval": interval,
                "critical_limits": (crit_low, crit_high),
                "status": status,
                "clinical_implication": clinical_imp,
                "recommendation": recommendation,
                "explanation": {
                    "finding": finding_text,
                    "reason": reason_text,
                    "confidence": "Deterministic rule based on Tietz & Harrison Reference Compendium.",
                    "limitation": "Must be verified against local hospital laboratory instrumentation ranges."
                }
            })

        total_tests = len(evaluated_tests)
        has_critical = len(critical_alerts) > 0

        # Construct overarching 10-second panel explanation
        if has_critical:
            panel_finding = f"CRITICAL ALERT: {len(critical_alerts)} panic laboratory value(s) identified."
            panel_reason = "; ".join(critical_alerts)
            panel_action = "Stat physician notification and emergency protocol intervention mandated."
        elif abnormal_count > 0:
            panel_finding = f"{abnormal_count} of {total_tests} test(s) fall outside standard reference intervals."
            panel_reason = "Out-of-range metabolic/cardiac/hematologic indicators present."
            panel_action = "Review within context of clinical presentation and plan diagnostic workup."
        elif total_tests > 0:
            panel_finding = f"All {total_tests} evaluated laboratory tests are within normal limits."
            panel_reason = "No acute metabolic or hematologic derangements detected."
            panel_action = "Continue standard clinical care."
        else:
            panel_finding = "No laboratory values submitted for analysis."
            panel_reason = "Empty laboratory panel input."
            panel_action = "Submit diagnostic laboratory panels when available."

        return {
            "total_tests": total_tests,
            "has_critical": has_critical,
            "critical_count": len(critical_alerts),
            "critical_alerts": critical_alerts,
            "abnormal_count": abnormal_count,
            "normal_count": normal_count,
            "tests": evaluated_tests,
            "warnings": validation_warnings,
            "explanation": {
                "finding": panel_finding,
                "reason": panel_reason,
                "confidence": "100% deterministic calibration against authoritative reference intervals.",
                "limitation": "Laboratory values must always be confirmed against institution-specific lab reference ranges."
            },
            "panel_action": panel_action
        }
