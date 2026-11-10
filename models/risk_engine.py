"""
MediAssist - Composite Patient Risk Scoring & Transparent Prioritization
Author: MediAssist Clinical AI Team
"""
from typing import Dict, Any, List, Optional
from src.config import TRIAGE_EMERGENCY, TRIAGE_URGENT, TRIAGE_ROUTINE


class CompositeRiskEngine:
    """
    Synthesizes multi-modal inputs (symptoms, vitals, labs, imaging) into a transparent risk score (0-100).
    Guarantees no risk-diluting averages when emergency life-threats are present.
    """

    def calculate_risk(
        self,
        triage_output: Optional[Dict[str, Any]] = None,
        lab_output: Optional[Dict[str, Any]] = None,
        xray_output: Optional[Dict[str, Any]] = None,
        vital_signs: Optional[Dict[str, Any]] = None,
        patient_age: int = 40
    ) -> Dict[str, Any]:
        """
        Executes multi-criteria prioritization with absolute deterministic safety overrides.
        """
        base_score = 15.0
        contributing_factors = []
        is_override = False

        # 1. Symptom Triage Contribution
        if triage_output:
            urgency = triage_output.get("urgency_category")
            if urgency == TRIAGE_EMERGENCY:
                base_score += 45.0
                contributing_factors.append("Emergency symptom presentation / red flag detected (+45)")
            elif urgency == TRIAGE_URGENT:
                base_score += 25.0
                contributing_factors.append("Urgent clinical symptom pattern (+25)")
            elif urgency == TRIAGE_ROUTINE:
                base_score += 5.0

            if triage_output.get("safety_override"):
                is_override = True
                contributing_factors.append("Deterministic clinical safety override engaged")

        # 2. Vital Signs Contribution
        if vital_signs and isinstance(vital_signs, dict):
            spo2 = vital_signs.get("SpO2")
            if spo2 is not None:
                if float(spo2) < 90.0:
                    base_score += 30.0
                    contributing_factors.append(f"Severe hypoxemia (SpO2 {spo2}% < 90%) (+30)")
                    is_override = True
                elif float(spo2) < 94.0:
                    base_score += 15.0
                    contributing_factors.append(f"Borderline oxygen saturation ({spo2}%) (+15)")

            hr = vital_signs.get("HeartRate")
            if hr is not None:
                if float(hr) > 130.0 or float(hr) < 45.0:
                    base_score += 20.0
                    contributing_factors.append(f"Hemodynamic heart rate abnormality ({hr} bpm) (+20)")

            sys_bp = vital_signs.get("SystolicBP")
            if sys_bp is not None:
                if float(sys_bp) < 85.0 or float(sys_bp) > 200.0:
                    base_score += 25.0
                    contributing_factors.append(f"Severe blood pressure derangement ({sys_bp} mmHg) (+25)")
                    is_override = True

        # 3. Laboratory Contribution
        if lab_output:
            if lab_output.get("has_critical"):
                crit_count = lab_output.get("critical_count", 1)
                lab_boost = min(40.0, 25.0 + 10.0 * crit_count)
                base_score += lab_boost
                contributing_factors.append(f"{crit_count} critical laboratory panic value(s) flagged (+{int(lab_boost)})")
                is_override = True
            elif lab_output.get("abnormal_count", 0) > 0:
                abn = lab_output["abnormal_count"]
                base_score += min(15.0, 5.0 * abn)
                contributing_factors.append(f"{abn} abnormal out-of-range lab result(s) (+{min(15, 5*abn)})")

        # 4. Chest Radiograph Contribution
        if xray_output and xray_output.get("success"):
            abnormalities = xray_output.get("detected_abnormalities", [])
            if abnormalities:
                if "Pneumothorax" in abnormalities:
                    base_score += 30.0
                    contributing_factors.append("Radiographic evidence of Pneumothorax (+30)")
                    is_override = True
                else:
                    xray_boost = min(20.0, 10.0 * len(abnormalities))
                    base_score += xray_boost
                    contributing_factors.append(f"Radiographic abnormalities ({', '.join(abnormalities)}) (+{int(xray_boost)})")

        # 5. Age-related vulnerability
        if patient_age >= 75 or patient_age < 2:
            base_score += 10.0
            contributing_factors.append(f"Vulnerable age group ({patient_age} yo) (+10)")

        # Absolute Safety Guarantee: Emergency overrides force score to >= 85
        if is_override:
            base_score = max(base_score, 85.0)

        composite_score = int(min(100.0, round(base_score)))

        # Prioritization Tiers
        if composite_score >= 80:
            priority_tier = "Emergency Priority (Red)"
            action_code = "STAT - Immediate Resuscitation / Physician Bedside Evaluation Required"
        elif composite_score >= 55:
            priority_tier = "Urgent Priority (Orange)"
            action_code = "URGENT - Clinician assessment recommended within 30-60 minutes"
        elif composite_score >= 35:
            priority_tier = "Guarded Priority (Yellow)"
            action_code = "SEMI-URGENT - Evaluation within 2-4 hours; monitor serial vitals"
        else:
            priority_tier = "Routine Priority (Green)"
            action_code = "ROUTINE - Standard queue; follow-up as clinical workflow permits"

        # 10-Second Explanation
        finding_str = f"Composite Risk Score: {composite_score}/100 [{priority_tier}]."
        reason_str = (
            f"Primary risk drivers: {'; '.join(contributing_factors)}."
            if contributing_factors else
            "All evaluated physiological, laboratory, and screening modalities fall within non-acute boundaries."
        )
        conf_str = "Deterministic multi-modal aggregation matrix (100% rule-level transparency)."
        limitation_str = (
            "Risk score is a prioritization aid; does not supersede clinical judgment, bedside examination, "
            "or direct physician assessment."
        )

        return {
            "composite_risk_score": composite_score,
            "priority_tier": priority_tier,
            "action_code": action_code,
            "contributing_factors": contributing_factors,
            "safety_override_active": is_override,
            "explanation": {
                "finding": finding_str,
                "reason": reason_str,
                "confidence": conf_str,
                "limitation": limitation_str
            }
        }
