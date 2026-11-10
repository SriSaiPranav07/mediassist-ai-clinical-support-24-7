"""
MediAssist - Composite Patient Risk Scoring & Transparent Priority Engine
Author: MediAssist Clinical AI Team
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .config import (
    RISK_LEVEL_EMERGENCY,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_STANDARD,
    RISK_LEVEL_INSUFFICIENT,
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
    TRIAGE_INSUFFICIENT
)

class CompositeRiskEngine:
    """
    Transparent Clinical Prioritization Engine:
    Integrates Symptom Triage, Vital Signs, Laboratory Results, and Radiograph Screening.
    Ensures deterministic safety overrides: Emergency triggers cannot be diluted by low scores elsewhere.
    """

    def __init__(self):
        # Weights for composite numerical score (0 - 100 scale)
        self.weights = {
            "symptoms": 30.0,
            "vitals": 25.0,
            "labs": 25.0,
            "radiology": 20.0
        }

    def evaluate_vitals_risk(self, vitals: Optional[Dict[str, Any]]) -> Tuple[float, List[str], bool]:
        """
        Calculates sub-score and triggers for vital signs based on modified early warning criteria.
        """
        if not vitals or not isinstance(vitals, dict):
            return 0.0, [], False

        score = 0.0
        triggers = []
        is_emergency = False

        # SpO2
        spo2 = vitals.get("SpO2")
        if spo2 is not None:
            try:
                val = float(spo2)
                if val < 90.0:
                    score += 25.0
                    triggers.append(f"Severe Hypoxemia (SpO2 {val}% < 90%)")
                    is_emergency = True
                elif val < 94.0:
                    score += 15.0
                    triggers.append(f"Borderline Hypoxemia (SpO2 {val}%)")
            except (ValueError, TypeError):
                pass

        # Heart Rate
        hr = vitals.get("HeartRate")
        if hr is not None:
            try:
                val = float(hr)
                if val > 140 or val < 40:
                    score += 20.0
                    triggers.append(f"Severe Tachy/Bradycardia ({val} bpm)")
                    is_emergency = True
                elif val > 110 or val < 50:
                    score += 10.0
                    triggers.append(f"Abnormal Heart Rate ({val} bpm)")
            except (ValueError, TypeError):
                pass

        # Blood Pressure
        sys = vitals.get("SystolicBP")
        if sys is not None:
            try:
                val = float(sys)
                if val < 80 or val > 200:
                    score += 20.0
                    triggers.append(f"Critical Blood Pressure ({val} mmHg)")
                    is_emergency = True
                elif val < 90 or val > 160:
                    score += 10.0
                    triggers.append(f"Abnormal Systolic BP ({val} mmHg)")
            except (ValueError, TypeError):
                pass

        # Respiratory Rate
        rr = vitals.get("RespiratoryRate")
        if rr is not None:
            try:
                val = float(rr)
                if val > 30 or val < 8:
                    score += 20.0
                    triggers.append(f"Critical Respiratory Rate ({val}/min)")
                    is_emergency = True
                elif val > 24 or val < 10:
                    score += 10.0
                    triggers.append(f"Tachypnea/Bradypnea ({val}/min)")
            except (ValueError, TypeError):
                pass

        # Temperature
        temp = vitals.get("Temperature")
        if temp is not None:
            try:
                val = float(temp)
                if val >= 39.5 or val <= 35.0:
                    score += 15.0
                    triggers.append(f"Severe Hyper/Hypothermia ({val} °C)")
                elif val >= 38.0:
                    score += 8.0
                    triggers.append(f"Pyrexia ({val} °C)")
            except (ValueError, TypeError):
                pass

        normalized_vitals_score = min(score, 25.0)
        return normalized_vitals_score, triggers, is_emergency

    def calculate_patient_risk(
        self,
        patient_id: str,
        triage_output: Optional[Dict[str, Any]] = None,
        lab_output: Optional[Dict[str, Any]] = None,
        xray_output: Optional[Dict[str, Any]] = None,
        vital_signs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Combines all modular inputs into a documented composite risk assessment.
        """
        contributing_factors = []
        emergency_reasons = []
        high_priority_reasons = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check for Insufficient Information
        has_any_data = any([
            triage_output and triage_output.get("urgency_category") != TRIAGE_INSUFFICIENT,
            lab_output and lab_output.get("total_tests", 0) > 0,
            xray_output and xray_output.get("success"),
            vital_signs and len(vital_signs) > 0
        ])

        if not has_any_data:
            return {
                "patient_id": patient_id,
                "priority_level": RISK_LEVEL_INSUFFICIENT,
                "composite_score": 0.0,
                "contributing_factors": ["No valid clinical data provided."],
                "urgency_badge": "Secondary",
                "recommended_action": "Enter symptom history, vital signs, or diagnostic findings.",
                "explanation": {
                    "finding": "Insufficient information to prioritize patient.",
                    "reason": "All clinical modules report missing or incomplete parameters.",
                    "confidence": "N/A",
                    "limitation": "Risk engine cannot synthesize a score without diagnostic input."
                },
                "timestamp": timestamp,
                "sub_scores": {}
            }

        # 1. Symptom Triage Evaluation
        symptom_score = 0.0
        if triage_output:
            cat = triage_output.get("urgency_category")
            if cat == TRIAGE_EMERGENCY or triage_output.get("safety_override"):
                symptom_score = 30.0
                reasons = triage_output.get("red_flags", ["Life-threatening symptoms reported"])
                emergency_reasons.extend(reasons)
                contributing_factors.append(f"Triage Emergency: {', '.join(reasons)}")
            elif cat == TRIAGE_URGENT:
                symptom_score = 20.0
                high_priority_reasons.append("Symptom triage identified urgent clinical urgency.")
                contributing_factors.append("Triage Urgency: Urgent assessment indicated")
            elif cat == TRIAGE_ROUTINE:
                symptom_score = 8.0
                contributing_factors.append("Triage: Routine assessment category")

        # 2. Vital Signs Evaluation
        vitals_score, vitals_triggers, vitals_emergency = self.evaluate_vitals_risk(vital_signs)
        if vitals_emergency:
            emergency_reasons.extend(vitals_triggers)
        elif vitals_triggers:
            high_priority_reasons.extend(vitals_triggers)
        if vitals_triggers:
            contributing_factors.extend([f"Vital Alert: {t}" for t in vitals_triggers])

        # 3. Laboratory Evaluation
        lab_score = 0.0
        if lab_output:
            crit_count = lab_output.get("critical_count", 0)
            abn_count = lab_output.get("abnormal_count", 0)
            if crit_count > 0:
                lab_score = 25.0
                crit_tests = [
                    f"{t['test_name']} ({t['value']} {t['unit']} - {t['status']})"
                    for t in lab_output.get("tests", []) if t.get("is_critical")
                ]
                emergency_reasons.append(f"Critical lab alert: {', '.join(crit_tests)}")
                contributing_factors.append(f"Laboratory: {crit_count} Critical value(s) detected")
            elif abn_count > 0:
                lab_score = min(abn_count * 6.0, 18.0)
                high_priority_reasons.append(f"Laboratory identified {abn_count} abnormal parameter(s).")
                contributing_factors.append(f"Laboratory: {abn_count} abnormal parameter(s)")
            else:
                lab_score = 4.0
                contributing_factors.append("Laboratory: All tested values within reference ranges")

        # 4. Radiograph Evaluation
        xray_score = 0.0
        if xray_output and xray_output.get("success"):
            flagged = xray_output.get("flagged_findings", [])
            probs = xray_output.get("probabilities", {})
            if "Pneumothorax" in flagged and probs.get("Pneumothorax", 0) > 0.60:
                xray_score = 20.0
                emergency_reasons.append(f"Radiograph flagged tension/acute Pneumothorax ({probs.get('Pneumothorax')*100:.1f}%)")
                contributing_factors.append("Chest X-Ray: Elevated Pneumothorax alert")
            elif flagged:
                xray_score = min(len(flagged) * 8.0, 16.0)
                high_priority_reasons.append(f"Radiograph flagged abnormalities: {', '.join(flagged)}")
                contributing_factors.append(f"Chest X-Ray: {', '.join(flagged)} detected")
            else:
                xray_score = 3.0
                contributing_factors.append("Chest X-Ray: Screening did not exceed acute pathology thresholds")

        # Compute Raw Composite Score (0-100)
        raw_composite = symptom_score + vitals_score + lab_score + xray_score
        composite_score = round(min(raw_composite, 100.0), 1)

        # Deterministic Safety Override: If ANY emergency reason exists, patient is EMERGENCY REVIEW
        if len(emergency_reasons) > 0:
            priority = RISK_LEVEL_EMERGENCY
            composite_score = max(composite_score, 85.0)  # Floor score for emergency alerts
            recommended_action = "Immediate physician evaluation, continuous monitoring, and emergency protocol activation."
            plain_reason = f"Immediate life-safety triggers: {'; '.join(emergency_reasons[:3])}."
        elif len(high_priority_reasons) > 0 or composite_score >= 45.0:
            priority = RISK_LEVEL_HIGH
            recommended_action = "Physician review advised within 1-2 hours; expedite diagnostic and treatment workup."
            plain_reason = f"Multiple abnormal findings: {'; '.join(high_priority_reasons[:3])}."
        else:
            priority = RISK_LEVEL_STANDARD
            recommended_action = "Standard clinical queuing; routine observation and physician review as scheduled."
            plain_reason = "No critical alerts or acute physiological decompensation markers flagged."

        # Plain-English 10-Second Explanation
        finding_text = f"Patient Priority: {priority} (Score: {composite_score}/100)."
        confidence_text = "Rule-governed multi-modal clinical aggregation (transparent weights)."
        limitation_text = "Prioritization score assists clinical queue workflow; it does not predict long-term clinical outcome or mortality."

        return {
            "patient_id": patient_id,
            "priority_level": priority,
            "composite_score": composite_score,
            "contributing_factors": contributing_factors,
            "emergency_reasons": emergency_reasons,
            "high_priority_reasons": high_priority_reasons,
            "recommended_action": recommended_action,
            "explanation": {
                "finding": finding_text,
                "reason": plain_reason,
                "confidence": confidence_text,
                "limitation": limitation_text
            },
            "timestamp": timestamp,
            "sub_scores": {
                "symptoms": symptom_score,
                "vitals": vitals_score,
                "labs": lab_score,
                "radiology": xray_score
            }
        }


