"""
MediAssist - Clinical Guardrails & Safety Engine
Author: Zelvora Industry Labs / MediAssist Team

Provides an independent, rule-based clinical safety net for symptom triage:
  1. ClinicalNarrativeParser:
     - Context-aware NegEx-style parser for negation ("no chest pain"),
       temporality ("had fever last week"), family history ("father had stroke"),
       and hypothetical concern ("worried about heart attack").
     - Distinguishes current patient symptoms from negated, historical, and family mentions.
  2. EmergencySafetyEngine:
     - Evaluates current patient symptoms against high-acuity red flags
       (Manchester Triage System & Emergency Severity Index Tier 1/2 criteria).
     - Evaluates physiological vital signs against acute destabilization boundaries.
     - Deterministic 100% priority override over any statistical ML classification.
  3. UrgentAssessmentEngine:
     - Identifies intermediate-acuity clinical criteria (pneumonia risk, high fever,
       suspected DVT, complicated UTI/pyelonephritis, diabetic decompensation, prolonged severe pain).
  4. InformationSufficiencyEngine:
     - Flags ambiguous, vague, or contradictory input (e.g., bare 'fever and cough' with no context)
       as 'Insufficient Information' before establishing a routine urgency category.
"""
import re
from typing import Dict, Any, List, Optional, Tuple, Set


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE BASE & CLINICAL VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

# Family member references
FAMILY_TERMS = {
    "father", "dad", "mother", "mom", "brother", "sister", "son", "daughter",
    "parent", "parents", "child", "children", "spouse", "wife", "husband",
    "uncle", "aunt", "cousin", "grandfather", "grandmother", "grandpa",
    "grandma", "friend", "colleague", "coworker", "neighbor", "roommate"
}

# Negation triggers
NEGATION_TRIGGERS = [
    r"\bno\b", r"\bnot\b", r"\bdon't\b", r"\bdo not\b", r"\bdoesn't\b",
    r"\bdoes not\b", r"\bdid not\b", r"\bdidn't\b", r"\bdenies\b",
    r"\bdeny\b", r"\bdenying\b", r"\bwithout\b", r"\bfree of\b",
    r"\bruled out\b", r"\bnever had\b", r"\bno history of\b",
    r"\bnegative for\b", r"\bneither\b", r"\bnor\b"
]

# Historical / resolved triggers
HISTORICAL_TRIGGERS = [
    r"\blast week\b", r"\blast month\b", r"\blast year\b", r"\byears ago\b",
    r"\bmonths ago\b", r"\bin the past\b", r"\bpreviously\b",
    r"\bused to have\b", r"\bhistory of\b", r"\bresolved\b",
    r"\bcleared up\b", r"\bwent away\b", r"\bstopped yesterday\b",
    r"\bprior episode\b", r"\bhad\b(?=\s+\w+\s+(?:last|previously|ago))"
]

# Hypothetical / anxiety triggers
HYPOTHETICAL_TRIGGERS = [
    r"\bworried about\b", r"\bworried that\b", r"\bafraid of\b",
    r"\banxious about\b", r"\bfear of\b", r"\bwhat if\b",
    r"\bscared of\b", r"\bconcerned about developing\b",
    r"\bworried i might have\b"
]

# Clinical symptom dictionary for contextual extraction
CLINICAL_SYMPTOMS_CATALOG = {
    "fever": ["fever", "high temperature", "febrile", "pyrexia", "feeling hot", "chills"],
    "cough": ["cough", "coughing", "dry cough", "productive cough", "hacking cough"],
    "chest pain": ["chest pain", "chest pressure", "chest tightness", "substernal pain", "heaviness in chest"],
    "shortness of breath": ["shortness of breath", "dyspnea", "breathless", "difficulty breathing", "cannot breathe", "gasping"],
    "headache": ["headache", "migraine", "head throbbing", "pain in head"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "vertigo", "spinning sensation"],
    "nausea": ["nausea", "nauseated", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "vomited", "throwing up", "emesis"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly ache", "stomach cramps"],
    "diarrhea": ["diarrhea", "loose stools", "watery stools"],
    "fatigue": ["fatigue", "exhaustion", "lethargy", "tiredness", "weakness"],
    "palpitations": ["palpitations", "racing heart", "heart pounding", "irregular heartbeat"],
    "sore throat": ["sore throat", "throat pain", "pain swallowing", "pharyngitis"],
    "rash": ["rash", "skin eruption", "hives", "urticaria", "red spots"],
    "joint pain": ["joint pain", "arthralgia", "swollen joint", "knee pain", "ankle pain"],
    "edema": ["edema", "swelling", "swollen legs", "swollen ankles", "puffy eyes"],
    "hemoptysis": ["coughing blood", "hemoptysis", "blood in sputum"],
    "syncope": ["fainted", "fainting", "passed out", "blackout", "syncope", "collapse"],
    "unconscious": ["unconscious", "unresponsive", "comatose", "cannot wake up"],
    "stroke signs": ["facial droop", "slurred speech", "arm weakness", "hemiparesis", "sudden weakness"],
    "anaphylaxis": ["swollen tongue", "throat closing", "stridor", "anaphylaxis", "lip swelling"],
    "heart attack": ["heart attack", "myocardial infarction", "cardiac arrest", "coronary", "angina"],
    "stroke": ["stroke", "brain attack", "tia", "mini stroke"]
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLINICAL NARRATIVE PARSER (Negation, Temporality, Family, Hypo)
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalNarrativeParser:
    """
    Parses a free-text clinical symptom description into structured symptom sets,
    explicitly accounting for negation, historical status, and family history.
    """

    @classmethod
    def split_into_clauses(cls, text: str) -> List[str]:
        """
        Splits narrative into clause units across sentence terminators,
        commas, and contrasting conjunctions.
        """
        cleaned = re.sub(r"\b(dr|mr|mrs|ms|e\.g|i\.e|approx)\.", r"\1", text, flags=re.IGNORECASE)
        pattern = r"[.!?;,]+|\b(?:but|however|except|although|whereas|and\s+no|with\s+no|without)\b"
        raw_clauses = re.split(pattern, cleaned, flags=re.IGNORECASE)
        clauses = [c.strip() for c in raw_clauses if c and len(c.strip()) > 1]
        return clauses if clauses else [text.strip()]

    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        """
        Performs full contextual extraction on a patient symptom narrative.
        """
        if not text or not isinstance(text, str):
            return {
                "current_symptoms": [],
                "negated_symptoms": [],
                "historical_symptoms": [],
                "family_history": [],
                "hypothetical_concerns": [],
                "clean_patient_text": ""
            }

        clauses = cls.split_into_clauses(text)
        current_syms: Set[str] = set()
        negated_syms: Set[str] = set()
        historical_syms: Set[str] = set()
        family_syms: Set[str] = set()
        hypo_syms: Set[str] = set()
        retained_patient_clauses: List[str] = []

        for clause in clauses:
            clause_lower = clause.lower()
            tokens = set(re.findall(r"\b[a-z]+\b", clause_lower))

            is_family = bool(tokens.intersection(FAMILY_TERMS))
            is_hypothetical = any(re.search(pat, clause_lower) for pat in HYPOTHETICAL_TRIGGERS)
            is_historical = any(re.search(pat, clause_lower) for pat in HISTORICAL_TRIGGERS)
            is_negated = any(re.search(pat, clause_lower) for pat in NEGATION_TRIGGERS)

            # Find matching clinical symptoms in this clause
            clause_matched = []
            for standard_name, synonyms in CLINICAL_SYMPTOMS_CATALOG.items():
                for syn in synonyms:
                    if re.search(r"\b" + re.escape(syn) + r"\b", clause_lower):
                        clause_matched.append(standard_name)
                        break

            for sym in clause_matched:
                if is_family:
                    family_syms.add(sym)
                elif is_hypothetical:
                    hypo_syms.add(sym)
                elif is_negated:
                    negated_syms.add(sym)
                elif is_historical:
                    historical_syms.add(sym)
                else:
                    current_syms.add(sym)

            # Keep only affirmative patient clauses for downstream ML model
            if not is_family and not is_hypothetical and not is_negated and not is_historical:
                retained_patient_clauses.append(clause)

        clean_text = " ".join(retained_patient_clauses).strip()
        if not clean_text:
            clean_text = text.strip()

        return {
            "current_symptoms": sorted(list(current_syms)),
            "negated_symptoms": sorted(list(negated_syms)),
            "historical_symptoms": sorted(list(historical_syms)),
            "family_history": sorted(list(family_syms)),
            "hypothetical_concerns": sorted(list(hypo_syms)),
            "clean_patient_text": clean_text
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. INDEPENDENT EMERGENCY RED-FLAG ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmergencySafetyEngine:
    """
    Deterministic clinical safety net. Evaluates overt life-threatening warning
    signs and acute physiological instability independently of any ML model.
    """

    NARRATIVE_RED_FLAGS = [
        (
            "EMERG-CARD-001",
            "Acute Coronary Syndrome / STEMI",
            [
                r"\b(crushing|radiating|substernal)\s+(?:chest\s+)?(?:pain|pressure|tightness)\b",
                r"\bchest\s+(?:pain|pressure|tightness)\s+(?:radiating\s+to|going\s+to)\s+(?:left\s+arm|jaw|neck|back)\b",
                r"\b(?:severe|crushing)\s+chest\s+(?:pain|pressure)\b",
                r"\bchest\s+pain\s+with\s+(?:cold\s+sweat|sweating|diaphoresis|nausea|shortness of breath)\b",
                r"\btearing\s+chest\s+pain\s+(?:radiating\s+to|through\s+to)\s+back\b"
            ],
            "Acute severe chest pain / suspected acute coronary syndrome or aortic dissection"
        ),
        (
            "EMERG-RESP-001",
            "Acute Respiratory Compromise",
            [
                r"\b(?:severe|acute|extreme)\s+(?:shortness of breath|breathlessness|dyspnea)\b",
                r"\b(?:cannot|unable to|can't)\s+(?:breathe|catch breath|speak in full sentences)\b",
                r"\bgasping\s+for\s+(?:air|breath)\b",
                r"\b(?:stridor|audible wheeze|choking|asphyxia)\b",
                r"\b(?:lips|face|fingers|nails)\s+(?:turning|are)?\s+blue\b",
                r"\b(?:cyanosis|cyanotic)\b"
            ],
            "Critical respiratory distress / impending respiratory failure"
        ),
        (
            "EMERG-NEURO-001",
            "Acute Stroke / FAST Criteria",
            [
                r"\b(?:facial\s+droop|face\s+drooping|one\s+side\s+of\s+face\s+drooping)\b",
                r"\b(?:slurred\s+speech|unable\s+to\s+speak|incoherent\s+speech|loss of speech)\b",
                r"\bsudden\s+(?:arm|leg|one-sided|unilateral)\s+(?:weakness|paralysis|numbness)\b",
                r"\bsudden\s+(?:hemiparesis|hemiplegia)\b"
            ],
            "Positive FAST criteria / suspected acute ischemic stroke or intracranial hemorrhage"
        ),
        (
            "EMERG-CONSC-001",
            "Altered Consciousness / Coma",
            [
                r"\b(?:unconscious|unresponsive|collapsed|in a coma|passed out and won't wake)\b",
                r"\b(?:loss of consciousness|blackout\s+with\s+confusion|found down)\b",
                r"\bactive\s+seizure\b|\bseizure\s+lasting\b|\bstatus epilepticus\b"
            ],
            "Loss of consciousness, collapse, or prolonged seizure activity"
        ),
        (
            "EMERG-ANAPH-001",
            "Anaphylaxis / Airway Compromise",
            [
                r"\b(?:swollen\s+tongue|tongue\s+swelling|throat\s+closing|throat\s+tightness\s+with\s+difficulty\s+swallowing)\b",
                r"\bangioedema\b|\banaphylaxis\b|\banaphylactic\b"
            ],
            "Severe allergic reaction / impending airway obstruction (anaphylaxis)"
        ),
        (
            "EMERG-HEM-001",
            "Massive Hemorrhage",
            [
                r"\b(?:coughing\s+up\s+large\s+(?:amounts|quantities)\s+of\s+blood|massive\s+hemoptysis)\b",
                r"\b(?:vomiting\s+blood|hematemesis|large\s+coffee\s+ground\s+emesis)\b",
                r"\b(?:arterial\s+bleeding|severe\s+uncontrolled\s+bleeding|spurting\s+blood)\b"
            ],
            "Massive active hemorrhage / severe gastrointestinal or respiratory bleeding"
        ),
        (
            "EMERG-HEAD-001",
            "Thunderclap Headache",
            [
                r"\b(?:sudden\s+thunderclap\s+headache|worst\s+headache\s+of\s+(?:my\s+)?life)\b",
                r"\bheadache\s+with\s+(?:stiff\s+neck|confusion|seizure|pupil\s+changes)\b"
            ],
            "Thunderclap headache / suspected subarachnoid hemorrhage or meningitis"
        )
    ]

    @classmethod
    def evaluate(
        cls,
        parsed_context: Dict[str, Any],
        raw_text: str,
        vital_signs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        triggered_rules: List[Dict[str, str]] = []
        vital_red_flags: List[str] = []

        patient_text = parsed_context.get("clean_patient_text", raw_text).lower()

        for rule_id, category, patterns, description in cls.NARRATIVE_RED_FLAGS:
            for pat in patterns:
                match = re.search(pat, patient_text)
                if match:
                    matched_str = match.group(0)
                    neg_window = r"(?:no|not|denies|without|never|ruled out)\s+(?:\w+\s+){0,3}" + re.escape(matched_str)
                    if not re.search(neg_window, raw_text.lower()):
                        reason_msg = f"{category}: {description} (matched '{matched_str}')"
                        reasons.append(reason_msg)
                        triggered_rules.append({
                            "rule_id": rule_id,
                            "category": category,
                            "description": description,
                            "evidence": matched_str
                        })
                        break

        # Evaluate physiological vitals
        if vital_signs and isinstance(vital_signs, dict):
            spo2 = vital_signs.get("SpO2")
            if spo2 is not None:
                try:
                    s_val = float(spo2)
                    if s_val < 90.0:
                        msg = f"Critical hypoxemia: SpO2 {s_val:.0f}% (< 90%)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-SPO2-001", "category": "Severe Hypoxemia", "description": msg, "evidence": f"{s_val}%"})
                except (ValueError, TypeError):
                    pass

            hr = vital_signs.get("HeartRate")
            if hr is not None:
                try:
                    h_val = float(hr)
                    if h_val < 40.0:
                        msg = f"Critical bradycardia: Heart Rate {h_val:.0f} bpm (< 40 bpm)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-HR-001", "category": "Critical Bradycardia", "description": msg, "evidence": f"{h_val} bpm"})
                    elif h_val > 150.0:
                        msg = f"Critical tachycardia: Heart Rate {h_val:.0f} bpm (> 150 bpm)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-HR-002", "category": "Critical Tachycardia", "description": msg, "evidence": f"{h_val} bpm"})
                except (ValueError, TypeError):
                    pass

            sys_bp = vital_signs.get("SystolicBP")
            if sys_bp is not None:
                try:
                    b_val = float(sys_bp)
                    if b_val < 80.0:
                        msg = f"Profound hypotension / shock: Systolic BP {b_val:.0f} mmHg (< 80 mmHg)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-BP-001", "category": "Profound Hypotension", "description": msg, "evidence": f"{b_val} mmHg"})
                    elif b_val > 220.0:
                        msg = f"Hypertensive crisis: Systolic BP {b_val:.0f} mmHg (> 220 mmHg)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-BP-002", "category": "Hypertensive Crisis", "description": msg, "evidence": f"{b_val} mmHg"})
                except (ValueError, TypeError):
                    pass

            rr = vital_signs.get("RespiratoryRate")
            if rr is not None:
                try:
                    r_val = float(rr)
                    if r_val < 8.0 or r_val > 35.0:
                        msg = f"Critical respiratory rate: {r_val:.0f} breaths/min (normal 12-20)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-RR-001", "category": "Respiratory Instability", "description": msg, "evidence": f"{r_val}/min"})
                except (ValueError, TypeError):
                    pass

            temp = vital_signs.get("Temperature")
            if temp is not None:
                try:
                    t_val = float(temp)
                    if t_val < 35.0:
                        msg = f"Severe hypothermia: Body Temperature {t_val:.1f} °C (< 35 °C)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-TEMP-001", "category": "Severe Hypothermia", "description": msg, "evidence": f"{t_val} °C"})
                    elif t_val > 40.5:
                        msg = f"Hyperpyrexia: Body Temperature {t_val:.1f} °C (> 40.5 °C)"
                        vital_red_flags.append(msg)
                        reasons.append(msg)
                        triggered_rules.append({"rule_id": "VITAL-TEMP-002", "category": "Hyperpyrexia", "description": msg, "evidence": f"{t_val} °C"})
                except (ValueError, TypeError):
                    pass

        return {
            "is_emergency": len(reasons) > 0,
            "emergency_reasons": reasons,
            "triggered_rules": triggered_rules,
            "vital_red_flags": vital_red_flags
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. URGENT ASSESSMENT SAFETY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class UrgentAssessmentEngine:
    """
    Evaluates intermediate clinical urgency criteria warranting prompt evaluation (4-12 hrs).
    """

    @classmethod
    def evaluate(
        cls,
        parsed_context: Dict[str, Any],
        severity_score: int,
        duration_days: float,
        trajectory: str = "Unchanged",
        vital_signs: Optional[Dict[str, Any]] = None,
        comorbidities: Optional[List[str]] = None,
        patient_age: int = 35
    ) -> Dict[str, Any]:
        urgent_reasons: List[str] = []
        current_syms = set(parsed_context.get("current_symptoms", []))
        negated_syms = set(parsed_context.get("negated_symptoms", []))
        clean_text = parsed_context.get("clean_patient_text", "").lower()

        # 1. High fever with productive cough / respiratory involvement (Pneumonia screening)
        has_fever = "fever" in current_syms and "fever" not in negated_syms
        has_cough = "cough" in current_syms and "cough" not in negated_syms
        has_pleurisy = bool(re.search(r"\b(?:pleuritic|pain\s+when\s+breathing|stabbing\s+pain\s+on\s+inhalation)\b", clean_text))

        temp_val = None
        if vital_signs and "Temperature" in vital_signs and vital_signs["Temperature"] is not None:
            try:
                temp_val = float(vital_signs["Temperature"])
            except (ValueError, TypeError):
                pass

        if has_fever and (has_cough or has_pleurisy):
            if temp_val and temp_val >= 38.5:
                urgent_reasons.append(f"High fever ({temp_val:.1f} °C) combined with productive respiratory symptoms (suspected pneumonia or lower respiratory infection)")
            elif duration_days >= 3.0 or severity_score >= 6:
                urgent_reasons.append(f"Prolonged fever and cough ({duration_days:.1f} days, severity {severity_score}/10) warranting pulmonary evaluation")

        # 2. Severe Pain criteria (unless contradicted by trivial narrative description)
        trivial_terms = ["mild tickle", "slight scrape", "tiny bump", "paper cut", "feeling fine", "just checking"]
        is_contradictory_trivial = any(t in clean_text for t in trivial_terms)
        if severity_score >= 8 and not is_contradictory_trivial:
            urgent_reasons.append(f"Severe patient-reported symptom pain/intensity ({severity_score}/10)")
        elif severity_score >= 7 and duration_days >= 2.0 and not is_contradictory_trivial:
            urgent_reasons.append(f"Elevated symptom severity ({severity_score}/10) persisting for {duration_days:.1f} days")

        # 3. Trajectory criteria
        if trajectory.lower() in ["worsening rapidly", "rapidly worsening"]:
            urgent_reasons.append("Symptom trajectory is worsening rapidly")

        # 4. Suspected DVT / Pulmonary Embolism risk factors
        if re.search(r"\b(?:swollen|painful|red)\s+(?:calf|leg)\b", clean_text) and (re.search(r"\b(?:flight|travel|surgery|immobilized)\b", clean_text) or "shortness of breath" in current_syms):
            urgent_reasons.append("Unilateral leg swelling with travel/immobility history or mild breathlessness (suspected deep vein thrombosis)")

        # 5. Suspected Acute Pyelonephritis / Complicated UTI
        if re.search(r"\b(?:burning\s+(?:urination|urine)|dysuria)\b", clean_text) and (has_fever or re.search(r"\b(?:flank\s+pain|back\s+pain|chills|rigors)\b", clean_text)):
            urgent_reasons.append("Urinary symptoms accompanied by fever or flank pain (suspected acute pyelonephritis)")

        # 6. Suspected Diabetic Decompensation (DKA / HHS)
        if comorbidities and any("diabetes" in str(c).lower() for c in comorbidities):
            if re.search(r"\b(?:polyuria|excessive\s+thirst|extreme\s+thirst|fruity\s+breath|vomiting|confusion|weakness)\b", clean_text):
                urgent_reasons.append("Diabetic patient presenting with acute osmotic symptoms / vomiting (suspected diabetic ketoacidosis)")

        # 7. Vulnerable demographic risks with acute presentation
        if (patient_age >= 75 or patient_age <= 1) and severity_score >= 6:
            urgent_reasons.append(f"High-risk age demographic ({patient_age} years) with moderate-to-severe clinical complaint")

        # 8. Borderline / Moderate Vital Sign Abnormalities
        if vital_signs and isinstance(vital_signs, dict):
            spo2 = vital_signs.get("SpO2")
            if spo2 is not None:
                try:
                    s_val = float(spo2)
                    if 90.0 <= s_val <= 93.0:
                        urgent_reasons.append(f"Moderate hypoxemia: SpO2 {s_val:.0f}% (90-93% requires prompt clinical assessment)")
                except (ValueError, TypeError):
                    pass

            hr = vital_signs.get("HeartRate")
            if hr is not None:
                try:
                    h_val = float(hr)
                    if 115.0 <= h_val <= 150.0:
                        urgent_reasons.append(f"Significant tachycardia: Heart Rate {h_val:.0f} bpm (115-150 bpm)")
                except (ValueError, TypeError):
                    pass

            bp = vital_signs.get("SystolicBP")
            if bp is not None:
                try:
                    b_val = float(bp)
                    if 180.0 <= b_val <= 220.0:
                        urgent_reasons.append(f"Stage 2 severe hypertension: Systolic BP {b_val:.0f} mmHg")
                except (ValueError, TypeError):
                    pass

        return {
            "is_urgent": len(urgent_reasons) > 0,
            "urgent_reasons": urgent_reasons
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFORMATION SUFFICIENCY & UNCERTAINTY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class InformationSufficiencyEngine:
    """
    Evaluates whether patient information is sufficient to safely establish
    a definitive triage category, or whether uncertainty requires more data.
    """

    @classmethod
    def evaluate(
        cls,
        chief_complaint: str,
        parsed_context: Dict[str, Any],
        severity_score: int,
        duration_days: float,
        vital_signs: Optional[Dict[str, Any]] = None,
        comorbidities: Optional[List[str]] = None,
        model_top_prob: float = 0.0
    ) -> Tuple[bool, List[str]]:
        missing_factors: List[str] = []
        clean_text = parsed_context.get("clean_patient_text", chief_complaint).strip()
        words = re.findall(r"\b[a-zA-Z]+\b", clean_text)

        # 1. Ultra-minimal / vague text
        if len(words) < 3:
            missing_factors.append("Narrative description is too brief (< 3 words) to capture clinical character or onset.")

        # 2. Vague dual presentation with zero metadata (the 'fever and cough' scenario)
        current_syms = set(parsed_context.get("current_symptoms", []))
        if {"fever", "cough"}.issubset(current_syms) or (len(current_syms) >= 2 and len(words) <= 5):
            temp_provided = vital_signs and vital_signs.get("Temperature") is not None
            if not temp_provided and duration_days <= 1.0 and severity_score <= 5:
                missing_factors.append(
                    "Isolated symptom pair ('fever and cough') lacks temperature recording, sputum description, "
                    "or progression context necessary to safely evaluate respiratory risk."
                )

        # 3. Contradiction check: high severity claimed but trivial wording
        trivial_terms = ["mild tickle", "slight scrape", "tiny bump", "paper cut", "feeling fine", "just checking"]
        if severity_score >= 8 and any(t in clean_text.lower() for t in trivial_terms):
            missing_factors.append(f"High severity score ({severity_score}/10) conflicts with mild/trivial narrative description.")

        # 4. Equivocal model confidence
        if 0.0 < model_top_prob < 0.40:
            missing_factors.append(
                f"Statistical classifier confidence ({model_top_prob*100:.1f}%) is equivocal across clinical categories."
            )

        is_sufficient = len(missing_factors) == 0
        return is_sufficient, missing_factors
