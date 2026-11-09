"""
MediAssist — Page 2: Symptom Triage Module
Clinical Safety, AI Accuracy & Explainability Upgrade
Hospital-Grade Clinical Decision Support System (CDSS)
"""
import sys
from pathlib import Path
import io
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_triage_engine, get_explainability_engine
from src.utils import DEMO_PATIENTS
from src.config import EMERGENCY_CONTACT_INDIA, TRIAGE_EMERGENCY, TRIAGE_URGENT, TRIAGE_ROUTINE, TRIAGE_INSUFFICIENT
from src.data_validation import validate_vital_signs

st.set_page_config(
    page_title="MediAssist — Clinical Symptom Triage",
    page_icon="🩺",
    layout="wide"
)

# Custom hospital-grade CSS styling
st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child { display: none; }
    .hospital-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .badge-emergency {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #f87171;
    }
    .badge-urgent {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #fcd34d;
    }
    .badge-routine {
        background-color: #ccfbf1;
        color: #115e59;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #5eead4;
    }
    .badge-insufficient {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #cbd5e1;
    }
    .clinical-quote {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.title("🩺 Clinical Symptom Triage & Safety Screening")
st.caption("Hybrid Decision Support Architecture: Calibrated NLP, Deterministic Safety Overrides & Contextual Linguistic Guardrails")

triage_engine = get_triage_engine()
exp_engine = get_explainability_engine()

# -----------------------------------------------------------------------------
# SESSION STATE HELPER
# -----------------------------------------------------------------------------
def set_triage_case(pt_id_or_custom: str):
    if pt_id_or_custom == "Custom":
        st.session_state["triage_narrative"] = ""
        st.session_state["triage_duration"] = 2.0
        st.session_state["triage_severity"] = 4
        st.session_state["triage_age"] = 35
        st.session_state["triage_sex"] = "Male"
        st.session_state["triage_hr"] = 75
        st.session_state["triage_spo2"] = 98
        st.session_state["triage_bp_sys"] = 120
        st.session_state["triage_bp_dia"] = 80
        st.session_state["triage_rr"] = 16
        st.session_state["triage_temp"] = 37.0
        st.session_state["include_vitals"] = False
        st.session_state["triage_comorbidities"] = []
        st.session_state["triage_medications"] = ""
        st.session_state["triage_risk_factors"] = []
        st.session_state["triage_onset"] = "Sub-acute (1-7 days)"
        st.session_state["triage_trajectory"] = "Unchanged / Static"
        st.session_state["triage_episodes"] = "First time experiencing this"
    else:
        pt = next((p for p in DEMO_PATIENTS if p["id"] == pt_id_or_custom), None)
        if pt:
            st.session_state["triage_narrative"] = pt["chief_complaint"]
            st.session_state["triage_duration"] = float(pt["duration_days"])
            st.session_state["triage_severity"] = int(pt["severity"])
            st.session_state["triage_age"] = int(pt["age"])
            st.session_state["triage_sex"] = pt.get("sex", "Male")
            v = pt.get("vitals", {})
            st.session_state["triage_hr"] = int(v.get("HeartRate", 75))
            st.session_state["triage_spo2"] = int(v.get("SpO2", 98))
            st.session_state["triage_bp_sys"] = int(v.get("SystolicBP", 120))
            st.session_state["triage_bp_dia"] = int(v.get("DiastolicBP", 80))
            st.session_state["triage_rr"] = int(v.get("RespiratoryRate", 16))
            st.session_state["triage_temp"] = float(v.get("Temperature", 37.0))
            st.session_state["include_vitals"] = bool(v)
            # Scenario specific comorbidities
            if pt["id"] == "PT-101":
                st.session_state["triage_comorbidities"] = ["Coronary Artery Disease (CAD)", "Hypertension"]
                st.session_state["triage_medications"] = "Aspirin, Atorvastatin"
                st.session_state["triage_risk_factors"] = ["Smoking / Tobacco use"]
                st.session_state["triage_onset"] = "Sudden (< 1 hour)"
                st.session_state["triage_trajectory"] = "Worsening rapidly"
                st.session_state["triage_episodes"] = "First time experiencing this"
            elif pt["id"] == "PT-102":
                st.session_state["triage_comorbidities"] = ["Asthma / COPD"]
                st.session_state["triage_medications"] = "Albuterol inhaler"
                st.session_state["triage_risk_factors"] = []
                st.session_state["triage_onset"] = "Sub-acute (1-7 days)"
                st.session_state["triage_trajectory"] = "Worsening gradually"
                st.session_state["triage_episodes"] = "First time experiencing this"
            elif pt["id"] == "PT-104":
                st.session_state["triage_comorbidities"] = ["Hypertension", "Prior Stroke / TIA"]
                st.session_state["triage_medications"] = "Amlodipine"
                st.session_state["triage_risk_factors"] = []
                st.session_state["triage_onset"] = "Sudden (< 1 hour)"
                st.session_state["triage_trajectory"] = "Worsening rapidly"
                st.session_state["triage_episodes"] = "First time experiencing this"
            elif pt["id"] == "PT-105":
                st.session_state["triage_comorbidities"] = ["Type 2 Diabetes"]
                st.session_state["triage_medications"] = "Metformin"
                st.session_state["triage_risk_factors"] = []
                st.session_state["triage_onset"] = "Sub-acute (1-7 days)"
                st.session_state["triage_trajectory"] = "Worsening gradually"
                st.session_state["triage_episodes"] = "First time experiencing this"
            else:
                st.session_state["triage_comorbidities"] = []
                st.session_state["triage_medications"] = ""
                st.session_state["triage_risk_factors"] = []
                st.session_state["triage_onset"] = "Sub-acute (1-7 days)"
                st.session_state["triage_trajectory"] = "Unchanged / Static"
                st.session_state["triage_episodes"] = "Recurring episode (known mild condition)"

if "triage_narrative" not in st.session_state:
    set_triage_case("Custom")
    st.session_state["selected_demo_case"] = "Custom Input (Blank)"

# -----------------------------------------------------------------------------
# QUICK PRESET SELECTORS
# -----------------------------------------------------------------------------
st.markdown("##### ⚡ Clinical Demonstration Scenarios")
b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    if st.button("🚨 STEMI / Heart Attack", use_container_width=True):
        set_triage_case("PT-101")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[0]['id']} — {DEMO_PATIENTS[0]['clinical_scenario']}"
        st.rerun()
with b2:
    if st.button("🧠 Acute Stroke", use_container_width=True):
        set_triage_case("PT-104")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[3]['id']} — {DEMO_PATIENTS[3]['clinical_scenario']}"
        st.rerun()
with b3:
    if st.button("🫁 Severe Pneumonia", use_container_width=True):
        set_triage_case("PT-102")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[1]['id']} — {DEMO_PATIENTS[1]['clinical_scenario']}"
        st.rerun()
with b4:
    if st.button("🤧 Viral Cold (Routine)", use_container_width=True):
        set_triage_case("PT-103")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[2]['id']} — {DEMO_PATIENTS[2]['clinical_scenario']}"
        st.rerun()
with b5:
    if st.button("✍️ Clear / Blank Input", use_container_width=True):
        set_triage_case("Custom")
        st.session_state["selected_demo_case"] = "Custom Input (Blank)"
        st.rerun()

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 1: STRUCTURED PATIENT INPUT FORM
# -----------------------------------------------------------------------------
st.subheader("📋 Patient Presentation & Clinical History")

# Collapsible baseline demographics
with st.expander("👤 1. Patient Demographics & Baseline Medical History", expanded=True):
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        patient_age = st.number_input(
            "Patient Age (Years):",
            min_value=0,
            max_value=120,
            value=int(st.session_state.get("triage_age", 35)),
            help="Age is a clinical risk factor for pediatric or geriatric acuity escalations."
        )
    with col_demo2:
        patient_sex = st.selectbox(
            "Biological Sex:",
            ["Male", "Female", "Other", "Unknown"],
            index=["Male", "Female", "Other", "Unknown"].index(st.session_state.get("triage_sex", "Male")),
            help="Used for reference interval and clinical demographic baseline."
        )
    with col_demo3:
        comorbidities = st.multiselect(
            "Known Medical History / Existing Conditions:",
            [
                "Hypertension", "Type 2 Diabetes", "Coronary Artery Disease (CAD)",
                "Asthma / COPD", "Chronic Kidney Disease", "Prior Stroke / TIA",
                "Immunocompromised / Cancer", "Congestive Heart Failure", "None"
            ],
            default=st.session_state.get("triage_comorbidities", []),
            help="Pre-existing conditions can escalate intermediate clinical urgency."
        )
    
    col_med1, col_med2 = st.columns(2)
    with col_med1:
        current_meds = st.text_input(
            "Current Medications (Voluntary):",
            value=st.session_state.get("triage_medications", ""),
            placeholder="e.g., Amlodipine 5mg, Metformin 500mg, Blood thinners / Aspirin",
            help="Medications like anticoagulants increase bleeding risk; insulin indicates diabetes."
        )
    with col_med2:
        risk_factors = st.multiselect(
            "Relevant Clinical Risk Factors:",
            [
                "Smoking / Tobacco use", "Recent long-haul travel / immobility",
                "Recent surgery / major trauma", "Pregnancy", "None"
            ],
            default=st.session_state.get("triage_risk_factors", []),
            help="Recent immobility elevates DVT/PE risk; smoking elevates ACS risk."
        )

# Symptom narrative & characteristics
with st.expander("📝 2. Symptom Information & Chronology", expanded=True):
    chief_complaint = st.text_area(
        "Chief Complaint & Symptom Narrative (in patient's own words):",
        value=st.session_state.get("triage_narrative", ""),
        height=120,
        placeholder="Describe symptoms in detail. Example: 'I have had mild dry cough for 3 days with runny nose, but I do not have chest pain or fever. My father had a heart attack last year.'"
    )
    st.caption("ℹ️ *The NLP engine parses current symptoms while accounting for negation ('no fever'), family history ('father had a stroke'), and historical episodes ('resolved last week').*")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        symptom_onset = st.selectbox(
            "Symptom Onset:",
            ["Sudden (< 1 hour)", "Acute (1-24 hours)", "Sub-acute (1-7 days)", "Gradual / Chronic (> 7 days)"],
            index=["Sudden (< 1 hour)", "Acute (1-24 hours)", "Sub-acute (1-7 days)", "Gradual / Chronic (> 7 days)"].index(st.session_state.get("triage_onset", "Sub-acute (1-7 days)"))
        )
    with col_s2:
        duration_days = st.number_input(
            "Duration (Days):",
            min_value=0.01,
            max_value=365.0,
            value=float(st.session_state.get("triage_duration", 2.0)),
            step=0.5
        )
    with col_s3:
        severity_score = st.slider(
            "Severity Scale (0 = None, 10 = Severe):",
            min_value=0,
            max_value=10,
            value=int(st.session_state.get("triage_severity", 4))
        )
    with col_s4:
        symptom_trajectory = st.selectbox(
            "Symptom Progression / Trajectory:",
            ["Worsening rapidly", "Worsening gradually", "Unchanged / Static", "Improving"],
            index=["Worsening rapidly", "Worsening gradually", "Unchanged / Static", "Improving"].index(st.session_state.get("triage_trajectory", "Unchanged / Static"))
        )

# Vital Signs Section
with st.expander("🩺 3. Recorded Vital Signs (Optional — Enable only if physically measured)", expanded=st.session_state.get("include_vitals", False)):
    enable_vitals = st.checkbox(
        "Vital signs have been measured and recorded",
        value=st.session_state.get("include_vitals", False)
    )
    vitals_dict = None
    if enable_vitals:
        v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
        with v_col1:
            v_temp = st.number_input("Temp (°C):", min_value=30.0, max_value=44.0, value=float(st.session_state.get("triage_temp", 37.0)), step=0.1)
        with v_col2:
            v_hr = st.number_input("Heart Rate (bpm):", min_value=20, max_value=260, value=int(st.session_state.get("triage_hr", 75)), step=1)
        with v_col3:
            v_sys = st.number_input("Systolic BP (mmHg):", min_value=40, max_value=300, value=int(st.session_state.get("triage_bp_sys", 120)), step=1)
        with v_col4:
            v_dia = st.number_input("Diastolic BP (mmHg):", min_value=20, max_value=200, value=int(st.session_state.get("triage_bp_dia", 80)), step=1)
        with v_col5:
            v_spo2 = st.number_input("SpO2 (%):", min_value=40, max_value=100, value=int(st.session_state.get("triage_spo2", 98)), step=1)

        vitals_dict = {
            "Temperature": v_temp,
            "HeartRate": v_hr,
            "SystolicBP": v_sys,
            "DiastolicBP": v_dia,
            "SpO2": v_spo2
        }
        # Validate vitals immediately
        is_v_valid, v_warnings = validate_vital_signs(vitals_dict)
        if not is_v_valid:
            for w in v_warnings:
                st.warning(f"⚠️ Physiological Limit Notice: {w}")

st.markdown("")
run_screening = st.button("Run Clinical Triage Screening 🔍", type="primary", use_container_width=False)

# -----------------------------------------------------------------------------
# RESULTS RENDERER (The 9 Structured Clinical Sections)
# -----------------------------------------------------------------------------
if run_screening:
    if not chief_complaint or not chief_complaint.strip():
        st.warning("⚠️ Please provide a symptom narrative before running screening.")
    else:
        with st.spinner("Executing Hybrid Clinical Triage Architecture (Safety overrides, linguistic guardrails, calibrated NLP)..."):
            res = triage_engine.classify_symptoms(
                chief_complaint=chief_complaint,
                duration_days=duration_days,
                severity_score=severity_score,
                patient_age=patient_age,
                vital_signs=vitals_dict,
                comorbidities=comorbidities,
                current_medications=current_meds,
                symptom_trajectory=symptom_trajectory,
                previous_episodes=st.session_state.get("triage_episodes", "First time"),
                confidence_threshold=0.50
            )

        urgency = res["urgency_category"]
        rep_syms = res["reported_symptoms"]
        red_flags = res["red_flags"]
        urgent_flags = res["urgent_flags"]
        missing_info = res["missing_information"]
        explanation = res["explanation"]
        raw_probs = res["probabilities"]

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 1: Patient Information Summary
        # ---------------------------------------------------------------------
        st.markdown("### 1. Patient Information & Clinical Context")
        pi1, pi2, pi3, pi4 = st.columns(4)
        pi1.metric("Age / Sex", f"{patient_age}y / {patient_sex}")
        pi2.metric("Symptom Duration", f"{duration_days} Day(s)", delta=symptom_onset)
        pi3.metric("Reported Severity", f"{severity_score} / 10", delta=symptom_trajectory, delta_color="inverse" if "worsening" in symptom_trajectory.lower() else "normal")
        pi4.metric("Comorbidities", f"{len(comorbidities)} recorded" if comorbidities else "None")

        # ---------------------------------------------------------------------
        # SECTION 2: Reported Symptoms Analysis
        # ---------------------------------------------------------------------
        st.markdown("### 2. Contextual Symptom Analysis")
        sym_col1, sym_col2, sym_col3 = st.columns(3)
        with sym_col1:
            st.markdown("**🟢 Current Patient Symptoms:**")
            if rep_syms["current"]:
                st.write(", ".join([f"`{s}`" for s in rep_syms["current"]]))
            else:
                st.caption("No specific acute symptom catalog keywords matched.")
        with sym_col2:
            st.markdown("**⚪ Negated / Denied Symptoms:**")
            if rep_syms["negated"]:
                st.write(", ".join([f"`{s}`" for s in rep_syms["negated"]]))
            else:
                st.caption("None reported.")
        with sym_col3:
            st.markdown("**🟣 Historical / Family Context:**")
            hist_items = [f"Past: {s}" for s in rep_syms["historical"]] + [f"Family: {s}" for s in rep_syms["family"]]
            if hist_items:
                st.write(", ".join([f"`{s}`" for s in hist_items]))
            else:
                st.caption("No historical or family references extracted.")

        # ---------------------------------------------------------------------
        # SECTION 3: Safety Screening & Red-Flag Audit
        # ---------------------------------------------------------------------
        st.markdown("### 3. Safety Screening & Red-Flag Audit")
        if red_flags:
            st.error("🚨 **CRITICAL SAFETY FLAGS IDENTIFIED:**")
            for rf in red_flags:
                st.error(f"• **{rf}**")
        elif urgent_flags:
            st.warning("⚠️ **ELEVATED URGENT CRITERIA IDENTIFIED:**")
            for uf in urgent_flags:
                st.warning(f"• {uf}")
        else:
            st.success("✅ **Deterministic Red-Flag Safety Audit:** No overt life-threatening warning signs or acute physiological destabilization detected in current input.")

        # ---------------------------------------------------------------------
        # SECTION 4: Final Urgency Assessment
        # ---------------------------------------------------------------------
        st.markdown("### 4. Final Urgency Assessment")
        if urgency == TRIAGE_EMERGENCY:
            st.error(f"""
            ## 🚨 URGENCY LEVEL: EMERGENCY WARNING
            **CRITICAL ADVISORY:** {res['action_advisory']}  
            📞 **National Emergency Services:** Dial **{EMERGENCY_CONTACT_INDIA}** immediately.
            """)
            if res.get("safety_override"):
                st.info("🛡️ **Deterministic Clinical Safety Override Engaged:** The emergency safety layer enforced an immediate emergency warning based on life-threatening signs, overriding any statistical model output.")
        elif urgency == TRIAGE_URGENT:
            st.warning(f"""
            ## ⚠️ URGENCY LEVEL: URGENT ASSESSMENT
            **Clinical Advisory:** {res['action_advisory']}  
            *Evaluation recommended within 4 to 12 hours at an urgent care center or clinical facility.*
            """)
        elif urgency == TRIAGE_INSUFFICIENT:
            st.info(f"""
            ## ❓ URGENCY LEVEL: INSUFFICIENT INFORMATION
            **Advisory:** {res['action_advisory']}  
            *To ensure patient safety, MediAssist avoids defaulting to a routine categorization when essential diagnostic parameters are unquantified.*
            """)
        else:
            st.success(f"""
            ## 🟢 URGENCY LEVEL: ROUTINE ASSESSMENT
            **Clinical Advisory:** {res['action_advisory']}  
            *Sub-acute, uncomplicated presentation suitable for outpatient primary care follow-up.*
            """)
            st.caption("⚠️ **Safety Notice:** A 'Routine Assessment' status is **not a guarantee that you are safe**. If your condition changes, new symptoms emerge, or breathing difficulties develop, seek immediate emergency medical care.")

        # ---------------------------------------------------------------------
        # SECTION 5: AI-Generated Clinical Decision Support Summary
        # ---------------------------------------------------------------------
        st.markdown("### 5. AI-Generated Clinical Decision Support Summary")
        st.caption("Structured synthesized rationale based on patient input and clinical rules (not a human clinician sign-off):")
        st.markdown(f"""
        <div class="clinical-quote">
            <strong>A. Assessment Result:</strong> {explanation['finding']}<br><br>
            <strong>B. Decision Source:</strong> {res.get('decision_source', 'Hybrid Architecture')}<br><br>
            <strong>C. Clinical Reasoning:</strong> {explanation['reason']}<br><br>
            <strong>D. Unquantified Parameters / Missing Information:</strong> {'; '.join(missing_info) if missing_info else 'None flagged.'}<br><br>
            <strong>E. Confidence Statement:</strong> {explanation['confidence']}<br><br>
            <strong>F. Safety Limitation:</strong> {explanation['limitation']}
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION 6: Model Confidence & Uncertainty
        # ---------------------------------------------------------------------
        st.markdown("### 6. Model Confidence, Probabilities & Uncertainty")
        mc_col1, mc_col2 = st.columns([1, 1])

        with mc_col1:
            st.markdown("**Calibrated Class Probability Distribution:**")
            if raw_probs:
                prob_data = [{"Category": k, "Probability": v} for k, v in raw_probs.items()]
                fig_prob = px.bar(
                    prob_data,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    text_auto=".1%",
                    range_x=[0, 1],
                    color="Category",
                    color_discrete_map={
                        TRIAGE_EMERGENCY: "#ef4444",
                        TRIAGE_URGENT: "#f59e0b",
                        TRIAGE_ROUTINE: "#0d9488"
                    }
                )
                fig_prob.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_prob, use_container_width=True)
                st.caption(f"✓ Probabilities sum strictly to **{sum(raw_probs.values())*100:.1f}%**.")

        with mc_col2:
            st.markdown("**Statistical Model Uncertainty Metrics:**")
            u_entropy = res.get("model_uncertainty", 0.0)
            st.metric(
                "Shannon Uncertainty Entropy",
                f"{u_entropy:.2f} / 1.00",
                help="Normalized Shannon entropy. 0.00 = complete statistical certainty; 1.00 = maximal ambiguity across categories."
            )
            st.write(f"• **Raw Model Top Prediction:** `{res.get('raw_model_class', 'N/A')}` ({res.get('raw_model_confidence', 0.0)*100:.1f}%)")
            st.write(f"• **Final Decision Source:** `{res.get('decision_source', 'Hybrid CDSS')}`")
            st.caption("ℹ️ *Probability estimates are experimental statistical calibration values on clinical datasets; they do not establish empirical biological certainty.*")

        # ---------------------------------------------------------------------
        # SECTION 7: LIME Explainability
        # ---------------------------------------------------------------------
        st.markdown("### 7. Local Model Feature Explainability (LIME)")
        st.caption("Local Interpretable Model-agnostic Explanations: shows which specific words influenced the NLP classifier's internal scoring.")

        if triage_engine.model is not None and triage_engine.vectorizer is not None and chief_complaint:
            with st.spinner("Computing LIME feature attributions for target prediction..."):
                lime_res = exp_engine.explain_text_with_lime(
                    text=chief_complaint,
                    classifier=triage_engine.model,
                    vectorizer=triage_engine.vectorizer,
                    target_class=res.get("raw_model_class", urgency)
                )

            if lime_res and lime_res.get("weights"):
                l_weights = lime_res["weights"]
                l_df = [{"Token": w[0], "Contribution": round(w[1], 4)} for w in l_weights]
                l_df = sorted(l_df, key=lambda x: x["Contribution"])

                fig_lime = go.Figure(go.Bar(
                    x=[d["Contribution"] for d in l_df],
                    y=[d["Token"] for d in l_df],
                    orientation="h",
                    marker=dict(color=["#ef4444" if d["Contribution"] > 0 else "#3b82f6" for d in l_df]),
                    text=[f"{d['Contribution']:+.3f}" for d in l_df],
                    textposition="auto"
                ))
                fig_lime.update_layout(
                    title=f"Token Contributions toward '{lime_res.get('target_class', 'Prediction')}'",
                    height=260,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Feature Weight (Positive = Supports Class, Negative = Opposes Class)"
                )
                st.plotly_chart(fig_lime, use_container_width=True)
                st.caption(f"ℹ️ *{lime_res.get('disclaimer')}*")
            else:
                st.info("Feature importance engine detected no significant vocabulary weights for the current narrative.")
        else:
            st.caption("LIME explanation waiting for model evaluation.")

        # ---------------------------------------------------------------------
        # SECTION 8: Recommended Next Steps
        # ---------------------------------------------------------------------
        st.markdown("### 8. Recommended Clinical Next Steps")
        if urgency == TRIAGE_EMERGENCY:
            st.error("""
            1. **Call 112 immediately** or have someone drive you to the nearest Emergency Department.
            2. **Do not drive yourself.**
            3. If experiencing chest pain or stroke symptoms, sit upright, remain calm, and avoid physical exertion.
            4. Inform emergency personnel of your symptoms, onset time, and current medications.
            """)
        elif urgency == TRIAGE_URGENT:
            st.warning("""
            1. Contact your physician or proceed to an **Urgent Care Center** within 4 to 12 hours.
            2. Monitor temperature, breathing rate, and pulse every 2 hours.
            3. If shortness of breath worsens or chest pain develops, escalate immediately to emergency care (112).
            """)
        elif urgency == TRIAGE_INSUFFICIENT:
            st.info("""
            1. **Measure baseline vitals:** check body temperature with a thermometer and record pulse rate.
            2. **Clarify symptom progression:** note whether symptoms began suddenly or gradually, and whether they are worsening.
            3. Re-run screening with complete details or consult a telemedicine nurse.
            """)
        else:
            st.success("""
            1. Rest, maintain hydration, and monitor symptoms over the next 24 to 48 hours.
            2. Schedule a **routine appointment** with your primary care provider if symptoms do not improve within 3 to 5 days.
            3. Seek prompt reassessment if new symptoms develop (fever > 38.5°C, breathing difficulty, or severe pain).
            """)

        # ---------------------------------------------------------------------
        # SECTION 9: Clinical Limitations & Medical Disclaimer
        # ---------------------------------------------------------------------
        st.markdown("### 9. Clinical Limitations & Medical Disclaimer")
        st.warning("""
        **MEDICAL DISCLAIMER & REGULATORY NOTICE:**  
        MediAssist is an **educational and research clinical decision-support prototype**. It does not provide a definitive medical diagnosis, replace qualified healthcare professionals, or guarantee the detection of medical emergencies. Algorithm predictions and statistical calibrations are experimental aids. If you or someone you are assisting is experiencing serious medical distress, contact local emergency medical services (**112 in India**) or visit the nearest emergency department immediately.
        """)

        # Privacy-preserving summary download
        st.markdown("---")
        summary_text = f"""MEDIASSIST CLINICAL TRIAGE CONSULTATION SUMMARY
Generated: Educational & Research Prototype
==================================================
PATIENT INFORMATION:
- Age: {patient_age} | Sex: {patient_sex}
- Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}
- Current Medications: {current_meds if current_meds else 'None reported'}
- Symptom Duration: {duration_days} day(s) | Severity: {severity_score}/10 | Trajectory: {symptom_trajectory}

REPORTED SYMPTOMS:
- Current Symptoms: {', '.join(rep_syms['current']) if rep_syms['current'] else 'None catalogued'}
- Negated Symptoms: {', '.join(rep_syms['negated']) if rep_syms['negated'] else 'None'}
- Historical/Family: {', '.join(hist_items) if hist_items else 'None'}

ASSESSMENT RESULT:
- Final Urgency Category: {urgency.upper()}
- Decision Source: {res.get('decision_source')}
- Safety Override Active: {res.get('safety_override')}
- Red Flags: {'; '.join(red_flags) if red_flags else 'None'}
- Urgent Flags: {'; '.join(urgent_flags) if urgent_flags else 'None'}

AI-GENERATED DECISION SUPPORT SUMMARY:
{explanation['finding']}
{explanation['reason']}

MODEL CONFIDENCE:
- Raw Calibrated Probabilities: {raw_probs}
- Model Uncertainty Entropy: {res.get('model_uncertainty')}

DISCLAIMER:
MediAssist is an assistive prototype for research. Not a medical diagnosis. In emergencies, dial 112.
==================================================
"""
        st.download_button(
            label="📄 Download Assessment Summary (Privacy-Safe In-Memory)",
            data=summary_text,
            file_name=f"MediAssist_Triage_Summary_{urgency.replace(' ', '_')}.txt",
            mime="text/plain",
            help="Summary is generated in-memory. No patient data is retained on disk or external servers."
        )
