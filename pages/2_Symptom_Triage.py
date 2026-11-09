"""
MediAssist — Page 2: Symptom Triage Module
Natural Language Processing with Deterministic Safety Overrides & 10-Second Explanations
"""
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_triage_engine, get_explainability_engine
from src.utils import DEMO_PATIENTS
from src.config import EMERGENCY_CONTACT_INDIA

st.set_page_config(page_title="MediAssist — Symptom Triage", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 NLP Symptom Triage & Safety Screening")
st.caption("Module A: Calibrated Symptom Urgency Classification with Red-Flag Safety Overrides")

triage_engine = get_triage_engine()
exp_engine = get_explainability_engine()

# Helper to populate session state
def set_triage_case(pt_id_or_custom: str):
    if pt_id_or_custom == "Custom":
        st.session_state["triage_narrative"] = ""
        st.session_state["triage_duration"] = 2.0
        st.session_state["triage_severity"] = 4
        st.session_state["triage_age"] = 35
        st.session_state["triage_hr"] = 75
        st.session_state["triage_spo2"] = 98
        st.session_state["triage_bp"] = 120
        st.session_state["include_vitals"] = False
    else:
        pt = next((p for p in DEMO_PATIENTS if p["id"] == pt_id_or_custom), None)
        if pt:
            st.session_state["triage_narrative"] = pt["chief_complaint"]
            st.session_state["triage_duration"] = float(pt["duration_days"])
            st.session_state["triage_severity"] = int(pt["severity"])
            st.session_state["triage_age"] = int(pt["age"])
            v = pt.get("vitals", {})
            st.session_state["triage_hr"] = int(v.get("HeartRate", 75))
            st.session_state["triage_spo2"] = int(v.get("SpO2", 98))
            st.session_state["triage_bp"] = int(v.get("SystolicBP", 120))
            st.session_state["include_vitals"] = bool(v)

# Initialize session state on first load
if "triage_narrative" not in st.session_state:
    set_triage_case("PT-001")
    st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[0]['id']} — {DEMO_PATIENTS[0]['clinical_scenario']}"

st.markdown("#### ⚡ Quick Demonstration Clinical Scenarios")
st.write("Click any quick preset or type custom symptoms below:")

# Quick Scenario Preset Buttons
btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
with btn_col1:
    if st.button("🚨 STEMI / Heart Attack", use_container_width=True, help="Load 58yo male with acute crushing chest pressure"):
        set_triage_case("PT-001")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[0]['id']} — {DEMO_PATIENTS[0]['clinical_scenario']}"
        st.rerun()
with btn_col2:
    if st.button("🧠 Acute Stroke", use_container_width=True, help="Load 67yo female with unilateral weakness and slurred speech"):
        set_triage_case("PT-005")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[4]['id']} — {DEMO_PATIENTS[4]['clinical_scenario']}"
        st.rerun()
with btn_col3:
    if st.button("🫁 Severe Pneumonia", use_container_width=True, help="Load 42yo female with high fever and productive cough"):
        set_triage_case("PT-002")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[1]['id']} — {DEMO_PATIENTS[1]['clinical_scenario']}"
        st.rerun()
with btn_col4:
    if st.button("🤧 Viral Cold (Routine)", use_container_width=True, help="Load 24yo male with mild runny nose and cough"):
        set_triage_case("PT-004")
        st.session_state["selected_demo_case"] = f"{DEMO_PATIENTS[3]['id']} — {DEMO_PATIENTS[3]['clinical_scenario']}"
        st.rerun()
with btn_col5:
    if st.button("✍️ Custom / Clear", use_container_width=True, help="Clear inputs for manual typing"):
        set_triage_case("Custom")
        st.session_state["selected_demo_case"] = "Custom Input (Blank)"
        st.rerun()

# Full Dropdown Selector
demo_options = [f"{p['id']} — {p['clinical_scenario']}" for p in DEMO_PATIENTS] + ["Custom Input (Blank)"]

def on_dropdown_change():
    selected = st.session_state.get("selected_demo_case", "")
    if selected == "Custom Input (Blank)":
        set_triage_case("Custom")
    else:
        case_id = selected.split(" — ")[0]
        set_triage_case(case_id)

st.selectbox(
    "Or select a patient cohort profile:",
    options=demo_options,
    key="selected_demo_case",
    on_change=on_dropdown_change
)

st.markdown("---")

# Input Section
col_input, col_meta = st.columns([3, 1])

with col_input:
    chief_complaint = st.text_area(
        "Patient Chief Complaint / Symptom Narrative:",
        value=st.session_state.get("triage_narrative", ""),
        height=140,
        placeholder="Type patient symptoms here (e.g. 'Mild headache after studying', 'High fever and productive cough', or 'Crushing central chest pain')..."
    )

with col_meta:
    duration_days = st.number_input(
        "Duration (Days):",
        min_value=0.01,
        max_value=365.0,
        value=float(st.session_state.get("triage_duration", 2.0)),
        step=0.5
    )
    severity_score = st.slider(
        "Severity Scale (1-10):",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get("triage_severity", 4))
    )
    patient_age = st.number_input(
        "Patient Age:",
        min_value=0,
        max_value=120,
        value=int(st.session_state.get("triage_age", 35))
    )

# Optional Vitals - Only included if user explicitly enables it
vitals_dict = None
with st.expander("➕ Physiological Vital Signs (Optional — Enable only if measured)", expanded=st.session_state.get("include_vitals", False)):
    enable_vitals = st.checkbox(
        "Include vital signs in triage evaluation",
        value=st.session_state.get("include_vitals", False)
    )
    
    c_hr, c_spo2, c_bp = st.columns(3)
    with c_hr:
        v_hr = st.number_input(
            "Heart Rate (bpm):",
            min_value=30,
            max_value=250,
            value=int(st.session_state.get("triage_hr", 75)),
            step=1
        )
    with c_spo2:
        v_spo2 = st.number_input(
            "SpO2 (%):",
            min_value=50,
            max_value=100,
            value=int(st.session_state.get("triage_spo2", 98)),
            step=1
        )
    with c_bp:
        v_sys = st.number_input(
            "Systolic BP (mmHg):",
            min_value=50,
            max_value=260,
            value=int(st.session_state.get("triage_bp", 120)),
            step=1
        )
    
    if enable_vitals:
        vitals_dict = {"HeartRate": v_hr, "SpO2": v_spo2, "SystolicBP": v_sys}

run_button = st.button("Run Triage Screening 🔍", type="primary", use_container_width=False)

# Evaluation trigger: run if button clicked or if narrative is non-empty
if run_button or (chief_complaint and chief_complaint.strip()):
    if not chief_complaint or not chief_complaint.strip():
        st.warning("⚠️ Please provide a symptom description or click one of the quick scenario buttons above.")
    else:
        result = triage_engine.classify_symptoms(
            chief_complaint=chief_complaint,
            duration_days=duration_days,
            severity_score=severity_score,
            patient_age=patient_age,
            vital_signs=vitals_dict
        )

        st.markdown("---")

        # Display Priority Results
        urgency = result["urgency_category"]
        if urgency == "Emergency Warning":
            st.error(f"### 🚨 Urgency Category: {urgency.upper()}")
            st.error(f"**CRITICAL ADVISORY:** {result['action_advisory']}")
            if result.get("emergency_contact"):
                st.error(f"📞 Immediate Emergency Contact: Dial **{result['emergency_contact']}** (National Emergency Services)")
        elif urgency == "Urgent Assessment":
            st.warning(f"### ⚠️ Urgency Category: {urgency}")
            st.write(f"**Clinical Advisory:** {result['action_advisory']}")
        elif urgency == "Routine Assessment":
            st.success(f"### 🟢 Urgency Category: {urgency}")
            st.write(f"**Clinical Advisory:** {result['action_advisory']}")
        else:
            st.info(f"### ℹ️ Urgency Category: {urgency}")
            st.write(f"**Advisory:** {result['action_advisory']}")

        if result.get("safety_override"):
            st.error(f"🛡️ **Deterministic Safety Override Engaged:** Flagged critical finding: `{', '.join(result.get('red_flags', []))}`. Statistical probability override enforced to protect patient life.")

        # Identified keywords
        if result.get("identified_symptoms"):
            st.markdown(f"**Identified Clinical Keywords:** `{', '.join(result['identified_symptoms'])}`")

        # Mandatory 10-Second Clinician Explanation
        st.markdown("### ⏱️ Clinician Explanation (10-Second Summary)")
        exp = result["explanation"]
        st.markdown(f"""
        > **Finding:** {exp['finding']}  
        > **Reason:** {exp['reason']}  
        > **Confidence:** {exp['confidence']}  
        > **Limitation:** {exp['limitation']}
        """)

        # Supporting Diagnostics & Explainability
        c_probs, c_lime = st.columns([1, 1])

        with c_probs:
            st.subheader("📊 Class Probability Distribution")
            probs = result.get("probabilities", {})
            if probs:
                prob_df = [{"Category": k, "Probability": v} for k, v in probs.items()]
                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    text_auto=".2%",
                    range_x=[0, 1],
                    color="Category",
                    color_discrete_map={
                        "Emergency Warning": "#ef4444",
                        "Urgent Assessment": "#f59e0b",
                        "Routine Assessment": "#10b981"
                    }
                )
                fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with c_lime:
            st.subheader("🔍 Local Keyword Importance (LIME)")
            if triage_engine.model is not None and triage_engine.vectorizer is not None and chief_complaint:
                with st.spinner("Calculating LIME token attributions..."):
                    lime_res = exp_engine.explain_text_with_lime(chief_complaint, triage_engine.model, triage_engine.vectorizer)
                if lime_res and lime_res.get("weights"):
                    lime_df = [{"Token": w[0], "Weight": round(w[1], 4)} for w in lime_res["weights"]]
                    fig_lime = px.bar(
                        lime_df,
                        x="Weight",
                        y="Token",
                        orientation="h",
                        color="Weight",
                        color_continuous_scale="RdBu_r"
                    )
                    fig_lime.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_lime, use_container_width=True)
                else:
                    st.caption("No significant vocabulary weights detected in current input.")
            else:
                st.caption("Model explanation engine waiting for input.")
