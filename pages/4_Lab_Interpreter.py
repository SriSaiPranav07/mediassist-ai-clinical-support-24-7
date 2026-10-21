"""
MediAssist — Page 4: Laboratory Result Interpreter
Authoritative Reference Intervals (Tietz / Harrison) with Critical Boundary Alerts
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_lab_interpreter
from src.config import LAB_REFERENCE_RANGES
from src.utils import DEMO_PATIENTS

st.set_page_config(page_title="MediAssist — Lab Interpreter", page_icon="🧪", layout="wide")

st.title("🧪 Laboratory Result Interpreter & Critical Value Alerts")
st.caption("Module C: Authoritative Clinical Reference Comparison (Tietz & Harrison Reference Compendium)")

lab_interpreter = get_lab_interpreter()

# Demonstration Case Loader
st.markdown("#### ⚡ Quick Demonstration Lab Panel Pre-filler")
demo_options = ["Custom Entry"] + [f"{p['id']} — {p['clinical_scenario']}" for p in DEMO_PATIENTS]
selected_case = st.selectbox("Load sample laboratory requisition:", demo_options)

prefill_labs = {}
patient_sex = "Male"
patient_age = 45

if selected_case != "Custom Entry":
    case_id = selected_case.split(" — ")[0]
    matched_pt = next((p for p in DEMO_PATIENTS if p["id"] == case_id), None)
    if matched_pt:
        prefill_labs = matched_pt.get("labs", {})
        patient_sex = matched_pt.get("sex", "Male")
        patient_age = int(matched_pt.get("age", 45))

# Demographics Header
c_sex, c_age, c_fast = st.columns(3)
with c_sex:
    sex_input = st.selectbox("Patient Sex (Reference Interval Context):", ["Male", "Female"], index=0 if patient_sex == "Male" else 1)
with c_age:
    age_input = st.number_input("Patient Age (Years):", min_value=0, max_value=125, value=patient_age)
with c_fast:
    fasting_input = st.checkbox("Fasting Requisition (applies to glucose limits)", value=False)

st.markdown("---")
st.subheader("📋 Enter Diagnostic Laboratory Values")

with st.form("lab_form"):
    col1, col2 = st.columns(2)

    form_values = {}
    test_names = list(LAB_REFERENCE_RANGES.keys())
    half = len(test_names) // 2

    with col1:
        for t_name in test_names[:half]:
            spec = LAB_REFERENCE_RANGES[t_name]
            default_val = prefill_labs.get(t_name, {}).get("value", "")
            val = st.text_input(
                f"{t_name} ({spec['unit']}) — Normal: {spec['ranges'].get(sex_input, spec['ranges'].get('All'))}",
                value=str(default_val) if default_val != "" else "",
                key=f"lab_{t_name}"
            )
            form_values[t_name] = {"value": float(val) if val.strip() != "" else None, "unit": spec["unit"]}

    with col2:
        for t_name in test_names[half:]:
            spec = LAB_REFERENCE_RANGES[t_name]
            default_val = prefill_labs.get(t_name, {}).get("value", "")
            val = st.text_input(
                f"{t_name} ({spec['unit']}) — Normal: {spec['ranges'].get(sex_input, spec['ranges'].get('All'))}",
                value=str(default_val) if default_val != "" else "",
                key=f"lab_{t_name}"
            )
            form_values[t_name] = {"value": float(val) if val.strip() != "" else None, "unit": spec["unit"]}

    submit_labs = st.form_submit_button("Interpret Laboratory Panel 🔬", type="primary")

if submit_labs or len(prefill_labs) > 0:
    panel_result = lab_interpreter.interpret_panel(
        panel_dict=form_values,
        patient_sex=sex_input,
        patient_age=age_input,
        fasting=fasting_input
    )

    st.markdown("---")
    # Top KPI summary
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Tests Evaluated", panel_result["total_tests"])
    with k2:
        st.metric("Critical Value Alerts", panel_result["critical_count"], delta="Emergency Protocol" if panel_result["critical_count"] > 0 else None, delta_color="inverse")
    with k3:
        st.metric("Abnormal / Out-of-Range", panel_result["abnormal_count"])
    with k4:
        st.metric("Within Reference Range", panel_result["normal_count"])

    # Critical Alert Callout if present
    if panel_result["has_critical"]:
        st.error("🚨 **CRITICAL LABORATORY ALERT ACTIVATED:** One or more values exceed critical boundaries requiring immediate stat physician notification and protocol intervention.")

    # Detailed Table
    st.subheader("📊 Individual Test Interpretations")
    rows = []
    for t in panel_result["tests"]:
        rows.append({
            "Test Name": t["test_name"],
            "Measured Value": f"{t['value']} {t['unit']}",
            "Status": t["status"],
            "Reference Interval": f"{t['reference_interval'][0]} - {t['reference_interval'][1]} {t['unit']}",
            "Clinical Implication": t["clinical_implication"],
            "Recommended Action": t["recommendation"]
        })

    df_labs = pd.DataFrame(rows)
    st.dataframe(df_labs, use_container_width=True, hide_index=True)

    # 10-Second Clinician Explanations
    st.markdown("### ⏱️ Clinician Laboratory Explanations (10-Second Summaries)")
    for t in panel_result["tests"]:
        if t["status"] != "Normal":
            exp = t["explanation"]
            st.markdown(f"""
            - **{t['test_name']}:** {exp['finding']}  
              *Reason:* {exp['reason']}  
              *Confidence:* {exp['confidence']} | *Limitation:* {exp['limitation']}
            """)
