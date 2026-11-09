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

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Laboratory Result Interpreter & Critical Value Alerts")
st.caption("Module C: Authoritative Clinical Reference Comparison (Tietz & Harrison Reference Compendium)")

lab_interpreter = get_lab_interpreter()

# Helper to populate lab session state
def set_lab_case(case_id_or_custom: str):
    if case_id_or_custom == "Custom":
        st.session_state["lab_sex"] = "Male"
        st.session_state["lab_age"] = 45
        st.session_state["lab_fasting"] = False
        for t_name in LAB_REFERENCE_RANGES:
            st.session_state[f"lab_val_{t_name}"] = ""
    else:
        pt = next((p for p in DEMO_PATIENTS if p["id"] == case_id_or_custom), None)
        if pt:
            st.session_state["lab_sex"] = pt.get("sex", "Male")
            st.session_state["lab_age"] = int(pt.get("age", 45))
            st.session_state["lab_fasting"] = False
            labs = pt.get("labs", {})
            for t_name in LAB_REFERENCE_RANGES:
                val = labs.get(t_name, {}).get("value", "")
                st.session_state[f"lab_val_{t_name}"] = str(val) if val != "" else ""

# Initialize default to PT-105 (DKA / Critical Labs) — blank slate on first load
if "lab_val_Glucose" not in st.session_state:
    set_lab_case("Custom")
    st.session_state["selected_lab_demo"] = "Custom Entry (Blank)"

st.markdown("#### ⚡ Quick Demonstration Laboratory Panels")
st.write("Click any quick preset or select a specific patient laboratory requisition:")

# Quick Presets
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🚨 Critical DKA (Glucose 465, K 5.8)", use_container_width=True):
        set_lab_case("PT-105")
        st.session_state["selected_lab_demo"] = f"{DEMO_PATIENTS[4]['id']} — {DEMO_PATIENTS[4]['clinical_scenario']}"
        st.rerun()
with c2:
    if st.button("⚠️ STEMI Labs (Troponin 1.45)", use_container_width=True):
        set_lab_case("PT-101")
        st.session_state["selected_lab_demo"] = f"{DEMO_PATIENTS[0]['id']} — {DEMO_PATIENTS[0]['clinical_scenario']}"
        st.rerun()
with c3:
    if st.button("🟢 Normal / Mild URTI (PT-103)", use_container_width=True):
        set_lab_case("PT-103")
        st.session_state["selected_lab_demo"] = f"{DEMO_PATIENTS[2]['id']} — {DEMO_PATIENTS[2]['clinical_scenario']}"
        st.rerun()
with c4:
    if st.button("✍️ Custom / Clear Labs", use_container_width=True):
        set_lab_case("Custom")
        st.session_state["selected_lab_demo"] = "Custom Entry (Blank)"
        st.rerun()

lab_options = [f"{p['id']} — {p['clinical_scenario']}" for p in DEMO_PATIENTS] + ["Custom Entry (Blank)"]

def on_lab_dropdown_change():
    sel = st.session_state.get("selected_lab_demo", "")
    if sel == "Custom Entry (Blank)":
        set_lab_case("Custom")
    else:
        cid = sel.split(" — ")[0]
        set_lab_case(cid)

st.selectbox(
    "Or choose patient laboratory requisition:",
    options=lab_options,
    key="selected_lab_demo",
    on_change=on_lab_dropdown_change
)

st.markdown("---")

# Demographics Header
c_sex, c_age, c_fast = st.columns(3)
with c_sex:
    sex_input = st.selectbox(
        "Patient Sex (Reference Interval Context):",
        ["Male", "Female"],
        index=0 if st.session_state.get("lab_sex", "Male") == "Male" else 1,
        key="lab_sex"
    )
with c_age:
    age_input = st.number_input(
        "Patient Age (Years):",
        min_value=0,
        max_value=125,
        key="lab_age"
    )
with c_fast:
    fasting_input = st.checkbox(
        "Fasting Requisition (applies to glucose limits)",
        key="lab_fasting"
    )

st.subheader("📋 Enter / Review Diagnostic Laboratory Values")

col1, col2 = st.columns(2)
form_values = {}
test_names = list(LAB_REFERENCE_RANGES.keys())
half = len(test_names) // 2

with col1:
    for t_name in test_names[:half]:
        spec = LAB_REFERENCE_RANGES[t_name]
        normal_range = spec['ranges'].get(sex_input, spec['ranges'].get('All'))
        val_str = st.text_input(
            f"{t_name} ({spec['unit']}) — Normal: {normal_range}",
            key=f"lab_val_{t_name}"
        )
        try:
            val_num = float(val_str.strip()) if val_str and val_str.strip() else None
        except ValueError:
            val_num = None
        form_values[t_name] = {"value": val_num, "unit": spec["unit"]}

with col2:
    for t_name in test_names[half:]:
        spec = LAB_REFERENCE_RANGES[t_name]
        normal_range = spec['ranges'].get(sex_input, spec['ranges'].get('All'))
        val_str = st.text_input(
            f"{t_name} ({spec['unit']}) — Normal: {normal_range}",
            key=f"lab_val_{t_name}"
        )
        try:
            val_num = float(val_str.strip()) if val_str and val_str.strip() else None
        except ValueError:
            val_num = None
        form_values[t_name] = {"value": val_num, "unit": spec["unit"]}

submit_labs = st.button("Interpret Laboratory Panel 🔬", type="primary")

# Execute interpretation if button pressed or if values are present
has_entered_vals = any(v["value"] is not None for v in form_values.values())

if submit_labs or has_entered_vals:
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
        st.metric("Critical Value Alerts", panel_result["critical_count"], delta="Stat Alert" if panel_result["critical_count"] > 0 else None, delta_color="inverse")
    with k3:
        st.metric("Abnormal / Out-of-Range", panel_result["abnormal_count"])
    with k4:
        st.metric("Within Reference Range", panel_result["normal_count"])

    # Critical Alert Callout if present
    if panel_result["has_critical"]:
        st.error("🚨 **CRITICAL LABORATORY ALERT ACTIVATED:** One or more values exceed critical panic boundaries requiring immediate physician notification and protocol intervention.")

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
    non_normal_tests = [t for t in panel_result["tests"] if t["status"] != "Normal"]
    if non_normal_tests:
        for t in non_normal_tests:
            exp = t["explanation"]
            st.markdown(f"""
            - **{t['test_name']}:** {exp['finding']}  
              *Reason:* {exp['reason']}  
              *Confidence:* {exp['confidence']} | *Limitation:* {exp['limitation']}
            """)
    else:
        st.success("All evaluated laboratory tests fall within standard physiological reference intervals.")
