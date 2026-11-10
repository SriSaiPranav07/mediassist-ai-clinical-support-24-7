"""
MediAssist - Clinical Consultation PDF Export Engine
Generates downloadable, professionally styled clinical summaries.
Supports ReportLab with robust FPDF2 and text fallbacks for resilient cloud deployment.
Author: MediAssist Clinical AI Team
"""
import io
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL ENGINE IMPORTS (Resilient to Linux / Cloud environments)
# ─────────────────────────────────────────────────────────────────────────────
REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

FPDF_AVAILABLE = False
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPORTLAB IMPLEMENTATION: FULL CLINICAL CONSULTATION PDF
# ─────────────────────────────────────────────────────────────────────────────
def _generate_clinical_summary_reportlab(
    session_data: Dict[str, Any],
    triage_result: Optional[Dict[str, Any]] = None,
    xray_result: Optional[Dict[str, Any]] = None,
    lab_result: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    rag_guidelines: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#991b1b")
    )
    
    story = []
    
    story.append(Paragraph("MediAssist — AI Clinical Decision Support System", title_style))
    consult_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"<b>Consultation Summary & Multi-Model Diagnostic Report</b> | Generated: {consult_time} | <i>Educational & Research Demo Only</i>",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))
    
    # 1. Patient Profile
    story.append(Paragraph("1. Patient Profile & Chief Complaint", section_heading))
    patient_name = session_data.get("patient_name", "Anonymous Consultation")
    age = session_data.get("age", 40)
    sex = session_data.get("sex", "Unspecified")
    duration = session_data.get("duration_days", 1.0)
    severity = session_data.get("severity_score", 5)
    complaint = session_data.get("chief_complaint", "No chief complaint recorded.")
    
    demographics_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Age / Sex:</b> {age} yo / {sex}", body_style),
            Paragraph(f"<b>Duration:</b> {duration} day(s)", body_style),
            Paragraph(f"<b>Severity:</b> {severity}/10", body_style)
        ]
    ]
    t_demo = Table(demographics_data, colWidths=[160, 120, 110, 110])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 4))
    
    t_narrative = Table([[Paragraph(f"<b>Reported Symptom Narrative:</b> {complaint}", body_style)]], colWidths=[540])
    t_narrative.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_narrative)
    story.append(Spacer(1, 8))
    
    # 2. Module A: Triage
    if triage_result:
        story.append(Paragraph("2. Module A — NLP Symptom Triage & Red-Flag Safety", section_heading))
        urgency = triage_result.get("urgency_category", "Pending")
        advisory = triage_result.get("action_advisory", "Review patient history.")
        conf = triage_result.get("confidence", 0.0) * 100.0
        override = "ACTIVE (Safety Rule Engaged)" if triage_result.get("safety_override") else "None (Statistical Mode)"
        
        bg_color = "#fee2e2" if urgency == "Emergency Warning" else ("#fef3c7" if urgency == "Urgent Assessment" else "#dcfce7")
        text_color = "#991b1b" if urgency == "Emergency Warning" else ("#92400e" if urgency == "Urgent Assessment" else "#166534")
        
        triage_table_data = [
            [
                Paragraph(f"<b>Urgency Category:</b> <font color='{text_color}'><b>{urgency.upper()}</b></font>", body_style),
                Paragraph(f"<b>Confidence:</b> {conf:.1f}%", body_style),
                Paragraph(f"<b>Safety Override:</b> {override}", body_style)
            ],
            [
                Paragraph(f"<b>Clinical Advisory:</b> {advisory}", body_style),
                "", ""
            ]
        ]
        t_triage = Table(triage_table_data, colWidths=[200, 140, 200])
        t_triage.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
            ("SPAN", (0, 1), (2, 1)),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_triage)
        story.append(Spacer(1, 8))
    
    # 3. Module B: Chest X-Ray
    if xray_result and xray_result.get("success"):
        story.append(Paragraph("3. Module B — MobileNetV2 Chest X-Ray Abnormality Screening", section_heading))
        abnormalities = xray_result.get("detected_abnormalities", [])
        abn_text = ", ".join(abnormalities) if abnormalities else "No acute focal abnormalities detected above threshold"
        prob_dict = xray_result.get("probabilities", {})
        
        prob_summary = " | ".join([f"{k}: {v*100:.1f}%" for k, v in list(prob_dict.items())[:4]])
        xray_data = [
            [
                Paragraph(f"<b>Screening Findings:</b> {abn_text}", body_style),
                Paragraph(f"<b>Model Probabilities:</b> {prob_summary}", body_style)
            ]
        ]
        t_xray = Table(xray_data, colWidths=[270, 270])
        t_xray.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_xray)
        story.append(Spacer(1, 8))
    
    # 4. Module C: Lab Results
    if lab_result and lab_result.get("tests"):
        story.append(Paragraph("4. Module C — Laboratory Interpretation & Critical Panic Values", section_heading))
        lab_rows = [["Test", "Value", "Ref Interval", "Status", "Clinical Implication"]]
        for t in lab_result["tests"]:
            status_text = t["status"]
            lab_rows.append([
                t["test_name"],
                f"{t['value']} {t['unit']}",
                f"{t['reference_interval'][0]}-{t['reference_interval'][1]} {t['unit']}",
                status_text,
                Paragraph(t["clinical_implication"], body_style)
            ])
        t_labs = Table(lab_rows, colWidths=[70, 75, 95, 75, 225])
        t_labs.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_labs)
        story.append(Spacer(1, 8))
    
    # 5. Module D: Composite Risk
    if risk_result:
        story.append(Paragraph("5. Module D — Composite Patient Risk Prioritization", section_heading))
        score = risk_result.get("composite_risk_score", 0)
        tier = risk_result.get("priority_tier", "Standard")
        factors = risk_result.get("contributing_factors", [])
        factors_text = "; ".join(factors) if factors else "No acute escalating factors."
        
        t_risk = Table([[
            Paragraph(f"<b>Composite Risk Score:</b> <b>{score}/100</b>", body_style),
            Paragraph(f"<b>Priority Tier:</b> <b>{tier}</b>", body_style),
            Paragraph(f"<b>Key Drivers:</b> {factors_text}", body_style)
        ]], colWidths=[130, 120, 290])
        t_risk.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_risk)
        story.append(Spacer(1, 8))
        
    # 6. Module E: RAG Evidence
    if rag_guidelines:
        story.append(Paragraph("6. Grounded Clinical Practice Guidelines (RAG Evidence)", section_heading))
        for gl in rag_guidelines[:2]:
            gl_box = [
                [Paragraph(f"<b>{gl.get('topic', 'Guideline')}:</b> {gl.get('source', '')} ({gl.get('year', '')})", body_style)],
                [Paragraph(f"<i>Recommendation:</i> {gl.get('recommendation', '')}", body_style)]
            ]
            t_gl = Table(gl_box, colWidths=[540])
            t_gl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ]))
            story.append(t_gl)
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 6))
    
    # 7. Clinician Explanation Box
    if triage_result and triage_result.get("explanation"):
        exp = triage_result["explanation"]
        story.append(Paragraph("7. Clinician Decision Explanation", section_heading))
        exp_table = [
            [Paragraph(f"<b>Finding:</b> {exp.get('finding', 'N/A')}", body_style)],
            [Paragraph(f"<b>Reason:</b> {exp.get('reason', 'N/A')}", body_style)],
            [Paragraph(f"<b>Confidence:</b> {exp.get('confidence', 'N/A')}", body_style)],
            [Paragraph(f"<b>Limitation:</b> {exp.get('limitation', 'N/A')}", body_style)],
        ]
        t_exp = Table(exp_table, colWidths=[540])
        t_exp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284c7")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_exp)
        story.append(Spacer(1, 12))
    
    # 8. Disclaimer & Signature
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER & REGULATORY NOTICE:</b> MediAssist is an educational and research prototype. "
        "It is NOT a medical device approved by FDA, CE, or CDSCO. It does not provide definitive medical diagnoses, "
        "triage determinations, or prescription directives. All findings must be reviewed, verified, and signed by a licensed physician."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 10))
    
    sig_data = [
        [
            Paragraph("<b>Reviewing Clinician:</b> ___________________________", body_style),
            Paragraph("<b>License / Reg No:</b> _______________", body_style),
            Paragraph("<b>Date / Signature:</b> ____________________", body_style)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[200, 160, 180])
    t_sig.setStyle(TableStyle([("PADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether(t_sig))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPORTLAB IMPLEMENTATION: MULTIMODAL CHEST X-RAY PDF
# ─────────────────────────────────────────────────────────────────────────────
def _generate_multimodal_xray_reportlab(report: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MMTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "MMSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )
    sec_heading = ParagraphStyle(
        "MMSecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "MMBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    table_head = ParagraphStyle(
        "MMTableHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#ffffff")
    )
    disclaimer_style = ParagraphStyle(
        "MMDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#991b1b")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("MediAssist AI — Multimodal Chest X-Ray & Clinical Decision Support", title_style))
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"<b>Integrated Radiograph & Symptom Consultation Summary</b> | Generated: {gen_time} | <i>Educational & Research Demo</i>",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=8))

    # 2. Decision Status Callout Box
    status = report.get("decision_status", "ROUTINE CLINICIAN REVIEW")
    status_bg = "#fee2e2" if "EMERGENCY" in status else "#fef3c7" if "PROMPT" in status else "#f1f5f9" if "INSUFFICIENT" in status else "#ccfbf1"
    status_border = "#dc2626" if "EMERGENCY" in status else "#d97706" if "PROMPT" in status else "#64748b" if "INSUFFICIENT" in status else "#0d9488"
    
    status_table_data = [
        [Paragraph(f"<b>APPLICATION DECISION STATUS:</b> <font size=11 color='{status_border}'><b>{status}</b></font>", body_style)],
        [Paragraph(f"<b>Clinical Advisory:</b> {report.get('action_advisory', 'N/A')}", body_style)]
    ]
    t_status = Table(status_table_data, colWidths=[540])
    t_status.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(status_bg)),
        ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(status_border)),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_status)
    story.append(Spacer(1, 8))

    # 3. Patient Information & Baseline History
    p_sum = report.get("patient_summary", {})
    story.append(Paragraph("1. Patient Profile, Clinical Chronology & Vitals", sec_heading))
    
    v_data = p_sum.get("vital_signs", {})
    v_str = ", ".join([f"{k}: {v}" for k, v in v_data.items() if v is not None]) if v_data else "Not recorded"
    hist_str = ", ".join(p_sum.get("history", [])) if p_sum.get("history") else "None reported"

    demo_data = [
        [
            Paragraph(f"<b>Age / Sex:</b> {p_sum.get('age', 'N/A')}y / {p_sum.get('sex', 'N/A')}", body_style),
            Paragraph(f"<b>Duration:</b> {p_sum.get('duration_days', 'N/A')} day(s)", body_style),
            Paragraph(f"<b>Reported Severity:</b> {p_sum.get('severity_score', 'N/A')}/10", body_style)
        ],
        [
            Paragraph(f"<b>Symptom Trajectory:</b> {p_sum.get('trajectory', 'Unchanged')}", body_style),
            Paragraph(f"<b>Smoking Status:</b> {p_sum.get('smoking', 'N/A')}", body_style),
            Paragraph(f"<b>Projection View:</b> {report.get('projection_view', 'Unknown')}", body_style)
        ],
        [
            Paragraph(f"<b>Medical History:</b> {hist_str}", body_style),
            Paragraph(f"<b>Allergies / Meds:</b> {p_sum.get('allergies', 'None')} / {p_sum.get('medications', 'None')}", body_style),
            Paragraph(f"<b>Recorded Vitals:</b> {v_str}", body_style)
        ]
    ]
    t_demo = Table(demo_data, colWidths=[180, 180, 180])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 8))

    # 4. Reported Symptoms Breakdown
    story.append(Paragraph("2. Contextual Symptom Evaluation", sec_heading))
    curr_s = ", ".join(p_sum.get("current_symptoms", [])) if p_sum.get("current_symptoms") else "None active"
    neg_s = ", ".join(p_sum.get("negated_symptoms", [])) if p_sum.get("negated_symptoms") else "None"
    narrative = p_sum.get("chief_complaint", "No free narrative provided.")

    sym_data = [
        [Paragraph(f"<b>Current Active Symptoms:</b> {curr_s}", body_style)],
        [Paragraph(f"<b>Explicitly Negated / Ruled-Out:</b> {neg_s}", body_style)],
        [Paragraph(f"<b>Reported Narrative:</b> {narrative}", body_style)]
    ]
    t_sym = Table(sym_data, colWidths=[540])
    t_sym.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sym)
    story.append(Spacer(1, 8))

    # 5. Radiograph Model Findings Table
    story.append(Paragraph("3. Deep Learning Radiograph Analysis (MobileNetV2 CNN)", sec_heading))
    probs = report.get("image_findings", {}).get("probabilities", {})
    detected = report.get("image_findings", {}).get("detected_abnormalities", [])

    rows = [[
        Paragraph("<b>Thoracic Pathology</b>", table_head),
        Paragraph("<b>Probability</b>", table_head),
        Paragraph("<b>Screening Threshold</b>", table_head),
        Paragraph("<b>Detection Status</b>", table_head)
    ]]
    for pathology, prob in probs.items():
        is_det = pathology in detected
        status_lbl = "<font color='#dc2626'><b>ABNORMAL</b></font>" if is_det else "<font color='#16a34a'>Within Baseline</font>"
        rows.append([
            Paragraph(pathology, body_style),
            Paragraph(f"{prob * 100:.1f}%", body_style),
            Paragraph("40.0%", body_style),
            Paragraph(status_lbl, body_style)
        ])
    t_xray = Table(rows, colWidths=[160, 110, 130, 140])
    t_xray.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_xray)
    story.append(Spacer(1, 8))

    # 6. Multimodal Synthesis: Concordance & Discordance
    story.append(Paragraph("4. Multimodal Synthesis (Concordance & Discordance Rules)", sec_heading))
    concordants = report.get("concordance_findings", [])
    discordants = report.get("discordance_alerts", [])

    synth_rows = []
    if concordants:
        for c in concordants:
            synth_rows.append([
                Paragraph(f"<font color='#047857'><b>CONCORDANT:</b> {c['condition']}</font>", body_style),
                Paragraph(c['synthesis'], body_style)
            ])
    if discordants:
        for d in discordants:
            synth_rows.append([
                Paragraph(f"<font color='#b91c1c'><b>DISCORDANT:</b> {d['clinical_concern']}</font>", body_style),
                Paragraph(d['explanation'], body_style)
            ])
    if not synth_rows:
        synth_rows.append([
            Paragraph("<b>Unremarkable Correlation:</b>", body_style),
            Paragraph("No acute concordance or critical discordance flagged between symptoms and radiograph screening.", body_style)
        ])
    
    t_synth = Table(synth_rows, colWidths=[170, 370])
    t_synth.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_synth)
    story.append(Spacer(1, 8))

    # 7. Suggested Next Steps & Missing Information
    story.append(Paragraph("5. Recommended Next Steps & Missing Clinical Parameters", sec_heading))
    steps = report.get("suggested_next_steps", [])
    missing = report.get("missing_information", [])

    steps_p = "<br/>".join([f"• {s}" for s in steps]) if steps else "None specified"
    missing_p = "<br/>".join([f"• {m}" for m in missing]) if missing else "All baseline parameters captured"

    rec_data = [
        [Paragraph("<b>Suggested Non-Prescriptive Clinical Next Steps:</b>", body_style)],
        [Paragraph(steps_p, body_style)],
        [Paragraph("<b>Missing / Recommended Further Diagnostics:</b>", body_style)],
        [Paragraph(missing_p, body_style)]
    ]
    t_rec = Table(rec_data, colWidths=[540])
    t_rec.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 8))

    # 8. Disclaimer & Signature Line
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER & REGULATORY NOTICE:</b> MediAssist AI is an educational and clinical research demonstration. "
        "It does NOT provide a certified medical diagnosis, a formal radiology report, or treatment directives. MobileNetV2 and "
        "Grad-CAM activations represent statistical pattern associations and may produce false positives or false negatives. "
        "A licensed radiologist and attending clinician must independently review the original radiograph alongside the patient's examination."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 8))

    sig_data = [
        [
            Paragraph("<b>Interpreting Physician:</b> ___________________________", body_style),
            Paragraph("<b>License / Reg No:</b> _______________", body_style),
            Paragraph("<b>Signature:</b> ____________________", body_style)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[200, 160, 180])
    t_sig.setStyle(TableStyle([("PADDING", (0, 0), (-1, -1), 2)]))
    story.append(KeepTogether(t_sig))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 3. FPDF2 FALLBACK IMPLEMENTATIONS (Pure Python, Zero C/Linux Lib Dependencies)
# ─────────────────────────────────────────────────────────────────────────────
def _clean_str(s: Any) -> str:
    """Strip XML/HTML tags and non-latin characters for FPDF safe rendering."""
    if s is None:
        return ""
    text = str(s)
    text = re.sub(r"<[^>]+>", "", text)
    # Replace common unicode dashes/quotes
    text = text.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'").replace("•", "*")
    return text.encode("latin-1", "replace").decode("latin-1")


def _generate_clinical_summary_fpdf(
    session_data: Dict[str, Any],
    triage_result: Optional[Dict[str, Any]] = None,
    xray_result: Optional[Dict[str, Any]] = None,
    lab_result: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    rag_guidelines: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, _clean_str("MediAssist - AI Clinical Decision Support System"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(71, 85, 105)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 5, _clean_str(f"Consultation Summary | Generated: {gen_time} | Educational & Research Demo"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # 1. Patient Profile
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "1. Patient Profile & Chief Complaint", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    
    p_name = session_data.get("patient_name", "Anonymous Consultation")
    age = session_data.get("age", 40)
    sex = session_data.get("sex", "Unspecified")
    duration = session_data.get("duration_days", 1.0)
    severity = session_data.get("severity_score", 5)
    complaint = session_data.get("chief_complaint", "No chief complaint recorded.")

    pdf.cell(0, 5, _clean_str(f"Patient: {p_name} | Age/Sex: {age}y / {sex} | Duration: {duration}d | Severity: {severity}/10"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 5, _clean_str(f"Chief Complaint Narrative: {complaint}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # 2. Triage
    if triage_result:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "2. Module A - NLP Symptom Triage", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        urgency = triage_result.get("urgency_category", "Pending")
        conf = triage_result.get("confidence", 0.0) * 100.0
        advisory = triage_result.get("action_advisory", "Review patient history.")
        pdf.cell(0, 5, _clean_str(f"Urgency: {urgency.upper()} (Confidence: {conf:.1f}%)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.multi_cell(0, 5, _clean_str(f"Advisory: {advisory}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # 3. Chest X-Ray
    if xray_result and xray_result.get("success"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "3. Module B - Chest X-Ray Screening (MobileNetV2)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        abn = xray_result.get("detected_abnormalities", [])
        abn_text = ", ".join(abn) if abn else "No acute focal abnormalities above threshold"
        pdf.multi_cell(0, 5, _clean_str(f"Findings: {abn_text}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # 4. Lab Results
    if lab_result and lab_result.get("tests"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "4. Module C - Laboratory Interpretation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        for t in lab_result["tests"]:
            pdf.cell(0, 4.5, _clean_str(f"• {t['test_name']}: {t['value']} {t['unit']} ({t['status']}) - {t['clinical_implication']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # 5. Composite Risk
    if risk_result:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "5. Module D - Composite Risk Prioritization", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        score = risk_result.get("composite_risk_score", 0)
        tier = risk_result.get("priority_tier", "Standard")
        pdf.cell(0, 5, _clean_str(f"Composite Risk: {score}/100 | Tier: {tier}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # Disclaimer
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(153, 27, 27)
    pdf.multi_cell(0, 4, _clean_str(
        "CLINICAL DISCLAIMER: MediAssist is an educational and clinical research demonstration prototype. "
        "It does NOT provide certified medical diagnoses or treatment directives. All outputs must be reviewed "
        "and verified by a licensed healthcare professional."
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


def _generate_multimodal_xray_fpdf(report: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, _clean_str("MediAssist AI - Multimodal Chest X-Ray & Symptom Report"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(71, 85, 105)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 5, _clean_str(f"Generated: {gen_time} | Educational & Clinical Research Demonstration"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Decision Status
    status = report.get("decision_status", "ROUTINE CLINICIAN REVIEW")
    pdf.set_font("Helvetica", "B", 11)
    if "EMERGENCY" in status:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(185, 28, 28)
    elif "PROMPT" in status:
        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(180, 83, 9)
    else:
        pdf.set_fill_color(204, 251, 241)
        pdf.set_text_color(13, 148, 136)

    pdf.cell(0, 8, _clean_str(f"DECISION STATUS: {status}"), border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Advisory
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5, _clean_str(f"Advisory: {report.get('action_advisory', 'N/A')}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 1. Patient Profile
    p_sum = report.get("patient_summary", {})
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "1. Patient Profile & Clinical Chronology", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.cell(0, 4.5, _clean_str(
        f"Age/Sex: {p_sum.get('age', 'N/A')}y / {p_sum.get('sex', 'N/A')} | "
        f"Duration: {p_sum.get('duration_days', 'N/A')}d | "
        f"Severity: {p_sum.get('severity_score', 'N/A')}/10 | "
        f"Projection: {report.get('projection_view', 'PA')}"
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    v_data = p_sum.get("vital_signs", {})
    v_str = ", ".join([f"{k}: {v}" for k, v in v_data.items() if v is not None]) if v_data else "None recorded"
    pdf.cell(0, 4.5, _clean_str(f"Vital Signs: {v_str}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    hist_str = ", ".join(p_sum.get("history", [])) if p_sum.get("history") else "None reported"
    pdf.cell(0, 4.5, _clean_str(f"Medical History: {hist_str}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 2. Symptoms
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "2. Contextual Symptom Evaluation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    curr_s = ", ".join(p_sum.get("current_symptoms", [])) if p_sum.get("current_symptoms") else "None active"
    neg_s = ", ".join(p_sum.get("negated_symptoms", [])) if p_sum.get("negated_symptoms") else "None"
    pdf.multi_cell(0, 4.5, _clean_str(f"Active Symptoms: {curr_s}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 4.5, _clean_str(f"Ruled Out / Negated: {neg_s}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 3. Model Findings
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "3. Deep Learning Radiograph Analysis (MobileNetV2 CNN)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    probs = report.get("image_findings", {}).get("probabilities", {})
    detected = report.get("image_findings", {}).get("detected_abnormalities", [])
    for path_name, p_val in probs.items():
        tag = "[ABNORMAL]" if path_name in detected else "[Within Baseline]"
        pdf.cell(0, 4.5, _clean_str(f"• {path_name}: {p_val*100:.1f}% {tag}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 4. Multimodal Synthesis
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "4. Multimodal Synthesis (Concordance & Discordance Rules)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    concordants = report.get("concordance_findings", [])
    discordants = report.get("discordance_alerts", [])
    for c in concordants:
        pdf.multi_cell(0, 4.5, _clean_str(f"[CONCORDANT] {c['condition']}: {c['synthesis']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for d in discordants:
        pdf.multi_cell(0, 4.5, _clean_str(f"[DISCORDANT ALERT] {d['clinical_concern']}: {d['explanation']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if not concordants and not discordants:
        pdf.cell(0, 4.5, _clean_str("No acute concordance or critical discordance flagged between symptoms and radiograph."), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 5. Suggested Next Steps
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "5. Suggested Next Steps & Diagnostics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    for s in report.get("suggested_next_steps", []):
        pdf.cell(0, 4.5, _clean_str(f"• {s}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for m in report.get("missing_information", []):
        pdf.cell(0, 4.5, _clean_str(f"• [Diagnostic Recommended]: {m}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Disclaimer
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(153, 27, 27)
    pdf.multi_cell(0, 3.8, _clean_str(
        "CLINICAL DISCLAIMER & REGULATORY NOTICE: MediAssist AI is an educational and clinical research demonstration. "
        "It does NOT provide a certified medical diagnosis, formal radiology report, or treatment directives. "
        "A licensed radiologist and attending clinician must independently review the original radiograph alongside the patient examination."
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLAIN TEXT UTF-8 FALLBACK (Guaranteed to always work)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_clinical_summary_text(
    session_data: Dict[str, Any],
    triage_result: Optional[Dict[str, Any]] = None,
    xray_result: Optional[Dict[str, Any]] = None,
    lab_result: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    rag_guidelines: Optional[List[Dict[str, Any]]] = None
) -> str:
    lines = [
        "MEDIASSIST AI - CLINICAL CONSULTATION SUMMARY",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "Notice: Educational & Clinical Research Demonstration Only",
        "-" * 60,
        f"Patient: {session_data.get('patient_name', 'Anonymous')} | Age/Sex: {session_data.get('age', 40)} / {session_data.get('sex', 'Unspecified')}",
        f"Duration: {session_data.get('duration_days', 1.0)} days | Severity: {session_data.get('severity_score', 5)}/10",
        f"Chief Complaint: {session_data.get('chief_complaint', 'N/A')}",
        "-" * 60,
    ]
    if triage_result:
        lines.append(f"TRIAGE CATEGORY: {triage_result.get('urgency_category', 'N/A')}")
        lines.append(f"Advisory: {triage_result.get('action_advisory', 'N/A')}")
    if xray_result:
        lines.append(f"X-RAY FINDINGS: {', '.join(xray_result.get('detected_abnormalities', [])) or 'None'}")
    if risk_result:
        lines.append(f"RISK SCORE: {risk_result.get('composite_risk_score', 0)}/100 ({risk_result.get('priority_tier', 'Standard')})")
    lines.append("-" * 60)
    lines.append("DISCLAIMER: For educational and clinical research demonstration only. Not a medical device.")
    return "\n".join(lines)


def _generate_multimodal_xray_text(report: Dict[str, Any]) -> str:
    p_sum = report.get("patient_summary", {})
    lines = [
        "MEDIASSIST AI - MULTIMODAL CHEST X-RAY & CLINICAL REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"DECISION STATUS: {report.get('decision_status', 'N/A')}",
        f"Advisory: {report.get('action_advisory', 'N/A')}",
        "-" * 60,
        f"Patient: {p_sum.get('age', 'N/A')}y / {p_sum.get('sex', 'N/A')} | Duration: {p_sum.get('duration_days', 'N/A')}d | Severity: {p_sum.get('severity_score', 'N/A')}/10",
        f"Active Symptoms: {', '.join(p_sum.get('current_symptoms', []))}",
        f"Negated Symptoms: {', '.join(p_sum.get('negated_symptoms', []))}",
        "-" * 60,
        "RADIOGRAPH FINDINGS:",
    ]
    probs = report.get("image_findings", {}).get("probabilities", {})
    detected = report.get("image_findings", {}).get("detected_abnormalities", [])
    for k, v in probs.items():
        flag = "[ABNORMAL]" if k in detected else ""
        lines.append(f"  • {k}: {v*100:.1f}% {flag}")
    lines.append("-" * 60)
    lines.append("NEXT STEPS:")
    for s in report.get("suggested_next_steps", []):
        lines.append(f"  • {s}")
    lines.append("-" * 60)
    lines.append("DISCLAIMER: Educational and clinical research demonstration only. Not a medical device.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RESILIENT PUBLIC INTERFACES
# ─────────────────────────────────────────────────────────────────────────────
def generate_clinical_summary_pdf(
    session_data: Dict[str, Any],
    triage_result: Optional[Dict[str, Any]] = None,
    xray_result: Optional[Dict[str, Any]] = None,
    lab_result: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    rag_guidelines: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    """
    Exports full clinical consultation PDF.
    Attempts ReportLab first, seamlessly falls back to FPDF2, then text buffer.
    """
    if REPORTLAB_AVAILABLE:
        try:
            return _generate_clinical_summary_reportlab(
                session_data=session_data,
                triage_result=triage_result,
                xray_result=xray_result,
                lab_result=lab_result,
                risk_result=risk_result,
                rag_guidelines=rag_guidelines
            )
        except Exception:
            pass

    if FPDF_AVAILABLE:
        try:
            return _generate_clinical_summary_fpdf(
                session_data=session_data,
                triage_result=triage_result,
                xray_result=xray_result,
                lab_result=lab_result,
                risk_result=risk_result,
                rag_guidelines=rag_guidelines
            )
        except Exception:
            pass

    return _generate_clinical_summary_text(
        session_data=session_data,
        triage_result=triage_result,
        xray_result=xray_result,
        lab_result=lab_result,
        risk_result=risk_result,
        rag_guidelines=rag_guidelines
    ).encode("utf-8")


def generate_multimodal_xray_pdf(report: Dict[str, Any]) -> bytes:
    """
    Exports multimodal chest radiograph consultation PDF.
    Attempts ReportLab first, seamlessly falls back to FPDF2, then text buffer.
    """
    if REPORTLAB_AVAILABLE:
        try:
            return _generate_multimodal_xray_reportlab(report)
        except Exception:
            pass

    if FPDF_AVAILABLE:
        try:
            return _generate_multimodal_xray_fpdf(report)
        except Exception:
            pass

    return _generate_multimodal_xray_text(report).encode("utf-8")
