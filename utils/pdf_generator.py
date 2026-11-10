"""
MediAssist - Clinical Consultation PDF Export Engine
Generates downloadable, professionally styled clinical summaries using ReportLab.
Author: MediAssist Clinical AI Team
"""
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
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


def generate_clinical_summary_pdf(
    session_data: Dict[str, Any],
    triage_result: Optional[Dict[str, Any]] = None,
    xray_result: Optional[Dict[str, Any]] = None,
    lab_result: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    rag_guidelines: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    """
    Compiles full clinical multi-model decision support session into an authoritative PDF report.
    Returns bytes buffer ready for st.download_button.
    """
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
    
    # Custom styles
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
    
    # Header Banner
    story.append(Paragraph("MediAssist — AI Clinical Decision Support System", title_style))
    consult_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"<b>Consultation Summary & Multi-Model Diagnostic Report</b> | Generated: {consult_time} | <i>Educational & Research Demo Only</i>",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))
    
    # Patient Profile Demographics Table
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
    
    # Narrative Box
    t_narrative = Table([[Paragraph(f"<b>Reported Symptom Narrative:</b> {complaint}", body_style)]], colWidths=[540])
    t_narrative.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_narrative)
    story.append(Spacer(1, 8))
    
    # Module A: Triage Assessment
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
    
    # Module B: Chest X-Ray Screening
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
    
    # Module C: Laboratory Results
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
    
    # Module D: Composite Risk Score
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
        
    # Module E: RAG Grounded Clinical Evidence
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
    
    # 10-Second Clinician Summary Box
    if triage_result and triage_result.get("explanation"):
        exp = triage_result["explanation"]
        story.append(Paragraph("7. Mandatory 10-Second Clinician Summary", section_heading))
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
    
    # Attending Physician Verification & Disclaimer
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER & REGULATORY NOTICE:</b> MediAssist is an educational and research prototype. "
        "It is NOT a medical device approved by FDA, CE, or CDSCO. It does not provide definitive medical diagnoses, "
        "triage determinations, or prescription directives. All findings must be reviewed, verified, and signed by a licensed physician."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 10))
    
    # Signature block
    sig_data = [
        [
            Paragraph("<b>Reviewing Clinician:</b> ___________________________", body_style),
            Paragraph("<b>License / Reg No:</b> _______________", body_style),
            Paragraph("<b>Date / Signature:</b> ____________________", body_style)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[200, 160, 180])
    t_sig.setStyle(TableStyle([
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether(t_sig))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_multimodal_xray_pdf(
    report: Dict[str, Any]
) -> bytes:
    """
    Generates a professional clinical PDF report for the Multimodal Chest X-Ray & Symptom Analysis.
    Returns bytes buffer ready for st.download_button.
    """
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
            Paragraph(f"<b>Vital Signs:</b> {v_str}", body_style),
            Paragraph("<b>Diagnostic Setting:</b> Outpatient / Screening", body_style)
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

    # 4. Symptoms Analysis (Current vs Negated)
    story.append(Paragraph("2. Contextual Symptom Analysis", sec_heading))
    curr_s = ", ".join(p_sum.get("current_symptoms", [])) if p_sum.get("current_symptoms") else "None catalogued"
    neg_s = ", ".join(p_sum.get("negated_symptoms", [])) if p_sum.get("negated_symptoms") else "None denied"

    sym_table = [
        [Paragraph("<b>Current Affirmative Symptoms:</b>", body_style), Paragraph(curr_s, body_style)],
        [Paragraph("<b>Negated / Denied Symptoms:</b>", body_style), Paragraph(neg_s, body_style)]
    ]
    t_sym = Table(sym_table, colWidths=[160, 380])
    t_sym.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sym)
    story.append(Spacer(1, 8))

    # 5. Radiograph Model Findings Table
    story.append(Paragraph("3. Chest Radiograph Model Findings (MobileNetV2 CNN)", sec_heading))
    probs = report.get("radiology_probabilities", {})
    r_rows = [[
        Paragraph("<b>Pathology Category</b>", table_head),
        Paragraph("<b>Model Probability</b>", table_head),
        Paragraph("<b>Screening Status</b>", table_head)
    ]]
    for path_name, p_val in probs.items():
        is_flagged = path_name in report.get("flagged_pathologies", [])
        status_txt = f"<font color='{'#dc2626' if is_flagged else '#475569'}'><b>{'FLAGGED' if is_flagged else 'Below Threshold'}</b></font>"
        r_rows.append([
            Paragraph(path_name, body_style),
            Paragraph(f"{p_val*100:.1f}%", body_style),
            Paragraph(status_txt, body_style)
        ])
    t_rad = Table(r_rows, colWidths=[200, 170, 170])
    t_rad.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_rad)
    story.append(Spacer(1, 8))

    # 6. Multimodal Concordance & Discordance Analysis
    story.append(Paragraph("4. Multimodal Concordance & Discordance Synthesis", sec_heading))
    concordants = report.get("concordant_insights", [])
    discordants = report.get("discordant_insights", [])

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
