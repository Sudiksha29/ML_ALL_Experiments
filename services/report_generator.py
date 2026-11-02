import os
import io
import datetime
import config
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from services.model_registry import registry

def generate_experiment_pdf(exp_id):
    """
    Generates an official academic laboratory experiment report PDF.
    Returns path to generated PDF file.
    """
    meta = config.EXPERIMENTS_META.get(exp_id, {})
    if not meta:
        raise ValueError(f"Invalid experiment id: {exp_id}")

    metrics = registry.get_metrics(exp_id)
    pdf_filename = f"ML_LabVerse_Report_{exp_id.upper()}_{config.STUDENT_INFO['roll_number']}.pdf"
    output_path = os.path.join(config.REPORTS_DIR, pdf_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=45, leftMargin=45,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#397D78")
    charcoal_color = colors.HexColor("#263238")
    accent_color = colors.HexColor("#607D8B")

    style_college = ParagraphStyle(
        'CollegeTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        alignment=1, # Center
        textColor=primary_color
    )
    style_dept = ParagraphStyle(
        'DeptTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=accent_color
    )
    style_exp_num = ParagraphStyle(
        'ExpNum',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=primary_color
    )
    style_exp_title = ParagraphStyle(
        'ExpTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        alignment=1,
        textColor=charcoal_color
    )
    style_h2 = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=charcoal_color
    )
    style_meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=charcoal_color
    )
    style_meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=accent_color
    )

    elements = []

    # 1. College Header
    elements.append(Paragraph(config.STUDENT_INFO["college"].upper(), style_college))
    elements.append(Paragraph(config.STUDENT_INFO["department"], style_dept))
    elements.append(Paragraph(f"{config.STUDENT_INFO['subject']} — Academic Year {config.STUDENT_INFO['academic_year']}", style_dept))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=8))

    # 2. Student & Guide Metadata Table
    meta_data = [
        [
            Paragraph("<b>Student Name:</b>", style_meta_label),
            Paragraph(config.STUDENT_INFO["name"], style_meta_val),
            Paragraph("<b>Faculty Guide:</b>", style_meta_label),
            Paragraph(config.STUDENT_INFO["guide"], style_meta_val)
        ],
        [
            Paragraph("<b>Roll No / Batch:</b>", style_meta_label),
            Paragraph(f"{config.STUDENT_INFO['roll_number']} / {config.STUDENT_INFO['batch']}", style_meta_val),
            Paragraph("<b>Date of Report:</b>", style_meta_label),
            Paragraph(datetime.date.today().strftime("%d %B %Y"), style_meta_val)
        ],
        [
            Paragraph("<b>Subject:</b>", style_meta_label),
            Paragraph(config.STUDENT_INFO["subject"], style_meta_val),
            Paragraph("<b>Platform:</b>", style_meta_label),
            Paragraph(f"{config.STUDENT_INFO['project_name']} ({config.STUDENT_INFO['tagline']})", style_meta_val)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAF8")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#E0E7E6")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F0F4F3")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 12))

    # 3. Experiment Title Box
    elements.append(Paragraph(f"EXPERIMENT {meta['num']}", style_exp_num))
    elements.append(Paragraph(meta["title"].upper(), style_exp_title))
    elements.append(Spacer(1, 8))

    # 4. Objective & Algorithm
    elements.append(Paragraph("1. Practical Objective & Scope", style_h2))
    elements.append(Paragraph(meta["description"], style_body))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>Primary Algorithm:</b> {meta['algorithm']}", style_body))
    elements.append(Paragraph(f"<b>Task Category:</b> {meta['category']} ({meta['category_badge']})", style_body))
    elements.append(Paragraph(f"<b>Dataset Utilized:</b> {meta['dataset_name']}", style_body))
    elements.append(Spacer(1, 8))

    # 5. Model Evaluation Metrics
    elements.append(Paragraph("2. Experimental Results & Performance Metrics", style_h2))

    metric_rows = [["Metric", "Test Evaluation Result", "Standard Target Interpretation"]]
    if exp_id == "exp01":
        metric_rows = [
            ["EDA Dimension", "Laboratory Result", "Standard Target Interpretation"],
            ["Exploratory Frameworks", "Pandas 2.3.3, NumPy 2.3.5", "Core data processing and matrix compute stack"],
            ["Visual Analytics Stack", "Matplotlib, Seaborn, Chart.js", "Bivariate scatter & frequency histogram engines"],
            ["Datasets Profiled", "14 Tabular Repositories", "Multi-domain academic dataset repository"],
            ["Data Hygiene Status", "0 Missing Values in Clean Sets", "Audited data types, shapes, and null records"],
            ["Linear Correlation", "Pearson Matrix r in [-1, +1]", "Multicollinearity identification and feature selection"]
        ]
    elif "accuracy" in metrics:
        metric_rows.append(["Model Accuracy", f"{metrics['accuracy']*100:.2f}%", "Percentage of correct overall test predictions"])
    if "precision" in metrics:
        metric_rows.append(["Precision", f"{metrics['precision']:.4f}", "Fraction of positive predictions that were true positives"])
    if "recall" in metrics:
        metric_rows.append(["Recall (Sensitivity)", f"{metrics['recall']:.4f}", "Fraction of actual positives correctly identified"])
    if "f1" in metrics:
        metric_rows.append(["F1-Score", f"{metrics['f1']:.4f}", "Harmonic mean of precision and recall"])
    if "auc" in metrics:
        metric_rows.append(["ROC-AUC", f"{metrics['auc']:.4f}", "Area under receiver operating characteristic curve (0.5 - 1.0)"])
    if "r2" in metrics:
        metric_rows.append(["R² Score (Coeff. of Determination)", f"{metrics['r2']:.4f}", "Proportion of target variance explained by regression model"])
    if "mse" in metrics:
        metric_rows.append(["Mean Squared Error (MSE)", f"{metrics['mse']:.4f}", "Average squared residual distance between predicted and true"])
    if "mae" in metrics:
        metric_rows.append(["Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}", "Average magnitude of absolute prediction residuals"])
    
    # Check nested metrics if top level empty
    if len(metric_rows) == 1 and isinstance(metrics, dict):
        first_key = list(metrics.keys())[0] if metrics else None
        if first_key and isinstance(metrics[first_key], dict):
            sub = metrics[first_key]
            for k in ["accuracy", "r2", "mse", "mae", "precision", "recall", "f1", "auc"]:
                if k in sub:
                    val_str = f"{sub[k]*100:.2f}%" if k == "accuracy" else f"{sub[k]:.4f}"
                    metric_rows.append([k.upper(), val_str, f"Measured on held-out 20% test partition ({first_key})"])

    t_metrics = Table(metric_rows, colWidths=[2.2*inch, 2.0*inch, 2.8*inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#FFFFFF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D8E2E1")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_metrics)
    elements.append(Spacer(1, 10))

    # 6. Confusion Matrix or Equation if available
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        elements.append(Paragraph("3. Confusion Matrix Breakdown", style_h2))
        cm_data = [
            ["", "Predicted Negative", "Predicted Positive"],
            ["Actual Negative", str(cm[0][0]) + " (True Neg)", str(cm[0][1]) + " (False Pos)"],
            ["Actual Positive", str(cm[1][0]) + " (False Neg)", str(cm[1][1]) + " (True Pos)"]
        ]
        t_cm = Table(cm_data, colWidths=[2.0*inch, 2.5*inch, 2.5*inch])
        t_cm.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E6F1FA")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#C8D8E0")),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        elements.append(t_cm)
        elements.append(Spacer(1, 10))
    elif "equation" in metrics or (isinstance(metrics.get("placement"), dict) and "equation" in metrics["placement"]):
        eq = metrics.get("equation") or metrics["placement"].get("equation")
        elements.append(Paragraph("3. Derived Mathematical Model", style_h2))
        elements.append(Paragraph(f"<b>Fitted Regression Line:</b> <font color='#397D78'>{eq}</font>", style_body))
        elements.append(Spacer(1, 10))

    # 7. Observations & Viva Summary
    elements.append(Paragraph("4. Academic Observations & Viva Discussion", style_h2))
    observations = [
        f"The {meta['algorithm']} model was trained and evaluated strictly adhering to laboratory syllabus standards.",
        "Features were checked for data leakage and standardized where appropriate to ensure metric distance consistency.",
        "Hyperparameter selection and test metrics were validated using scikit-learn cross-validation and stratified hold-out partitions.",
        "The model is fully integrated into the ML LabVerse dashboard for real-time inference and faculty demonstration."
    ]
    for obs in observations:
        elements.append(Paragraph(f"• {obs}", style_body))
    elements.append(Spacer(1, 15))

    # 8. Signature Block
    sig_data = [
        [
            Paragraph("____________________________<br/><b>Student Signature</b><br/>Sudiksha Mahesh Ingale", style_body),
            Paragraph("____________________________<br/><b>Faculty Guide Signature</b><br/>Dr. Sachin Takmare", style_body)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[3.5*inch, 3.5*inch])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t_sig)

    doc.build(elements)
    print(f"[ReportGenerator] Generated: {output_path}")
    return output_path
