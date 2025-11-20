# utils/pdf_utils.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os

def generate_medical_report(scan_data, output_path):
    """
    Generate a professional BREAST TUMOR medical PDF report
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title Style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#d6336c'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Section headings
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    # Title (updated)
    title = Paragraph("BREAST TUMOR ANALYSIS REPORT", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))

    # Patient Info Table
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Patient Name:', scan_data['patient_name']],
        ['Patient ID:', scan_data['patient_id']],
        ['Scan Date:', scan_data['scan_date']],
        ['Doctor:', scan_data['doctor_name']],
    ]

    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))

    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))

    # Diagnosis Results
    story.append(Paragraph("Diagnosis Results", heading_style))

    result_color = colors.HexColor('#dc3545') if scan_data['prediction'] == 'Malignant:Cancerous tissue detected' else colors.HexColor('#198754')

    result_data = [
        ['Prediction:', scan_data['prediction']],
        ['Confidence:', f"{scan_data['confidence']:.2f}%"],
    ]

    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))

    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))

    # Scan Images
    story.append(Paragraph("Breast Scan Analysis", heading_style))

    # Original image
    if os.path.exists(scan_data['image_path']):
        story.append(Paragraph("Original Scan:", normal_style))
        story.append(Image(scan_data['image_path'], width=2.5*inch, height=2.5*inch))
        story.append(Spacer(1, 0.2*inch))

    # GradCAM
    if os.path.exists(scan_data['gradcam_path']):
        story.append(Paragraph("AI Heatmap (Grad-CAM):", normal_style))
        story.append(Image(scan_data['gradcam_path'], width=2.5*inch, height=2.5*inch))
        story.append(Spacer(1, 0.2*inch))

    # Notes
    if scan_data.get('notes'):
        story.append(Paragraph("Clinical Notes", heading_style))
        story.append(Paragraph(scan_data['notes'], normal_style))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer = Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ParagraphStyle('Footer', fontSize=9, textColor=colors.HexColor('#adb5bd'), alignment=TA_CENTER)
    )
    story.append(footer)

    doc.build(story)
    return output_path
