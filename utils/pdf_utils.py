# utils/pdf_utils.py
import os
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import cv2
import uuid
from config import REPORTS_DIR

def generate_pdf(patient_name, prediction, confidence, gradcam_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Breast Tumor Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}", ln=True)

    # Save gradcam to temporary file
    grad_file = os.path.join(REPORTS_DIR, f"gradcam_{uuid.uuid4().hex}.png")
    cv2.imwrite(grad_file, gradcam_img)

    # Insert image (fit to width)
    pdf.ln(8)
    try:
        pdf.image(grad_file, x=20, w=170)
    except Exception:
        pass

    # Output PDF to bytes buffer
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    # Remove temp grad file
    try:
        if os.path.exists(grad_file):
            os.remove(grad_file)
    except Exception:
        pass

    return pdf_buffer
