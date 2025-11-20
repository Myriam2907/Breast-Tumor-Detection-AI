from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prediction import predict_breast_tumor

from sqlalchemy.orm import Session
from datetime import timedelta
import os
import uuid
from typing import Optional

from database import engine, get_db, Base
from models import Doctor, Scan
from auth import (
    verify_password, get_password_hash, create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES, get_current_doctor
)
from prediction import predict_breast_tumor, load_model
from pdf_generator import generate_medical_report

# ------------------------
# Database setup
# ------------------------
Base.metadata.create_all(bind=engine)

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Breast Tumor Detection API")

# ------------------------
# CORS
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Directories
# ------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ------------------------
# Serve frontend
# ------------------------
app.mount("/app", StaticFiles(directory="front-end", html=True), name="front-end")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------
# Load model on startup
# ------------------------
@app.on_event("startup")
async def startup_event():
    load_model()
    print("✅ Model loaded successfully!")

# ============================================================
# AUTH ENDPOINTS
# ============================================================
@app.post("/signup")
async def signup(
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    license_number: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_doctor = db.query(Doctor).filter(Doctor.email == email).first()
    if existing_doctor:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    existing_license = db.query(Doctor).filter(Doctor.license_number == license_number).first()
    if existing_license:
        raise HTTPException(status_code=400, detail="License number already registered")
    
    hashed_password = get_password_hash(password)
    new_doctor = Doctor(
        email=email,
        full_name=full_name,
        license_number=license_number,
        hashed_password=hashed_password
    )
    
    db.add(new_doctor)
    db.commit()
    db.refresh(new_doctor)
    
    return {"message": "Doctor registered successfully", "doctor_id": new_doctor.id}


@app.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    doctor = db.query(Doctor).filter(Doctor.email == form_data.username).first()
    
    if not doctor or not verify_password(form_data.password, doctor.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": doctor.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "doctor": {
            "id": doctor.id,
            "email": doctor.email,
            "full_name": doctor.full_name,
            "license_number": doctor.license_number
        }
    }


@app.get("/me")
async def get_current_user(current_doctor: Doctor = Depends(get_current_doctor)):
    return {
        "id": current_doctor.id,
        "email": current_doctor.email,
        "full_name": current_doctor.full_name,
        "license_number": current_doctor.license_number
    }

# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict")
async def predict_tumor(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_id: str = Form(...),
    notes: Optional[str] = Form(None),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    unique_id = str(uuid.uuid4())
    image_filename = f"{unique_id}_original.jpg"
    gradcam_filename = f"{unique_id}_gradcam.jpg"

    image_path = os.path.join("uploads", image_filename)
    gradcam_path = os.path.join("uploads", gradcam_filename)

    image_bytes = await file.read()

    try:
        result = await predict_breast_tumor(image_bytes, image_path, gradcam_path)
        prediction = result["prediction"]
        confidence = result["confidence"]
        malignant_percentage = result["malignant_percentage"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # -------------------------------
    # Save scan in database (FIXED)
    # -------------------------------
    new_scan = Scan(
        doctor_id=current_doctor.id,
        patient_name=patient_name,
        patient_id=patient_id,
        image_path=image_path,
        prediction=prediction,
        confidence=confidence,
        malignant_percentage=malignant_percentage,   # ✔ ADDED
        gradcam_path=gradcam_path,
        notes=notes
    )

    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    # -------------------------------
    # Generate PDF report
    # -------------------------------
    pdf_path = os.path.join("reports", f"Report_{new_scan.id}.pdf")
    generate_medical_report({
        "patient_name": patient_name,
        "patient_id": patient_id,
        "scan_date": new_scan.scan_date.strftime("%Y-%m-%d %H:%M:%S"),
        "doctor_name": current_doctor.full_name,
        "prediction": prediction,
        "confidence": confidence,
        "malignant_percentage": malignant_percentage,
        "image_path": image_path,
        "gradcam_path": gradcam_path,
        "notes": notes
    }, pdf_path)

    return {
        "scan_id": new_scan.id,
        "prediction": prediction,
        "confidence": confidence,
        "malignant_percentage": malignant_percentage,
        "image_url": f"/uploads/{image_filename}",
        "gradcam_url": f"/uploads/{gradcam_filename}",
        "patient_name": patient_name,
        "patient_id": patient_id,
        "scan_date": new_scan.scan_date.isoformat(),
        "pdf_url": f"/download-report/{new_scan.id}"
    }

# ============================================================
# DOWNLOAD REPORT ENDPOINT
# ============================================================
@app.get("/download-report/{scan_id}")
async def download_report(scan_id: str):
    pdf_path = f"reports/Report_{scan_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=pdf_path, filename=f"Report_{scan_id}.pdf", media_type='application/pdf')

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
