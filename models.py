# models.py
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    license_number = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    scans = relationship("Scan", back_populates="doctor")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    
    patient_name = Column(String, nullable=False)
    patient_id = Column(String, nullable=False)

    image_path = Column(String, nullable=False)
    gradcam_path = Column(String, nullable=True)

    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    malignant_percentage = Column(Float, nullable=True)

    notes = Column(String, nullable=True)

    scan_date = Column(DateTime, default=datetime.utcnow)
    doctor = relationship("Doctor", back_populates="scans")
