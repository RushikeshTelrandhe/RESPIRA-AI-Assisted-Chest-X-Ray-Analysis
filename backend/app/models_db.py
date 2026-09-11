"""SQLAlchemy models.

Doctor -> Patient -> XRayStudy -> Analysis (+ Report, AuditLog).
Every medical record carries ownership (doctor_id / patient_id / study_id).
"""

from __future__ import annotations

import uuid
from datetime import datetime, date

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def _uid() -> str:
    return uuid.uuid4().hex


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    license_no: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(50), default="")
    hospital: Mapped[str] = mapped_column(String(200), default="")
    specialization: Mapped[str] = mapped_column(String(200), default="")
    department: Mapped[str] = mapped_column(String(200), default="")
    experience_years: Mapped[int] = mapped_column(Integer, default=0)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    theme: Mapped[str] = mapped_column(String(20), default="light")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patients: Mapped[list["Patient"]] = relationship(back_populates="doctor", cascade="all, delete-orphan")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    doctor_id: Mapped[str] = mapped_column(String(32), ForeignKey("doctors.id"), nullable=False, index=True)
    patient_code: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    age: Mapped[int] = mapped_column(Integer, default=0)
    gender: Mapped[str] = mapped_column(String(20), default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    email: Mapped[str] = mapped_column(String(320), default="")
    address: Mapped[str] = mapped_column(Text, default="")
    medical_history: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    doctor: Mapped[Doctor] = relationship(back_populates="patients")
    studies: Mapped[list["XRayStudy"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class XRayStudy(Base):
    __tablename__ = "xray_studies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    doctor_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(String(32), ForeignKey("patients.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(300), default="")
    image_width: Mapped[int] = mapped_column(Integer, default=0)
    image_height: Mapped[int] = mapped_column(Integer, default=0)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    image_path: Mapped[str] = mapped_column(String(500), default="")  # relative to STORAGE_ROOT
    status: Mapped[str] = mapped_column(String(30), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped[Patient] = relationship(back_populates="studies")
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="study", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    doctor_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    study_id: Mapped[str] = mapped_column(String(32), ForeignKey("xray_studies.id"), nullable=False, index=True)
    primary_class: Mapped[str] = mapped_column(String(100), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sample_uncertainty: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainty_level: Mapped[str] = mapped_column(String(20), default="")
    margin_uncertainty: Mapped[float] = mapped_column(Float, default=0.0)
    probabilities_json: Mapped[str] = mapped_column(Text, default="{}")
    uncertainties_json: Mapped[str] = mapped_column(Text, default="{}")
    disease_weights_json: Mapped[str] = mapped_column(Text, default="{}")
    model_version: Mapped[str] = mapped_column(String(100), default="respira-fusion-1.0")
    device: Mapped[str] = mapped_column(String(50), default="")
    preprocessing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    explainability_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    gradcam_path: Mapped[str] = mapped_column(String(500), default="")
    gradcam_overlay_path: Mapped[str] = mapped_column(String(500), default="")
    vit_attention_path: Mapped[str] = mapped_column(String(500), default="")
    vit_overlay_path: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    study: Mapped[XRayStudy] = relationship(back_populates="analyses")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    doctor_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    analysis_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    pdf_path: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[str] = mapped_column(String(32), default="", index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity: Mapped[str] = mapped_column(String(100), default="")
    entity_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
