"""Pydantic request/response schemas for the clinical API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field

from app.backend.config import CLASS_NAMES  # canonical disease ordering

DISEASES = CLASS_NAMES
DISCLAIMER = (
    "Respira provides AI-assisted analysis for research and clinical decision support. "
    "Results should be reviewed by a qualified medical professional and should not be "
    "considered a definitive diagnosis."
)


# ---------- Auth ----------


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=200)
    license_no: str = Field(min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(default="", max_length=50)
    hospital: str = Field(default="", max_length=200)
    specialization: str = Field(default="", max_length=200)
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class DoctorOut(BaseModel):
    id: str
    full_name: str
    license_no: str
    email: str
    phone: str = ""
    hospital: str = ""
    specialization: str = ""
    department: str = ""
    experience_years: int = 0
    theme: str = "light"


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    doctor: DoctorOut


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    hospital: Optional[str] = None
    specialization: Optional[str] = None
    department: Optional[str] = None
    experience_years: Optional[int] = None
    theme: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


# ---------- Patients ----------


class PatientCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=200)
    patient_code: str = Field(default="", max_length=50)
    date_of_birth: Optional[str] = None  # YYYY-MM-DD
    age: int = Field(default=0, ge=0, le=150)
    gender: str = Field(default="", max_length=20)
    phone: str = Field(default="", max_length=50)
    email: str = Field(default="", max_length=320)
    address: str = ""
    medical_history: str = ""
    notes: str = ""


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    patient_code: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None


class PatientOut(BaseModel):
    id: str
    patient_code: str = ""
    full_name: str
    date_of_birth: Optional[str] = None
    age: int = 0
    gender: str = ""
    phone: str = ""
    email: str = ""
    address: str = ""
    medical_history: str = ""
    notes: str = ""
    test_count: int = 0
    created_at: Optional[datetime] = None


# ---------- Analysis ----------


class DiseaseScore(BaseModel):
    name: str
    probability: float


class AnalyzeResponse(BaseModel):
    analysis_id: str
    patient: dict
    study: dict
    prediction: dict
    diseases: list[DiseaseScore]
    explainability: dict
    timing: dict
    disclaimer: str = DISCLAIMER


class AnalysisDetail(BaseModel):
    analysis_id: str
    patient: dict
    study: dict
    prediction: dict
    diseases: list[DiseaseScore]
    timing: dict
    model: dict
    explainability: dict[str, Any]
    created_at: Optional[datetime] = None
    disclaimer: str = DISCLAIMER


class HistoryRow(BaseModel):
    analysis_id: str
    study_id: str
    patient_id: str
    patient_name: str
    created_at: Optional[datetime] = None
    primary_class: str
    confidence: float
    sample_uncertainty: float
    uncertainty_level: str
    status: str = "completed"
