"""Auth routes: signup / login / logout / me."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import AuditLog, Doctor
from backend.app.schemas import AuthResponse, DoctorOut, LoginRequest, SignupRequest
from backend.app.security import create_token, get_current_doctor, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _out(d: Doctor) -> DoctorOut:
    return DoctorOut(
        id=d.id, full_name=d.full_name, license_no=d.license_no, email=d.email,
        phone=d.phone or "", hospital=d.hospital or "", specialization=d.specialization or "",
        department=d.department or "", experience_years=d.experience_years or 0, theme=d.theme or "light",
    )


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    email = payload.email.strip().lower()
    if db.query(Doctor).filter(Doctor.email == email).first():
        raise HTTPException(status_code=400, detail="An account with this email already exists")
    doctor = Doctor(
        full_name=payload.full_name.strip(), license_no=payload.license_no.strip(),
        email=email, phone=payload.phone.strip(), hospital=payload.hospital.strip(),
        specialization=payload.specialization.strip(), password_hash=hash_password(payload.password),
    )
    db.add(doctor)
    db.add(AuditLog(doctor_id=doctor.id, action="signup", entity="doctor", entity_id=doctor.id))
    db.commit()
    db.refresh(doctor)
    return AuthResponse(access_token=create_token(doctor.id), doctor=_out(doctor))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == payload.email.strip().lower()).first()
    if doctor is None or not verify_password(payload.password, doctor.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    db.add(AuditLog(doctor_id=doctor.id, action="login", entity="doctor", entity_id=doctor.id))
    db.commit()
    return AuthResponse(access_token=create_token(doctor.id), doctor=_out(doctor))


@router.post("/logout")
def logout(doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    db.add(AuditLog(doctor_id=doctor.id, action="logout", entity="doctor", entity_id=doctor.id))
    db.commit()
    return {"ok": True}


@router.get("/me", response_model=DoctorOut)
def me(doctor: Doctor = Depends(get_current_doctor)):
    return _out(doctor)
