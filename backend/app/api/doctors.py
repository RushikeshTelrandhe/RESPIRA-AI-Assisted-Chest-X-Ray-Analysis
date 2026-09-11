"""Doctor profile routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.auth import _out
from backend.app.database import get_db
from backend.app.models_db import Doctor
from backend.app.schemas import DoctorOut, PasswordChange, ProfileUpdate
from backend.app.security import get_current_doctor, hash_password, verify_password

router = APIRouter(prefix="/doctors", tags=["doctors"])


@router.get("/profile", response_model=DoctorOut)
def get_profile(doctor: Doctor = Depends(get_current_doctor)):
    return _out(doctor)


@router.put("/profile", response_model=DoctorOut)
def update_profile(payload: ProfileUpdate, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    for field in ("full_name", "phone", "hospital", "specialization", "department", "theme"):
        val = getattr(payload, field)
        if val is not None:
            setattr(doctor, field, val.strip() if isinstance(val, str) else val)
    if payload.experience_years is not None:
        if payload.experience_years < 0 or payload.experience_years > 80:
            raise HTTPException(status_code=400, detail="Invalid experience value")
        doctor.experience_years = payload.experience_years
    db.commit()
    db.refresh(doctor)
    return _out(doctor)


@router.post("/change-password")
def change_password(payload: PasswordChange, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    if not verify_password(payload.current_password, doctor.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    doctor.password_hash = hash_password(payload.new_password)
    db.commit()
    return {"ok": True}
