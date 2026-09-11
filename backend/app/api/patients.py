"""Patient CRUD. Ownership enforced: doctors see only their own patients."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import AuditLog, Doctor, Patient, XRayStudy
from backend.app.schemas import PatientCreate, PatientOut, PatientUpdate
from backend.app.security import get_current_doctor

router = APIRouter(prefix="/patients", tags=["patients"])


def _out(p: Patient, test_count: int = 0) -> PatientOut:
    dob = p.date_of_birth.isoformat() if p.date_of_birth else None
    return PatientOut(
        id=p.id, patient_code=p.patient_code or "", full_name=p.full_name,
        date_of_birth=dob, age=p.age or 0, gender=p.gender or "", phone=p.phone or "",
        email=p.email or "", address=p.address or "", medical_history=p.medical_history or "",
        notes=p.notes or "", test_count=test_count, created_at=p.created_at,
    )


def _owned(db: Session, doctor_id: str, patient_id: str) -> Patient:
    p = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor_id).first()
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@router.get("", response_model=list[PatientOut])
def list_patients(
    q: str = Query(default="", max_length=100),
    include_archived: bool = False,
    db: Session = Depends(get_db),
    doctor: Doctor = Depends(get_current_doctor),
):
    query = db.query(Patient).filter(Patient.doctor_id == doctor.id)
    if not include_archived:
        query = query.filter(Patient.archived.is_(False))
    if q:
        like = f"%{q}%"
        query = query.filter((Patient.full_name.ilike(like)) | (Patient.patient_code.ilike(like)) | (Patient.phone.ilike(like)))
    patients = query.order_by(Patient.created_at.desc()).limit(500).all()
    out = []
    for p in patients:
        n = db.query(XRayStudy).filter(XRayStudy.patient_id == p.id).count()
        out.append(_out(p, n))
    return out


@router.post("", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    dob = None
    if payload.date_of_birth:
        try:
            dob = datetime.strptime(payload.date_of_birth, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="date_of_birth must be YYYY-MM-DD") from exc
    p = Patient(
        doctor_id=doctor.id, patient_code=payload.patient_code.strip(), full_name=payload.full_name.strip(),
        date_of_birth=dob, age=payload.age, gender=payload.gender.strip(), phone=payload.phone.strip(),
        email=payload.email.strip(), address=payload.address, medical_history=payload.medical_history, notes=payload.notes,
    )
    db.add(p)
    db.flush()
    db.add(AuditLog(doctor_id=doctor.id, action="patient_created", entity="patient", entity_id=p.id))
    db.commit()
    db.refresh(p)
    return _out(p, 0)


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    p = _owned(db, doctor.id, patient_id)
    n = db.query(XRayStudy).filter(XRayStudy.patient_id == p.id).count()
    return _out(p, n)


@router.put("/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: str, payload: PatientUpdate, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    p = _owned(db, doctor.id, patient_id)
    data = payload.model_dump(exclude_unset=True)
    if "date_of_birth" in data:
        if data["date_of_birth"]:
            try:
                p.date_of_birth = datetime.strptime(data["date_of_birth"], "%Y-%m-%d").date()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="date_of_birth must be YYYY-MM-DD") from exc
        else:
            p.date_of_birth = None
        del data["date_of_birth"]
    for k, v in data.items():
        setattr(p, k, v.strip() if isinstance(v, str) else v)
    db.add(AuditLog(doctor_id=doctor.id, action="patient_updated", entity="patient", entity_id=p.id))
    db.commit()
    db.refresh(p)
    n = db.query(XRayStudy).filter(XRayStudy.patient_id == p.id).count()
    return _out(p, n)


@router.delete("/{patient_id}")
def archive_patient(patient_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    p = _owned(db, doctor.id, patient_id)
    p.archived = True
    db.add(AuditLog(doctor_id=doctor.id, action="patient_archived", entity="patient", entity_id=p.id))
    db.commit()
    return {"ok": True}
