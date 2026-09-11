"""History + patient-specific history. Ownership enforced, filters/sort/pagination."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import Analysis, Doctor, Patient, XRayStudy
from backend.app.schemas import HistoryRow
from backend.app.security import get_current_doctor

router = APIRouter(tags=["history"])


def _row(a: Analysis, patient_name: str, status: str = "completed") -> HistoryRow:
    return HistoryRow(
        analysis_id=a.id, study_id=a.study_id, patient_id=a.patient_id, patient_name=patient_name,
        created_at=a.created_at, primary_class=a.primary_class, confidence=a.confidence,
        sample_uncertainty=a.sample_uncertainty, uncertainty_level=a.uncertainty_level, status=status,
    )


@router.get("/history", response_model=dict)
def history(
    patient_id: str | None = None,
    disease: str | None = None,
    uncertainty: str | None = None,
    q: str = Query(default="", max_length=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="recent"),
    db: Session = Depends(get_db),
    doctor: Doctor = Depends(get_current_doctor),
):
    query = db.query(Analysis).filter(Analysis.doctor_id == doctor.id)
    if patient_id:
        query = query.filter(Analysis.patient_id == patient_id)
    if disease:
        query = query.filter(Analysis.primary_class == disease)
    if uncertainty:
        query = query.filter(Analysis.uncertainty_level.ilike(uncertainty))
    if sort == "confidence":
        query = query.order_by(Analysis.confidence.desc())
    else:
        query = query.order_by(Analysis.created_at.desc())
    all_rows = query.all()
    # name search (join-free, small scale)
    names = {p.id: p.full_name for p in db.query(Patient).filter(Patient.doctor_id == doctor.id).all()}
    statuses = {s.id: s.status for s in db.query(XRayStudy).filter(XRayStudy.doctor_id == doctor.id).all()}
    if q:
        ql = q.lower()
        all_rows = [a for a in all_rows if ql in names.get(a.patient_id, "").lower() or ql in a.primary_class.lower() or ql in a.id.lower()]
    total = len(all_rows)
    start = (page - 1) * page_size
    items = [_row(a, names.get(a.patient_id, "—"), statuses.get(a.study_id, "completed")) for a in all_rows[start:start + page_size]]
    return {"total": total, "page": page, "page_size": page_size, "items": [i.model_dump() for i in items]}


@router.get("/patients/{patient_id}/history", response_model=list[HistoryRow])
def patient_history(patient_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if patient is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Patient not found")
    analyses = db.query(Analysis).filter(Analysis.doctor_id == doctor.id, Analysis.patient_id == patient_id).order_by(Analysis.created_at.asc()).all()
    statuses = {s.id: s.status for s in db.query(XRayStudy).filter(XRayStudy.patient_id == patient_id).all()}
    return [_row(a, patient.full_name, statuses.get(a.study_id, "completed")) for a in analyses]
