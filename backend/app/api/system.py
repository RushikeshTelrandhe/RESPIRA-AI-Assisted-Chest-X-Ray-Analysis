"""System routes: health, model status, dashboard stats."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import Analysis, Doctor, Patient
from backend.app.security import get_current_doctor
from backend.app.services.model_manager import get_manager

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    mgr = get_manager()
    return {"status": "ok", "model_ready": mgr.ready, "device": str(mgr.device)}


@router.get("/models/status")
def model_status():
    return get_manager().status()


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    total_patients = db.query(Patient).filter(Patient.doctor_id == doctor.id, Patient.archived.is_(False)).count()
    analyses = db.query(Analysis).filter(Analysis.doctor_id == doctor.id).order_by(Analysis.created_at.desc()).all()
    week_ago = datetime.utcnow() - timedelta(days=7)
    tests_week = sum(1 for a in analyses if a.created_at and a.created_at >= week_ago)
    high = sum(1 for a in analyses if (a.uncertainty_level or "").lower() == "high")
    names = {p.id: p.full_name for p in db.query(Patient).filter(Patient.doctor_id == doctor.id).all()}
    recent = [{
        "analysis_id": a.id, "patient_id": a.patient_id, "patient_name": names.get(a.patient_id, "—"),
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "primary_class": a.primary_class, "confidence": a.confidence,
        "sample_uncertainty": a.sample_uncertainty, "uncertainty_level": a.uncertainty_level, "status": "completed",
    } for a in analyses[:8]]
    mgr = get_manager()
    return {
        "doctor_name": doctor.full_name,
        "stats": {"total_patients": total_patients, "total_tests": len(analyses),
                  "tests_this_week": tests_week, "high_uncertainty": high},
        "recent_tests": recent,
        "model": {"loaded": mgr.ready, "device": str(mgr.device)},
    }
