"""Reports: generate PDF / download PDF / export JSON."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import Analysis, AuditLog, Doctor, Patient, Report, XRayStudy
from backend.app.security import get_current_doctor
from backend.app.config import STORAGE_ROOT
from backend.app.services.report_service import build_pdf
from backend.app.utils.storage import resolve

router = APIRouter(tags=["reports"])


def _owned_analysis(db: Session, doctor_id: str, analysis_id: str) -> Analysis:
    a = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.doctor_id == doctor_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return a


@router.post("/reports/{analysis_id}")
def generate_report(analysis_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    a = _owned_analysis(db, doctor.id, analysis_id)
    study = db.get(XRayStudy, a.study_id)
    patient = db.get(Patient, a.patient_id)
    if study is None or patient is None:
        raise HTTPException(status_code=404, detail="Study or patient for this analysis no longer exists")
    images = {
        "original": resolve(study.image_path) if study else None,
        "gradcam": resolve(a.gradcam_overlay_path),
        "vit": resolve(a.vit_overlay_path),
    }
    try:
        pdf = build_pdf(doctor, patient, study, a, images)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate report: {exc}") from exc
    try:
        rel = pdf.relative_to(STORAGE_ROOT).as_posix()
    except ValueError:
        # build_pdf always writes under STORAGE_ROOT; fall back to absolute
        rel = str(pdf)
    existing = db.query(Report).filter(Report.analysis_id == a.id).first()
    if existing:
        # Always store the path relative to STORAGE_ROOT so resolve() works
        # across machines (older rows may hold absolute paths - still readable).
        existing.pdf_path = rel
    else:
        db.add(Report(doctor_id=doctor.id, analysis_id=a.id, pdf_path=rel))
    db.add(AuditLog(doctor_id=doctor.id, action="report_generated", entity="analysis", entity_id=a.id))
    db.commit()
    return {"ok": True, "analysis_id": a.id, "pdf": f"/api/v1/reports/{a.id}/pdf"}


@router.get("/reports/{analysis_id}/pdf")
def download_pdf(analysis_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    a = _owned_analysis(db, doctor.id, analysis_id)
    rep = db.query(Report).filter(Report.analysis_id == a.id, Report.doctor_id == doctor.id).first()
    pdf = resolve(rep.pdf_path) if rep and rep.pdf_path else None
    if pdf is None or not pdf.exists():  # generate on demand, then persist
        study = db.get(XRayStudy, a.study_id)
        patient = db.get(Patient, a.patient_id)
        if study is None or patient is None:
            raise HTTPException(status_code=404, detail="Study or patient for this analysis no longer exists")
        try:
            built = build_pdf(doctor, patient, study, a, {
                "original": resolve(study.image_path) if study else None,
                "gradcam": resolve(a.gradcam_overlay_path), "vit": resolve(a.vit_overlay_path)})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not generate report: {exc}") from exc
        try:
            rel = built.relative_to(STORAGE_ROOT).as_posix()
        except ValueError:
            rel = str(built)
        if rep:
            rep.pdf_path = rel
        else:
            db.add(Report(doctor_id=doctor.id, analysis_id=a.id, pdf_path=rel))
        db.commit()
        pdf = built
    if pdf is None or not pdf.exists():
        raise HTTPException(status_code=404, detail="Report file not found. Try generating the report again.")
    return FileResponse(str(pdf), media_type="application/pdf", filename=f"respira-report-{a.id[:8]}.pdf")


@router.get("/reports/{analysis_id}")
def report_json(analysis_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    a = _owned_analysis(db, doctor.id, analysis_id)
    study = db.get(XRayStudy, a.study_id)
    patient = db.get(Patient, a.patient_id)
    return {
        "analysis_id": a.id,
        "doctor": {"name": doctor.full_name, "hospital": doctor.hospital, "license": doctor.license_no},
        "patient": {"id": patient.id, "name": patient.full_name} if patient else {},
        "study": {"id": study.id} if study else {},
        "prediction": {"primary_class": a.primary_class, "confidence": a.confidence,
                       "uncertainty": a.sample_uncertainty, "uncertainty_level": a.uncertainty_level,
                       "probabilities": json.loads(a.probabilities_json or "{}")},
        "timing": {"inference_ms": a.inference_ms, "total_ms": a.total_ms},
        "model": {"version": a.model_version, "device": a.device},
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


@router.get("/reports")
def list_reports(db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    reps = db.query(Report).filter(Report.doctor_id == doctor.id).order_by(Report.created_at.desc()).limit(200).all()
    return [{"report_id": r.id, "analysis_id": r.analysis_id,
             "created_at": r.created_at.isoformat() if r.created_at else None} for r in reps]
