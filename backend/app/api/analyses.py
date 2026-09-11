"""X-ray upload + REAL analysis + analysis detail.

POST /xray/upload   - validate file, return preview metadata (no ML call).
POST /analyze       - validate, run REAL Respira pipeline, persist Study+Analysis.
GET  /analysis/{id} - full persisted result (ownership checked).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models_db import Analysis, AuditLog, Doctor, Patient, XRayStudy
from backend.app.schemas import AnalyzeResponse, AnalysisDetail, DiseaseScore
from backend.app.security import get_current_doctor
from backend.app.services import inference_service
from backend.app.services.explainability_service import gradcam_for, vit_attention_for
from backend.app.utils.storage import save_analysis_images, save_upload_temp

router = APIRouter(tags=["xray"])


def _thumb_b64(img: Image.Image) -> str:
    t = img.copy()
    t.thumbnail((512, 512))
    buf = io.BytesIO()
    t.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/xray/upload")
async def upload_xray(
    file: UploadFile = File(...),
    doctor: Doctor = Depends(get_current_doctor),
):
    raw = await file.read()
    try:
        img, meta = inference_service.validate_image(file.filename or "upload.png", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_upload_temp(doctor.id, file.filename or "upload.png", raw)
    return {
        "filename": meta["filename"], "width": meta["width"], "height": meta["height"],
        "file_size": meta["size"], "preview": _thumb_b64(img),
    }


def _persist_analysis(db: Session, doctor: Doctor, patient: Patient, filename: str,
                      img: Image.Image, file_size: int, result: dict,
                      grad: dict | None, vit: dict | None) -> tuple[XRayStudy, Analysis]:
    rel_paths = save_analysis_images(analysis_id="pending", image=img, grad=grad, vit=vit) if settings.RETAIN_IMAGES else {}
    study = XRayStudy(
        doctor_id=doctor.id, patient_id=patient.id, filename=filename,
        image_width=img.size[0], image_height=img.size[1], file_size=file_size,
        image_path=rel_paths.get("original", ""), status="completed",
    )
    db.add(study)
    db.flush()
    if settings.RETAIN_IMAGES:  # rewrite under real analysis id path
        rel_paths = save_analysis_images(analysis_id=study.id, image=img, grad=grad, vit=vit)
        study.image_path = rel_paths.get("original", "")
    probs = result["probabilities"]
    analysis = Analysis(
        doctor_id=doctor.id, patient_id=patient.id, study_id=study.id,
        primary_class=result["predicted_class"], confidence=result["confidence"],
        sample_uncertainty=result["sample_uncertainty"], uncertainty_level=result["uncertainty_level"],
        margin_uncertainty=result["margin_uncertainty"],
        probabilities_json=json.dumps(probs), uncertainties_json=json.dumps(result["uncertainties"]),
        disease_weights_json=json.dumps(result.get("disease_weights", {})),
        model_version="respira-fusion-1.0", device=result.get("device", ""),
        preprocessing_ms=result["timing"]["preprocessing_ms"], inference_ms=result["timing"]["inference_ms"],
        explainability_ms=(grad or {}).get("explainability_ms", 0.0) + (vit or {}).get("explainability_ms", 0.0),
        total_ms=result["timing"]["total_ms"],
        gradcam_path=rel_paths.get("gradcam", ""), gradcam_overlay_path=rel_paths.get("gradcam_overlay", ""),
        vit_attention_path=rel_paths.get("vit", ""), vit_overlay_path=rel_paths.get("vit_overlay", ""),
    )
    db.add(analysis)
    db.flush()
    db.add(AuditLog(doctor_id=doctor.id, action="xray_analyzed", entity="analysis", entity_id=analysis.id))
    db.commit()
    db.refresh(study)
    db.refresh(analysis)
    return study, analysis


def _analyze_payload(patient: Patient, study: XRayStudy, analysis: Analysis, diseases: list[DiseaseScore]) -> AnalyzeResponse:
    return AnalyzeResponse(
        analysis_id=analysis.id,
        patient={"id": patient.id, "name": patient.full_name},
        study={"id": study.id, "created_at": study.created_at.isoformat() if study.created_at else None},
        prediction={
            "primary_class": analysis.primary_class, "confidence": analysis.confidence,
            "uncertainty": analysis.sample_uncertainty, "uncertainty_level": analysis.uncertainty_level,
            "margin_uncertainty": analysis.margin_uncertainty,
        },
        diseases=diseases,
        explainability={"gradcam": True, "vit_attention": True},
        timing={
            "preprocessing_ms": analysis.preprocessing_ms, "inference_ms": analysis.inference_ms,
            "explainability_ms": analysis.explainability_ms, "total_ms": analysis.total_ms,
        },
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    doctor: Doctor = Depends(get_current_doctor),
):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    raw = await file.read()
    try:
        img, meta = inference_service.validate_image(file.filename or "xray.png", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = await asyncio.to_thread(inference_service.run_inference, img)
        # Real explainability generated at analysis time (disease = predicted argmax)
        from app.backend import config as ml_config

        pred_idx = ml_config.CLASS_NAMES.index(result["predicted_class"])
        grad = await asyncio.to_thread(gradcam_for, img, pred_idx)
        vit = await asyncio.to_thread(vit_attention_for, img)
        result["timing"]["explainability_ms"] = grad["explainability_ms"] + vit["explainability_ms"]
        result["timing"]["total_ms"] += result["timing"]["explainability_ms"]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Respira could not analyze this image. Please verify the X-ray file and try again.")

    study, analysis = _persist_analysis(db, doctor, patient, meta["filename"], img, meta["size"], result, grad, vit)
    # store overlays as data URLs? No - files on disk; record relative paths only (never fs paths to client)
    diseases = [DiseaseScore(name=n, probability=float(p)) for n, p in result["probabilities"].items()]
    payload = _analyze_payload(patient, study, analysis, diseases)
    return payload


@router.get("/analysis/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(get_current_doctor)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.doctor_id == doctor.id).first()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    study = db.get(XRayStudy, analysis.study_id)
    patient = db.get(Patient, analysis.patient_id)
    probs = json.loads(analysis.probabilities_json or "{}")
    return AnalysisDetail(
        analysis_id=analysis.id,
        patient={"id": patient.id, "name": patient.full_name} if patient else {},
        study={"id": study.id, "created_at": study.created_at.isoformat() if study and study.created_at else None,
               "filename": study.filename if study else "", "width": study.image_width if study else 0,
               "height": study.image_height if study else 0},
        prediction={"primary_class": analysis.primary_class, "confidence": analysis.confidence,
                    "uncertainty": analysis.sample_uncertainty, "uncertainty_level": analysis.uncertainty_level,
                    "margin_uncertainty": analysis.margin_uncertainty,
                    "uncertainties": json.loads(analysis.uncertainties_json or "{}"),
                    "disease_weights": json.loads(analysis.disease_weights_json or "{}")},
        diseases=[DiseaseScore(name=n, probability=float(p)) for n, p in probs.items()],
        timing={"preprocessing_ms": analysis.preprocessing_ms, "inference_ms": analysis.inference_ms,
                "explainability_ms": analysis.explainability_ms, "total_ms": analysis.total_ms},
        model={"version": analysis.model_version, "device": analysis.device},
        explainability={"gradcam": bool(analysis.gradcam_overlay_path), "vit_attention": bool(analysis.vit_overlay_path)},
        created_at=analysis.created_at,
    )
