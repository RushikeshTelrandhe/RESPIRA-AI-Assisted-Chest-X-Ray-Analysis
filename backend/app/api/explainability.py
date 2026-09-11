"""Explainability endpoints: per-disease Grad-CAM + ViT attention.

Accepts either a persisted analysis_id (ownership checked, image reloaded
from storage) or a fresh multipart upload. All heatmaps computed live by
the real backend pipeline.
"""

from __future__ import annotations

import asyncio
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from app.backend import config as ml_config
from backend.app.database import get_db
from backend.app.models_db import Analysis, Doctor
from backend.app.security import get_current_doctor
from backend.app.services import inference_service
from backend.app.services.explainability_service import gradcam_for, vit_attention_for
from backend.app.utils.storage import resolve

router = APIRouter(prefix="/explainability", tags=["explainability"])

EXPLAIN_TEXT = (
    "Grad-CAM highlights spatial regions contributing to the CNN prediction, "
    "while ViT attention visualizes attention within the transformer representation."
)


async def _load_image(analysis_id: str | None, file: UploadFile | None,
                      db: Session, doctor: Doctor) -> Image.Image:
    if analysis_id:
        a = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.doctor_id == doctor.id).first()
        if a is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        from backend.app.models_db import XRayStudy

        study = db.get(XRayStudy, a.study_id)
        p = resolve(study.image_path) if study else None
        if p is None:
            raise HTTPException(status_code=404, detail="Source image no longer retained for this analysis")
        return Image.open(p).convert("RGB")
    if file is None:
        raise HTTPException(status_code=400, detail="Provide analysis_id or an image file")
    raw = await file.read()
    try:
        img, _ = inference_service.validate_image(file.filename or "xray.png", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return img


@router.post("/gradcam")
async def gradcam(
    target_class: int = Form(...),
    analysis_id: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    doctor: Doctor = Depends(get_current_doctor),
):
    if not 0 <= target_class <= 5:
        raise HTTPException(status_code=400, detail="target_class must be 0-5")
    img = await _load_image(analysis_id, file, db, doctor)
    try:
        out = await asyncio.to_thread(gradcam_for, img, target_class)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Respira could not generate the explanation. Please try again.")
    out["explanation"] = EXPLAIN_TEXT
    out["classes"] = list(ml_config.CLASS_NAMES)
    return out


@router.post("/vit-attention")
async def vit_attention(
    analysis_id: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    doctor: Doctor = Depends(get_current_doctor),
):
    img = await _load_image(analysis_id, file, db, doctor)
    try:
        out = await asyncio.to_thread(vit_attention_for, img)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Respira could not generate the explanation. Please try again.")
    out["explanation"] = EXPLAIN_TEXT
    return out
