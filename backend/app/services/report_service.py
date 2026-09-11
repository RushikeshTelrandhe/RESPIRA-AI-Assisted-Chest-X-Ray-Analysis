"""PDF report generation (reportlab). Real patient/study/AI data only."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from backend.app.config import STORAGE_ROOT
from backend.app.schemas import DISCLAIMER


def _draw_bar(c: canvas.Canvas, x: float, y: float, w: float, h: float, frac: float) -> None:
    c.setFillColorRGB(0.90, 0.93, 0.95)
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setFillColorRGB(0.10, 0.45, 0.65)
    c.rect(x, y, w * max(0.0, min(1.0, frac)), h, stroke=0, fill=1)


def build_pdf(doctor, patient, study, analysis, images: dict[str, Path | None]) -> Path:
    if patient is None or study is None:
        raise ValueError("Cannot build report: patient or study is missing")
    probs = json.loads(analysis.probabilities_json or "{}")
    out_dir = STORAGE_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"respira-report-{analysis.id}.pdf"

    def _num(v, default: float = 0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    doctor_name = getattr(doctor, "full_name", "") or ""
    doctor_hosp = getattr(doctor, "hospital", "") or ""
    doctor_lic = getattr(doctor, "license_no", "") or ""
    patient_name = getattr(patient, "full_name", "") or ""
    patient_code = getattr(patient, "patient_code", "") or ""
    patient_age = getattr(patient, "age", "") or ""
    patient_gender = getattr(patient, "gender", "") or ""
    aid = str(getattr(analysis, "id", "") or "")
    sid = str(getattr(study, "id", "") or "")
    created = getattr(analysis, "created_at", "") or ""
    device = getattr(analysis, "device", "") or ""
    model_version = getattr(analysis, "model_version", "") or ""
    primary = getattr(analysis, "primary_class", "") or ""
    conf = _num(getattr(analysis, "confidence", 0.0))
    samp_unc = _num(getattr(analysis, "sample_uncertainty", 0.0))
    unc_level = getattr(analysis, "uncertainty_level", "") or ""
    margin = _num(getattr(analysis, "margin_uncertainty", 0.0))
    infer_ms = _num(getattr(analysis, "inference_ms", 0.0))

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    y = H - 18 * mm
    c.setFont("Helvetica-Bold", 20)
    c.drawString(18 * mm, y, "RESPIRA")
    c.setFont("Helvetica", 10)
    c.drawString(18 * mm, y - 6 * mm, "AI-Assisted Chest X-Ray Analysis")
    y -= 16 * mm
    c.setFont("Helvetica", 9)
    c.drawString(18 * mm, y, f"Doctor: {doctor_name}  |  {doctor_hosp}  |  Reg: {doctor_lic}")
    y -= 5 * mm
    c.drawString(18 * mm, y, f"Patient: {patient_name} (ID: {patient_code or aid[:8]})  |  Age: {patient_age}  |  Gender: {patient_gender}")
    y -= 5 * mm
    c.drawString(18 * mm, y, f"Study: {sid[:8]}  |  Analysis: {aid[:8]}  |  Date: {created}  |  Device: {device}")
    y -= 5 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(18 * mm, y, f"AI Prediction: {primary}")
    c.setFont("Helvetica", 9)
    y -= 6 * mm
    c.drawString(18 * mm, y, f"Confidence: {conf:.2%}   Uncertainty: {samp_unc:.3f} ({unc_level})   Margin: {margin:.3f}")
    y -= 8 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(18 * mm, y, "Disease probabilities")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    for name, p in probs.items():
        if y < 60 * mm:
            c.showPage()
            y = H - 18 * mm
        c.drawString(18 * mm, y, f"{name}")
        _draw_bar(c, 70 * mm, y - 1 * mm, 70 * mm, 4 * mm, float(p))
        c.drawString(145 * mm, y, f"{float(p):.2%}")
        y -= 7 * mm

    # Visualizations (original + overlays) if stored
    for label, key in [("Original X-ray", "original"), ("Grad-CAM overlay", "gradcam"), ("ViT attention overlay", "vit")]:
        p = images.get(key)
        if p and Path(p).exists():
            if y < 80 * mm:
                c.showPage()
                y = H - 18 * mm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(18 * mm, y, label)
            y -= 4 * mm
            try:
                img = ImageReader(str(p))
                iw, ih = img.getSize()
                scale = min(90 * mm / iw, 70 * mm / ih)
                c.drawImage(img, 18 * mm, y - ih * scale, width=iw * scale, height=ih * scale)
                y -= ih * scale + 6 * mm
            except Exception:
                y -= 4 * mm

    c.setFont("Helvetica", 7.5)
    for line in [DISCLAIMER, f"Generated {datetime.utcnow().isoformat()}Z  |  Model: {model_version}  |  Inference: {infer_ms:.0f} ms"]:
        for chunk in [line[i:i + 130] for i in range(0, len(line), 130)]:
            if y < 15 * mm:
                c.showPage()
                y = H - 18 * mm
            c.drawString(18 * mm, y, chunk)
            y -= 4 * mm
    c.showPage()
    c.save()
    buf_check = io.BytesIO()  # keep import used
    del buf_check
    return pdf_path
