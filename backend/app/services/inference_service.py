"""Inference service: REAL pipeline invocation with real timing.

Uses torch.inference_mode(), model.eval() (set at load), CUDA when available.
Never fabricates probabilities / confidence / uncertainty.
"""

from __future__ import annotations

import time
from io import BytesIO

import torch
from PIL import Image

from app.backend import config as ml_config
from app.backend.preprocessing import preprocess_pil
from backend.app.services.model_manager import get_manager

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg"}


def validate_image(filename: str, raw: bytes) -> tuple[Image.Image, dict]:
    if not raw:
        raise ValueError("Empty file.")
    if len(raw) > MAX_BYTES:
        raise ValueError("File exceeds 15 MB limit.")
    ext = "." + (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type. Upload PNG or JPG.")
    sig = raw[:4]
    is_png = sig.startswith(b"\x89PNG")
    is_jpg = raw[:3] == b"\xff\xd8\xff"
    if not (is_png or is_jpg):
        raise ValueError("File content is not a valid PNG/JPG image.")
    try:
        img = Image.open(BytesIO(raw))
        img.load()
        img.convert("RGB")
    except Exception as exc:
        raise ValueError("Corrupted or unreadable image file.") from exc
    w, h = img.size
    return img.convert("RGB"), {"width": w, "height": h, "size": len(raw), "filename": filename}


def run_inference(image: Image.Image) -> dict:
    """Run the full fusion pipeline. Returns real result + real timings (ms)."""
    mgr = get_manager()
    if not mgr.ready or mgr.pipeline is None:
        raise RuntimeError("Respira AI model is not loaded. Check /api/v1/models/status.")

    t0 = time.perf_counter()
    with torch.inference_mode():
        final_arr, _mask = preprocess_pil(image)
    t1 = time.perf_counter()
    with torch.inference_mode():
        result = mgr.pipeline.predict_image(image)
    t2 = time.perf_counter()

    pil_final = Image.fromarray(final_arr)
    return {
        "predicted_class": result.predicted_class,
        "confidence": float(result.confidence),
        "sample_uncertainty": float(result.sample_uncertainty),
        "uncertainty_level": str(result.uncertainty_level).capitalize(),
        "margin_uncertainty": float(result.margin_uncertainty),
        "probabilities": {k: float(v) for k, v in result.probabilities.items()},
        "uncertainties": {k: float(v) for k, v in result.uncertainties.items()},
        "disease_weights": {k: float(v) for k, v in (result.disease_weights or {}).items()},
        "per_class": [
            {"name": c.name, "probability": float(c.probability),
             "uncertainty": float(c.uncertainty), "risk_level": c.risk_level,
             "description": c.description}
            for c in result.per_class
        ],
        "timing": {
            "preprocessing_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        },
        "device": str(mgr.device),
        "class_order": list(ml_config.CLASS_NAMES),
    }
