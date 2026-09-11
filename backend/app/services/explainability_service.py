"""Explainability service: REAL Grad-CAM + REAL ViT cross-attention.

Grad-CAM targets the EfficientNet-B0 branch (last Conv2d), per-disease via
target_class index. ViT attention uses the live vit_to_cnn cross-attention
mean over CNN tokens -> 14x14 map (same method as STEP 8.8 analysis).
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
from PIL import Image
from torchvision import transforms as T

from app.backend import config as ml_config
from app.backend.preprocessing import preprocess_pil
from app.backend.vit_attention import create_vit_attention_map
from backend.app.services.model_manager import get_manager


def _to_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def gradcam_for(image: Image.Image, target_class: int) -> dict:
    if not 0 <= target_class <= 5:
        raise ValueError("target_class must be 0-5")
    mgr = get_manager()
    if not mgr.ready or mgr.gradcam is None:
        raise RuntimeError("Explainability model is not loaded.")
    t0 = time.perf_counter()
    original, heatmap, overlay = mgr.gradcam.generate(image, target_class)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "image": _to_data_url(Image.fromarray(original)),
        "heatmap": _to_data_url(Image.fromarray(heatmap).convert("RGB")),
        "overlay": _to_data_url(Image.fromarray(overlay)),
        "target_class": ml_config.CLASS_NAMES[target_class],
        "target_index": target_class,
        "explainability_ms": ms,
    }


def vit_attention_for(image: Image.Image) -> dict:
    import torch

    mgr = get_manager()
    if not mgr.ready or mgr.pipeline is None:
        raise RuntimeError("Explainability model is not loaded.")
    t0 = time.perf_counter()
    final, _ = preprocess_pil(image)
    tensor = (
        T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(Image.fromarray(final))
        .unsqueeze(0)
        .to(mgr.pipeline.device)
    )
    with torch.inference_mode():
        _c2v, vit_to_cnn = mgr.pipeline.capture_cross_attention(tensor)
    attn14 = create_vit_attention_map(vit_to_cnn[0] if vit_to_cnn.ndim == 3 else vit_to_cnn)
    heat = Image.fromarray((attn14 * 255).astype("uint8"), mode="L").resize((224, 224), Image.BILINEAR).convert("RGB")
    orig = Image.fromarray(final).convert("RGB")
    overlay = Image.fromarray(((np.array(orig).astype(np.float32) * 0.5 + np.array(heat).astype(np.float32) * 0.5)).astype("uint8"))
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "image": _to_data_url(orig),
        "heatmap": _to_data_url(heat),
        "overlay": _to_data_url(overlay),
        "attention_map": attn14.tolist(),
        "explainability_ms": ms,
    }
