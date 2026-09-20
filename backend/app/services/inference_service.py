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
from app.backend.config import CLASS_DESCRIPTIONS
from app.backend.preprocessing import preprocess_pil
from backend.app.services.model_manager import MODEL_KEYS, MODEL_REGISTRY, get_manager

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


def _standalone_logits(model, tensor: torch.Tensor) -> torch.Tensor:
    """Run a standalone classifier; always returns [B,6] logits."""
    out = model(tensor)
    if isinstance(out, dict):
        return out["logits"]
    return out


def run_standalone(image: Image.Image, model_key: str) -> dict:
    """Run a single backbone model (softmax, as trained with CrossEntropy)."""
    import numpy as np

    mgr = get_manager()
    if not mgr.is_model_ready(model_key):
        err = mgr.model_errors.get(model_key) or "Respira AI model is not loaded. Check /api/v1/models/status."
        raise RuntimeError(err)
    assert mgr.pipeline is not None
    if model_key == "efficientnet":
        model = mgr.pipeline.efficientnet
    elif model_key == "vit":
        model = mgr.pipeline.vit
    elif model_key == "densenet":
        model = mgr.densenet
    else:  # guarded by caller; defense in depth
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {', '.join(MODEL_KEYS)}.")

    t0 = time.perf_counter()
    tensor = mgr.pipeline._preprocess(image).to(mgr.device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        logits = _standalone_logits(model, tensor)
    t2 = time.perf_counter()
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

    from app.backend.pipeline import RespiraPipeline

    entropy = RespiraPipeline._binary_entropy(np.asarray(probs))
    top_idx = int(np.argmax(probs))
    ordered = np.sort(np.asarray(probs))
    sample_uncertainty = float(np.mean(entropy))
    version = next(m["version"] for m in MODEL_REGISTRY if m["key"] == model_key)
    return {
        "predicted_class": ml_config.CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "sample_uncertainty": sample_uncertainty,
        "uncertainty_level": ("Low" if sample_uncertainty < ml_config.LOW_UNCERTAINTY
                              else "Moderate" if sample_uncertainty < ml_config.MODERATE_UNCERTAINTY else "High"),
        "margin_uncertainty": float(1.0 - (ordered[-1] - ordered[-2])),
        "probabilities": {k: float(v) for k, v in zip(ml_config.CLASS_NAMES, probs)},
        "uncertainties": {k: float(v) for k, v in zip(ml_config.CLASS_NAMES, entropy)},
        "disease_weights": {},
        "per_class": [
            {"name": n, "probability": float(probs[i]),
             "uncertainty": float(entropy[i]),
             "risk_level": ("low" if entropy[i] < ml_config.LOW_UNCERTAINTY
                            else "moderate" if entropy[i] < ml_config.MODERATE_UNCERTAINTY else "high"),
             "description": CLASS_DESCRIPTIONS.get(n, "")}
            for i, n in enumerate(ml_config.CLASS_NAMES)
        ],
        "timing": {
            "preprocessing_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        },
        "device": str(mgr.device),
        "class_order": list(ml_config.CLASS_NAMES),
        "model_key": model_key,
        "model_version": version,
    }


def _preprocess_tensor(image: Image.Image, image_size: int, device) -> tuple["torch.Tensor", float, float]:
    """Preprocess PIL image to a normalized tensor of the requested size.

    Returns (tensor_on_device, preprocess_ms_start, ...). Timing is handled
    by the caller; this helper only builds the tensor.
    """
    from torchvision import transforms

    from app.backend.preprocessing import preprocess_pil as preprocess_pil_size

    final_arr, _mask = preprocess_pil_size(image, image_size)
    pil_final = Image.fromarray(final_arr)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = transform(pil_final).unsqueeze(0).to(device)
    return tensor


def _build_standalone_result(probs, model_key: str, version: str, device: str, timing: dict) -> dict:
    """Build the standard standalone response dict from softmax probs."""
    import numpy as np

    from app.backend.pipeline import RespiraPipeline

    entropy = RespiraPipeline._binary_entropy(np.asarray(probs))
    top_idx = int(np.argmax(probs))
    ordered = np.sort(np.asarray(probs))
    sample_uncertainty = float(np.mean(entropy))
    return {
        "predicted_class": ml_config.CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "sample_uncertainty": sample_uncertainty,
        "uncertainty_level": ("Low" if sample_uncertainty < ml_config.LOW_UNCERTAINTY
                              else "Moderate" if sample_uncertainty < ml_config.MODERATE_UNCERTAINTY else "High"),
        "margin_uncertainty": float(1.0 - (ordered[-1] - ordered[-2])),
        "probabilities": {k: float(v) for k, v in zip(ml_config.CLASS_NAMES, probs)},
        "uncertainties": {k: float(v) for k, v in zip(ml_config.CLASS_NAMES, entropy)},
        "disease_weights": {},
        "per_class": [
            {"name": n, "probability": float(probs[i]),
             "uncertainty": float(entropy[i]),
             "risk_level": ("low" if entropy[i] < ml_config.LOW_UNCERTAINTY
                            else "moderate" if entropy[i] < ml_config.MODERATE_UNCERTAINTY else "high"),
             "description": CLASS_DESCRIPTIONS.get(n, "")}
            for i, n in enumerate(ml_config.CLASS_NAMES)
        ],
        "timing": timing,
        "device": str(device),
        "class_order": list(ml_config.CLASS_NAMES),
        "model_key": model_key,
        "model_version": version,
    }


def run_standalone_512(image: Image.Image, model_key: str) -> dict:
    """Run a high-resolution 512x512 single backbone (softmax, CrossEntropy)."""
    mgr = get_manager()
    if not mgr.is_model_ready(model_key):
        err = mgr.model_errors.get(model_key) or "Respira 512 model is not loaded. Check /api/v1/models/status."
        raise RuntimeError(err)
    if model_key == "efficientnet-512":
        model = mgr.efficientnet_512
    elif model_key == "vit-512":
        model = mgr.vit_512
    else:
        raise ValueError(f"Unknown 512 model '{model_key}'.")

    t0 = time.perf_counter()
    tensor = _preprocess_tensor(image, 512, mgr.device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        logits = _standalone_logits(model, tensor)
    t2 = time.perf_counter()
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

    version = next(m["version"] for m in MODEL_REGISTRY if m["key"] == model_key)
    return _build_standalone_result(
        probs, model_key, version, mgr.device,
        {
            "preprocessing_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        },
    )


def run_fusion_512(image: Image.Image) -> dict:
    """Weighted probability fusion of the two 512x512 backbones."""
    import numpy as np

    mgr = get_manager()
    if not mgr.is_model_ready("fusion-512"):
        err = mgr.model_errors.get("fusion-512") or "Respira Fusion-512 is not loaded. Check /api/v1/models/status."
        raise RuntimeError(err)
    assert mgr.efficientnet_512 is not None and mgr.vit_512 is not None

    weights = mgr.fusion_512_weights or {"efficientnet_weight": 0.35, "vit_weight": 0.65}
    eff_w = float(weights.get("efficientnet_weight", 0.35))
    vit_w = float(weights.get("vit_weight", 0.65))

    t0 = time.perf_counter()
    tensor = _preprocess_tensor(image, 512, mgr.device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        eff_logits = _standalone_logits(mgr.efficientnet_512, tensor)
        vit_logits = _standalone_logits(mgr.vit_512, tensor)
    t2 = time.perf_counter()
    eff_probs = torch.softmax(eff_logits, dim=1)[0].detach().cpu().numpy().astype(float)
    vit_probs = torch.softmax(vit_logits, dim=1)[0].detach().cpu().numpy().astype(float)
    fused = eff_w * np.asarray(eff_probs) + vit_w * np.asarray(vit_probs)

    version = next(m["version"] for m in MODEL_REGISTRY if m["key"] == "fusion-512")
    result = _build_standalone_result(
        fused, "fusion-512", version, mgr.device,
        {
            "preprocessing_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        },
    )
    result["disease_weights"] = {"efficientnet_512": eff_w, "vit_512": vit_w}
    return result


def run_inference(image: Image.Image, model_key: str = "fusion") -> dict:
    """Run the selected model. Returns real result + real timings (ms)."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {', '.join(MODEL_KEYS)}.")
    if model_key in ("efficientnet-512", "vit-512"):
        return run_standalone_512(image, model_key)
    if model_key == "fusion-512":
        return run_fusion_512(image)
    if model_key != "fusion":
        return run_standalone(image, model_key)
    return run_fusion(image)


def run_fusion(image: Image.Image) -> dict:
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
        "model_key": "fusion",
        "model_version": "respira-fusion-1.0",
    }
