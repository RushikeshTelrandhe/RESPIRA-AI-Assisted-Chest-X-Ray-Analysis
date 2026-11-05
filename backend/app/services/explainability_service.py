"""Explainability service: REAL Grad-CAM + REAL ViT cross-attention across all models.

Supports:
  1. Finetuned Respira Fusion (512x512): Hooks cnn_backbone[8][0] and extracts
     disease-conditioned ViT attention (32x32) at full unmasked 512x512 resolution.
  2. High-resolution 512 backbones (EfficientNet-512, Fusion-512).
  3. Legacy 224 models (Respira Fusion 224, DenseNet-121).
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from app.backend import config as ml_config
from app.backend.gradcam import self_and_jet
from app.backend.preprocessing import preprocess_pil
from app.backend.vit_attention import create_vit_attention_map
from backend.app.services.model_manager import get_manager


def _to_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_cnn_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image: Image.Image,
    target_class: int,
    image_size: int = 512,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device

    final_arr, _ = preprocess_pil(image, image_size=image_size)
    pil_img = Image.fromarray(final_arr)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_m, _i, out):
        activations.append(out)

    def bwd_hook(_m, _gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.enable_grad():
            out = model(tensor)
            logits = out["logits"] if isinstance(out, dict) else out
            score = logits[0, target_class]
            score.backward()
    finally:
        h1.remove()
        h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM activations/gradients could not be captured.")

    act = activations[0]
    grad = gradients[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    # Boundary taper: attenuate extreme outer perimeter (outer ~4%)
    # Prevents burned-in viewer annotations (digital scale rulers, orientation labels,
    # collimation borders) outside the patient's thorax from dominating the diagnostic heatmap.
    border_px = int(image_size * 0.04)
    margin_mask = np.ones_like(cam, dtype=np.float32)
    for b in range(border_px):
        factor = float((b / border_px) ** 2)
        margin_mask[b, :] = np.minimum(margin_mask[b, :], factor)
        margin_mask[image_size - 1 - b, :] = np.minimum(margin_mask[image_size - 1 - b, :], factor)
        margin_mask[:, b] = np.minimum(margin_mask[:, b], factor)
        margin_mask[:, image_size - 1 - b] = np.minimum(margin_mask[:, image_size - 1 - b], factor)

    cam = cam * margin_mask
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    original_rgb = np.array(pil_img.convert("RGB"))
    heatmap_np = np.uint8(255 * cam)
    colored = self_and_jet(np.stack([cam, cam, cam], axis=-1))
    overlay_np = np.uint8(0.5 * colored + 0.5 * original_rgb)

    return original_rgb, heatmap_np, overlay_np


def _generate_finetuned_vit_attention(
    image: Image.Image,
    target_class: int = 0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, Image.Image, Image.Image, list]:
    mgr = get_manager()
    model = mgr.finetuned_fusion
    if model is None:
        raise RuntimeError("Finetuned Respira Fusion model is not loaded.")
    if device is None:
        device = mgr.device

    final_arr, _ = preprocess_pil(image, image_size=512)
    pil_img = Image.fromarray(final_arr)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(tensor, return_intermediates=True)

    disease_att = out.get("disease_att")
    if disease_att is None:
        raise RuntimeError("Disease-conditioned attention was not returned.")

    target_idx = max(0, min(5, target_class))
    # Tokens 256 to 1280 (1024 tokens) represent the 32x32 ViT spatial patches
    vit_att = disease_att[0, target_idx, 256:].reshape(32, 32).detach().cpu().numpy()
    # Soft boundary taper for outer ViT patch perimeter
    margin_vit = np.ones_like(vit_att, dtype=np.float32)
    margin_vit[0, :] *= 0.25
    margin_vit[-1, :] *= 0.25
    margin_vit[:, 0] *= 0.25
    margin_vit[:, -1] *= 0.25
    vit_att = vit_att * margin_vit
    vit_att = vit_att - vit_att.min()
    if vit_att.max() > 0:
        vit_att = vit_att / vit_att.max()

    vit_pil = Image.fromarray((vit_att * 255).astype("uint8"), mode="L").resize((512, 512), Image.BILINEAR)
    vit_arr = np.array(vit_pil) / 255.0

    original_rgb = np.array(pil_img.convert("RGB"))
    colored = self_and_jet(np.stack([vit_arr, vit_arr, vit_arr], axis=-1))
    overlay_np = np.uint8(0.5 * colored + 0.5 * original_rgb)

    return original_rgb, Image.fromarray(np.uint8(colored)), Image.fromarray(overlay_np), vit_att.tolist()


def gradcam_for(image: Image.Image, target_class: int, model_key: str = "fusion") -> dict:
    if not 0 <= target_class <= 5:
        raise ValueError("target_class must be 0-5")
    mgr = get_manager()
    t0 = time.perf_counter()

    key = (model_key or "fusion").strip().lower().replace("_", "-")

    if key in ("finetuned-fusion", "respira-fusion-finetuned") and mgr.finetuned_fusion is not None:
        target_layer = mgr.finetuned_fusion.cnn_backbone[8][0]
        original, heatmap, overlay = _generate_cnn_gradcam(
            model=mgr.finetuned_fusion,
            target_layer=target_layer,
            image=image,
            target_class=target_class,
            image_size=512,
            device=mgr.device,
        )
    elif key in ("efficientnet-512", "fusion-512") and mgr.efficientnet_512 is not None:
        target_layer = mgr.efficientnet_512.backbone.features[8][0]
        original, heatmap, overlay = _generate_cnn_gradcam(
            model=mgr.efficientnet_512,
            target_layer=target_layer,
            image=image,
            target_class=target_class,
            image_size=512,
            device=mgr.device,
        )
    else:
        if not mgr.ready or mgr.gradcam is None:
            raise RuntimeError("Explainability model is not loaded.")
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


def vit_attention_for(image: Image.Image, target_class: int = 0, model_key: str = "fusion") -> dict:
    mgr = get_manager()
    t0 = time.perf_counter()

    key = (model_key or "fusion").strip().lower().replace("_", "-")

    if key in ("finetuned-fusion", "respira-fusion-finetuned") and mgr.finetuned_fusion is not None:
        original, heat, overlay, attn_map = _generate_finetuned_vit_attention(
            image=image,
            target_class=target_class,
            device=mgr.device,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "image": _to_data_url(Image.fromarray(original)),
            "heatmap": _to_data_url(heat),
            "overlay": _to_data_url(overlay),
            "attention_map": attn_map,
            "explainability_ms": ms,
        }

    # Fallback to legacy 224 pipeline
    if not mgr.ready or mgr.pipeline is None:
        raise RuntimeError("Explainability model is not loaded.")
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
