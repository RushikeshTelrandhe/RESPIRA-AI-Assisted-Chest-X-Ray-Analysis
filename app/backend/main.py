"""
RESPIRA - FastAPI Backend
"""

import io
import base64

from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.backend import config
from app.backend.models import (
    AttentionMap,
    ClassInfo,
    ClassDetail,
    ExplanationImage,
    GradCAMRequest,
    HealthResponse,
    PredictionResponse,
)
from app.backend.pipeline import RespiraPipeline
from app.backend.gradcam import (
    GradCAMGenerator,
    encode_png,
)
from app.backend.vit_attention import (
    compute_vit_attention,
    create_vit_attention_map,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_TAGLINE,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBALS (populated at startup)
# ============================================================

pipeline: Optional[RespiraPipeline] = None
gradcam: Optional[GradCAMGenerator] = None


@app.on_event("startup")
def startup():
    global pipeline, gradcam

    import torch

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pipeline = RespiraPipeline(device=device)
    pipeline.load()

    gradcam = GradCAMGenerator(device=device)
    gradcam.load()


# ============================================================
# HELPERS
# ============================================================


def _decode_image(
    file_bytes: bytes = None,
    base64_data: str = None,
) -> Image.Image:
    """Decode from either file upload or base64 string."""
    if file_bytes:
        return Image.open(io.BytesIO(file_bytes))
    if base64_data:
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        decoded = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(decoded))
    raise ValueError(
        "No image provided. "
        "Use 'file' (multipart) or 'image' (base64 JSON)."
    )


def _image_to_data_url(
    image: Image.Image,
) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ============================================================
# ROUTES
# ============================================================


@app.get(
    config.API_PREFIX + "/health",
    response_model=HealthResponse,
)
def health():
    return HealthResponse(
        status="ok",
        device=str(pipeline.device),
        ready=pipeline.ready,
        architecture=pipeline.model_info()["architecture"],
    )


@app.get(config.API_PREFIX + "/classes")
def classes():
    return [
        ClassInfo(
            index=i,
            name=name,
            description=config.CLASS_DESCRIPTIONS[name],
        )
        for i, name in enumerate(config.CLASS_NAMES)
    ]


@app.post(
    config.API_PREFIX + "/predict",
    response_model=PredictionResponse,
)
def predict(
    file: Optional[UploadFile] = File(None),
    image: Optional[str] = Form(None),
):
    img = _decode_image(
        file_bytes=file.file.read() if file else None,
        base64_data=image,
    )

    result = pipeline.predict_image(img)

    return PredictionResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        sample_uncertainty=result.sample_uncertainty,
        uncertainty_level=result.uncertainty_level,
        margin_uncertainty=result.margin_uncertainty,
        probabilities=result.probabilities,
        uncertainties=result.uncertainties,
        disease_weights=result.disease_weights,
        per_class=[
            ClassDetail(
                name=c.name,
                probability=c.probability,
                uncertainty=c.uncertainty,
                risk_level=c.risk_level,
                description=c.description,
            )
            for c in result.per_class
        ],
    )


@app.post(
    config.API_PREFIX + "/gradcam",
    response_model=ExplanationImage,
)
def gradcam_endpoint(
    target_class: int = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[str] = Form(None),
):
    img = _decode_image(
        file_bytes=file.file.read() if file else None,
        base64_data=image,
    )

    original, heatmap, overlay = gradcam.generate(
        img,
        target_class,
    )

    orig_pil = Image.fromarray(original)
    heat_pil = Image.fromarray(heatmap)
    over_pil = Image.fromarray(overlay)

    return ExplanationImage(
        image_url=_image_to_data_url(orig_pil),
        heatmap_url=_image_to_data_url(heat_pil),
        overlay_url=_image_to_data_url(over_pil),
    )


@app.post(
    config.API_PREFIX + "/vit-attention",
    response_model=AttentionMap,
)
def vit_attention_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[str] = Form(None),
):
    img = _decode_image(
        file_bytes=file.file.read() if file else None,
        base64_data=image,
    )

    # Need to re-run to capture vit_to_cnn
    from torchvision import transforms as T

    from app.backend.preprocessing import preprocess_pil

    final, _ = preprocess_pil(img)
    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])(Image.fromarray(final)).unsqueeze(0).to(
        pipeline.device
    )

    _, vit_to_cnn = pipeline.capture_cross_attention(tensor)

    attention_map = create_vit_attention_map(
        vit_to_cnn
    )

    # Upscale for visual
    heatmap_pil = Image.fromarray(
        (attention_map * 255).astype("uint8"),
        mode="L",
    ).resize((224, 224), resample=Image.BILINEAR)

    heatmap_rgb = heatmap_pil.convert("RGB")

    orig_pil = Image.fromarray(final).convert("RGB")

    # Overlay
    import numpy as np
    orig_np = np.array(orig_pil).astype(np.float32)
    heat_np = np.array(heatmap_rgb).astype(np.float32)
    overlay_np = (
        0.5 * orig_np + 0.5 * heat_np
    ).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)

    return AttentionMap(
        attention_map=attention_map.tolist(),
        heatmap_url=_image_to_data_url(heatmap_rgb),
        overlay_url=_image_to_data_url(overlay_pil),
    )