"""
RESPIRA - Pydantic response/request schemas
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class ClassDetail(BaseModel):
    name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    description: str = ""


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    sample_uncertainty: float
    uncertainty_level: str
    margin_uncertainty: float
    probabilities: Dict[str, float]
    uncertainties: Dict[str, float]
    disease_weights: Dict[str, float] = {}
    per_class: List[ClassDetail]


class GradCAMRequest(BaseModel):
    target_class: int = Field(
        ...,
        ge=0,
        le=5,
        description="Disease class index (0-5)",
    )


class ExplanationImage(BaseModel):
    image_url: str = Field(
        ...,
        description="Base64 data URL of the image",
    )
    heatmap_url: str = Field(
        ...,
        description="Base64 data URL of the heatmap",
    )
    overlay_url: str = Field(
        ...,
        description="Base64 data URL of the overlay",
    )


class AttentionMap(BaseModel):
    attention_map: List[List[float]] = Field(
        ...,
        description="14x14 attention grid",
    )
    heatmap_url: str
    overlay_url: str


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    ready: bool
    architecture: dict


class ClassInfo(BaseModel):
    index: int
    name: str
    description: str