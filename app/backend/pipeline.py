"""
RESPIRA - End-to-End Inference Pipeline

Chains the complete Respira fusion architecture from a single
chest X-ray image to per-disease probabilities and uncertainty:

    Image
      |-- EfficientNet-B0  -> spatial features (1, 1280, 7, 7)
      |-- ViT-B/16         -> patch tokens (1, 196, 768)
      |
      +-> FeatureProjection            -> (1, 49, 512) + (1, 196, 512)
      +-> BidirectionalCrossAttention -> fused tokens (1, 245, 512)
      +-> DiseaseConditionedAttention -> disease features (1, 6, 512)
      +-> DiseaseRelationshipModel    -> relationship matrix (1, 6, 6)
      +-> MultiLabelClassificationHeads -> logits (1, 6)

    sigmoid(logits) -> probabilities
    binary entropy  -> uncertainty

The EfficientNet, ViT, and classification-head checkpoints are
trained artifacts. The intermediate fusion modules were used at
feature-extraction time with default random initialization and
their weights were not persisted; they are therefore recreated
deterministically using a fixed seed so the app is reproducible.
"""

import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.backend import config
from app.backend.preprocessing import preprocess_pil
from app.backend.networks import (
    AdaptiveDiseaseFusion,
    BidirectionalCrossAttention,
    DiseaseConditionedAttention,
    DiseaseRelationshipModel,
    FeatureProjection,
    MultiLabelClassificationHeads,
)

from src.models.efficientnet import EfficientNetB0Classifier
from src.models.vit import VisionTransformerModel


# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass
class DiseaseResult:
    name: str
    probability: float
    uncertainty: float
    risk_level: str
    description: str


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    sample_uncertainty: float
    uncertainty_level: str
    margin_uncertainty: float
    probabilities: dict
    uncertainties: dict
    disease_weights: dict = field(default_factory=dict)
    per_class: List[DiseaseResult] = field(default_factory=list)


# ============================================================
# PIPELINE
# ============================================================


class RespiraPipeline:
    """End-to-end Respira inference pipeline."""

    def __init__(
        self,
        device: torch.device = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.ready = False
        self.info = {}

        self.efficientnet = None
        self.vit = None
        self.projection = None
        self.cross_attention = None
        self.disease_attention = None
        self.adaptive_fusion = None
        self.relationship = None
        self.classification_heads = None

    # --------------------------------------------------------
    # MODEL CONSTRUCTION
    # --------------------------------------------------------

    def _set_seed(self) -> None:
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

    def _build_intermediate_modules(self) -> None:
        """
        The projection / cross-attention / disease-attention /
        relationship modules have no persisted weights, so they
        are recreated deterministically with a fixed seed.
        """
        self._set_seed()

        self.projection = FeatureProjection(
            efficientnet_dim=1280,
            vit_dim=768,
            projection_dim=config.FEATURE_DIM,
        )

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            dropout=0.1,
        )

        self.disease_attention = DiseaseConditionedAttention(
            num_diseases=config.NUM_CLASSES,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
        )

        self.relationship = DiseaseRelationshipModel(
            feature_dim=config.FEATURE_DIM,
            num_heads=config.NUM_HEADS,
            dropout=0.1,
        )

        for module in [
            self.projection,
            self.cross_attention,
            self.disease_attention,
            self.relationship,
        ]:
            module.to(self.device)
            module.eval()

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------

    def load(self) -> "RespiraPipeline":
        if not config.EFFICIENTNET_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"EfficientNet checkpoint missing:\n"
                f"{config.EFFICIENTNET_CHECKPOINT}"
            )
        if not config.VIT_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"ViT checkpoint missing:\n"
                f"{config.VIT_CHECKPOINT}"
            )
        if not config.CLASSIFICATION_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Classification checkpoint missing:\n"
                f"{config.CLASSIFICATION_CHECKPOINT}"
            )

        # EfficientNet-B0 -------------------------------------
        self.efficientnet = EfficientNetB0Classifier(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            dropout=0.3,
        )
        ckpt = torch.load(
            config.EFFICIENTNET_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )
        self.efficientnet.load_state_dict(ckpt["model_state_dict"])
        self.efficientnet.to(self.device)
        self.efficientnet.eval()

        # ViT-B/16 --------------------------------------------
        self.vit = VisionTransformerModel(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
        )
        ckpt = torch.load(
            config.VIT_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )
        self.vit.load_state_dict(ckpt["model_state_dict"])
        self.vit.to(self.device)
        self.vit.eval()

        # Intermediate fusion modules --------------------------
        self._build_intermediate_modules()

        # AdaptiveDiseaseFusion (trained) ----------------------
        if config.ADAPTIVE_FUSION_CHECKPOINT.exists():
            self.adaptive_fusion = AdaptiveDiseaseFusion(
                feature_dim=config.FEATURE_DIM,
                num_classes=config.NUM_CLASSES,
                hidden_dim=config.HIDDEN_DIM,
                dropout=0.2,
            )
            ckpt = torch.load(
                config.ADAPTIVE_FUSION_CHECKPOINT,
                map_location=self.device,
                weights_only=False,
            )
            self.adaptive_fusion.load_state_dict(
                ckpt["model_state_dict"]
            )
            self.adaptive_fusion.to(self.device)
            self.adaptive_fusion.eval()

        # Multi-Label Classification Heads (trained) -----------
        self.classification_heads = MultiLabelClassificationHeads(
            num_classes=config.NUM_CLASSES,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=0.3,
        )
        ckpt = torch.load(
            config.CLASSIFICATION_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )
        self.classification_heads.load_state_dict(
            ckpt["model_state_dict"]
        )
        self.classification_heads.to(self.device)
        self.classification_heads.eval()

        self.ready = True
        return self

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    @torch.no_grad()
    def _extract_features(
        self,
        tensor: torch.Tensor,
    ):
        """
        Run image -> EfficientNet + ViT features.

        Returns
        -------
        efficientnet_spatial:
            (B, 1280, 7, 7)
        vit_tokens:
            (B, 196, 768)
        """

        # EfficientNet spatial feature map
        efficientnet_spatial = self.efficientnet.backbone.features(
            tensor
        )

        # ViT patch tokens (B, 196, 768)
        vit_output = self.vit(tensor)
        vit_tokens = vit_output["tokens"]

        return efficientnet_spatial, vit_tokens

    @torch.no_grad()
    def _fusion_features(
        self,
        efficientnet_spatial: torch.Tensor,
        vit_tokens: torch.Tensor,
    ):
        """
        Run projection + cross-attention + disease attention
        + relationship modeling.

        Returns
        -------
        disease_features:
            (B, 6, 512)
        relationship_matrix:
            (B, 6, 6)
        """

        # EfficientNet: (B, 1280, 7, 7) -> (B, 49, 1280)
        cnn_tokens = (
            efficientnet_spatial
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                efficientnet_spatial.shape[0],
                49,
                1280,
            )
        )

        # Projection to common 512 dimension
        projected_cnn, projected_vit = self.projection(
            cnn_tokens,
            vit_tokens,
        )

        # Bidirectional cross-attention
        cnn_enhanced, vit_enhanced, _, _ = self.cross_attention(
            projected_cnn,
            projected_vit,
        )

        # Fused tokens: (B, 49+196, 512) = (B, 245, 512)
        fused_tokens = torch.cat(
            [cnn_enhanced, vit_enhanced],
            dim=1,
        )

        # Disease-conditioned attention
        disease_features, _ = self.disease_attention(
            fused_tokens
        )

        # Disease relationship modeling
        _, attention_matrices = self.relationship(
            disease_features
        )

        # Relationship matrix: average over attention heads
        # (B, heads, 6, 6) -> (B, 6, 6)
        relationship_matrix = attention_matrices.mean(
            dim=1
        )

        return disease_features, relationship_matrix

    @torch.no_grad()
    def predict_image(
        self,
        image: Image.Image,
    ):
        """
        Run the full pipeline for a single image.

        Parameters
        ----------
        image:
            PIL image (RGB or grayscale).

        Returns
        -------
        PredictionResult
        """
        if not self.ready:
            raise RuntimeError(
                "Pipeline not loaded. Call load() first."
            )

        tensor = self._preprocess(image)

        tensor = tensor.to(self.device)

        efficientnet_spatial, vit_tokens = (
            self._extract_features(tensor)
        )

        disease_features, relationship_matrix = (
            self._fusion_features(
                efficientnet_spatial,
                vit_tokens,
            )
        )

        # Optional adaptive fusion disease weights
        disease_weights = {}
        if self.adaptive_fusion is not None:
            fusion_out = self.adaptive_fusion(
                disease_features
            )
            weights = fusion_out["disease_weights"]
            disease_weights = {
                config.CLASS_NAMES[i]:
                    float(weights[0, i].item())
                for i in range(config.NUM_CLASSES)
            }

        # Final logits -> probabilities
        logits = self.classification_heads(
            disease_features,
            relationship_matrix,
        )

        probabilities = torch.sigmoid(
            logits
        )[0].cpu().numpy()

        # Uncertainty: normalized binary entropy per class
        uncertainty = self._binary_entropy(
            probabilities
        )

        return self._build_result(
            probabilities,
            uncertainty,
            disease_weights,
        )

    def _preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        from torchvision import transforms

        final, _ = preprocess_pil(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return transform(
            Image.fromarray(final)
        ).unsqueeze(0)

    # --------------------------------------------------------
    # UTILITIES
    # --------------------------------------------------------

    @staticmethod
    def _binary_entropy(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        epsilon = 1e-8
        p = np.clip(
            probabilities,
            epsilon,
            1.0 - epsilon,
        )
        entropy = -(
            p * np.log(p)
            +
            (1.0 - p) * np.log(1.0 - p)
        )
        return entropy / np.log(2.0)

    @staticmethod
    def _uncertainty_level(
        value: float,
    ) -> str:
        if value < config.LOW_UNCERTAINTY:
            return "low"
        if value < config.MODERATE_UNCERTAINTY:
            return "moderate"
        return "high"

    def _build_result(
        self,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        disease_weights: dict,
    ) -> PredictionResult:
        predicted_index = int(
            np.argmax(probabilities)
        )
        predicted_class = config.CLASS_NAMES[
            predicted_index
        ]
        confidence = float(
            probabilities[predicted_index]
        )

        sample_uncertainty = float(
            uncertainty.mean()
        )

        sorted_probs = np.sort(probabilities)
        margin_uncertainty = float(
            1.0 - (sorted_probs[-1] - sorted_probs[-2])
        )

        per_class = []
        for i, name in enumerate(config.CLASS_NAMES):
            per_class.append(
                DiseaseResult(
                    name=name,
                    probability=float(probabilities[i]),
                    uncertainty=float(uncertainty[i]),
                    risk_level=self._uncertainty_level(
                        float(uncertainty[i])
                    ),
                    description=(
                        config.CLASS_DESCRIPTIONS[name]
                    ),
                )
            )

        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            sample_uncertainty=sample_uncertainty,
            uncertainty_level=self._uncertainty_level(
                sample_uncertainty
            ),
            margin_uncertainty=margin_uncertainty,
            probabilities={
                name: float(probabilities[i])
                for i, name in enumerate(config.CLASS_NAMES)
            },
            uncertainties={
                name: float(uncertainty[i])
                for i, name in enumerate(config.CLASS_NAMES)
            },
            disease_weights=disease_weights,
            per_class=per_class,
        )

    # --------------------------------------------------------
    # ADDITIONAL INTERMEDIATE OUTPUTS (for explainability)
    # --------------------------------------------------------

    @torch.no_grad()
    def capture_cross_attention(
        self,
        tensor: torch.Tensor,
    ):
        efficientnet_spatial, vit_tokens = (
            self._extract_features(tensor)
        )

        cnn_tokens = (
            efficientnet_spatial
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                efficientnet_spatial.shape[0],
                49,
                1280,
            )
        )

        projected_cnn, projected_vit = self.projection(
            cnn_tokens,
            vit_tokens,
        )

        cnn_enhanced, vit_enhanced, cnn_to_vit, vit_to_cnn = (
            self.cross_attention(
                projected_cnn,
                projected_vit,
            )
        )

        return (
            cnn_to_vit,
            vit_to_cnn,
        )

    @torch.no_grad()
    def capture_disease_attention(
        self,
        tensor: torch.Tensor,
    ):
        efficientnet_spatial, vit_tokens = (
            self._extract_features(tensor)
        )

        cnn_tokens = (
            efficientnet_spatial
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                efficientnet_spatial.shape[0],
                49,
                1280,
            )
        )

        projected_cnn, projected_vit = self.projection(
            cnn_tokens,
            vit_tokens,
        )

        cnn_enhanced, vit_enhanced, _, _ = self.cross_attention(
            projected_cnn,
            projected_vit,
        )

        fused_tokens = torch.cat(
            [cnn_enhanced, vit_enhanced],
            dim=1,
        )

        disease_features, attention_maps = (
            self.disease_attention(fused_tokens)
        )

        return disease_features, attention_maps

    def model_info(self) -> dict:
        return {
            "device": str(self.device),
            "ready": self.ready,
            "classes": config.CLASS_NAMES,
            "architecture": {
                "cnn": "EfficientNet-B0",
                "transformer": "ViT-B/16",
                "projection_dim": config.FEATURE_DIM,
                "cross_attention_heads": config.NUM_HEADS,
                "disease_attention_heads": config.NUM_HEADS,
                "disease_count": config.NUM_CLASSES,
                "relationship_heads": config.NUM_HEADS,
            },
            "intermediate_modules": [
                "FeatureProjection",
                "BidirectionalCrossAttention",
                "DiseaseConditionedAttention",
                "DiseaseRelationshipModel",
            ],
            "intermediate_note": (
                "Intermediate fusion modules have no persisted "
                "weights and are recreated with a fixed seed."
            ),
        }