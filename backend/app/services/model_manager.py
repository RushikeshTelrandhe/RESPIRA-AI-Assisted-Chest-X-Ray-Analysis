"""ModelManager: load the REAL Respira checkpoints exactly once.

Wraps app.backend.pipeline.RespiraPipeline + GradCAMGenerator (which in turn
import src.models / src.fusion logic).

No mock models, no fake warmup.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch

from app.backend import config as ml_config


# Selectable prediction models (keys accepted by POST /analyze).
#
# fusion/efficientnet/vit share the RespiraPipeline backbones.
# densenet loads its own standalone checkpoint.
#
# Fusion uses sigmoid multi-label heads.
# Standalone CNN baselines use their respective trained checkpoints.
MODEL_REGISTRY: list[dict] = [
    {
        "key": "fusion",
        "name": "Respira Fusion",
        "version": "respira-fusion-1.0",
        "description": (
            "Full EfficientNet-B0 + ViT-B/16 fusion pipeline (recommended)"
        ),
    },
    {
        "key": "efficientnet",
        "name": "EfficientNet-B0",
        "version": "efficientnet-b0-1.0",
        "description": "CNN baseline, fast single-model prediction",
    },
    {
        "key": "vit",
        "name": "ViT-B/16",
        "version": "vit-b16-1.0",
        "description": "Transformer baseline, attention-based prediction",
    },
    {
        "key": "densenet",
        "name": "DenseNet-121",
        "version": "densenet121-1.0",
        "description": "Research baseline CNN",
    },
]

MODEL_KEYS = [m["key"] for m in MODEL_REGISTRY]


class ModelManager:
    """Singleton model manager.

    Loads all required models once at backend startup and keeps them
    resident in memory for inference.
    """

    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.pipeline = None
        self.gradcam = None
        self.densenet = None

        self.ready = False
        self.error: str | None = None
        self.model_errors: dict[str, str] = {}
        self.research_metrics: dict = {}

        self._load()

    @classmethod
    def instance(cls) -> "ModelManager":
        """Return the singleton ModelManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()

        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the complete Respira model stack."""

        # --------------------------------------------------------------
        # Main Respira Fusion pipeline + Grad-CAM
        # --------------------------------------------------------------
        try:
            from app.backend.pipeline import RespiraPipeline
            from app.backend.gradcam import GradCAMGenerator

            self.pipeline = RespiraPipeline(device=self.device)
            self.pipeline.load()

            self.gradcam = GradCAMGenerator(device=self.device)
            self.gradcam.load()

            self.ready = True

        except Exception as exc:
            # Error is exposed through /models/status.
            # Do not expose raw traceback information to the frontend.
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"

        # --------------------------------------------------------------
        # DenseNet-121 standalone research baseline
        # --------------------------------------------------------------
        try:
            from src.models.densenet import DenseNet121Model

            self.densenet = DenseNet121Model(
                num_classes=6,
                pretrained=False,
                dropout=0.2,
            )

            # Deployment checkpoint location.
            # This is now stored under:
            #
            # models/densenet/best_model.pth
            #
            # instead of:
            #
            # outputs/densenet/checkpoints/best_model.pth
            ckpt_path = ml_config.DENSENET_CHECKPOINT

            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing DenseNet-121 checkpoint: {ckpt_path}"
                )

            obj = torch.load(
                str(ckpt_path),
                map_location=self.device,
                weights_only=False,
            )

            state = (
                obj.get("model_state_dict", obj)
                if isinstance(obj, dict)
                else obj
            )

            self.densenet.load_state_dict(
                state,
                strict=True,
            )

            self.densenet.to(self.device)
            self.densenet.eval()

        except Exception as exc:
            self.densenet = None
            self.model_errors["densenet"] = (
                f"{type(exc).__name__}: {exc}"
            )

        # --------------------------------------------------------------
        # Research evaluation metrics
        # --------------------------------------------------------------
        self.research_metrics = self._read_research_metrics()

    # ------------------------------------------------------------------
    # Research metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _read_research_metrics() -> dict:
        """Read verified test-set metrics from research pipeline outputs."""

        candidates = [
            (
                ml_config.OUTPUTS_DIR
                / "final_prediction"
                / "reports"
                / "final_metrics.json"
            ),
            (
                ml_config.OUTPUTS_DIR
                / "final_prediction"
                / "final_summary.json"
            ),
        ]

        for path in candidates:
            try:
                if Path(path).exists():
                    return json.loads(
                        Path(path).read_text()
                    )
            except Exception:
                continue

        return {}

    # ------------------------------------------------------------------
    # Model status
    # ------------------------------------------------------------------

    def is_model_ready(self, key: str) -> bool:
        """Return whether a requested model is ready for inference."""

        if key in ("fusion", "efficientnet", "vit"):
            return bool(
                self.ready
                and self.pipeline is not None
            )

        if key == "densenet":
            return self.densenet is not None

        return False

    def available_models(self) -> list[dict]:
        """Return model registry with current loading status."""

        out = []

        for model in MODEL_REGISTRY:
            key = model["key"]
            err = None

            if key in ("fusion", "efficientnet", "vit"):
                if not (
                    self.ready
                    and self.pipeline is not None
                ):
                    err = (
                        self.error
                        or "Fusion pipeline is not loaded"
                    )

            elif key == "densenet":
                err = self.model_errors.get(key)

            out.append(
                {
                    **model,
                    "loaded": err is None,
                    "error": err,
                }
            )

        return out

    # ------------------------------------------------------------------
    # Backend / model status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return complete backend and model status."""

        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )

        arch = {}

        if self.pipeline is not None:
            try:
                arch = (
                    self.pipeline
                    .model_info()
                    .get("architecture", {})
                )
            except Exception:
                arch = {}

        return {
            "respira_ai": (
                "loaded"
                if self.ready
                else ("error" if self.error else "loading")
            ),
            "backend": "ok",
            "device": str(self.device),
            "gpu_name": gpu_name,
            "model_loaded": self.ready,
            "error": self.error,
            "architecture": {
                "DenseNet121": "research-baseline",
                "EfficientNet-B0": arch.get(
                    "cnn",
                    "EfficientNet-B0",
                ),
                "ViT-B/16": arch.get(
                    "transformer",
                    "ViT-B/16",
                ),
                "BidirectionalCrossAttention": (
                    f"{arch.get('cross_attention_heads', 8)} heads"
                ),
                "DiseaseConditionedAttention": (
                    f"{arch.get('disease_attention_heads', 8)} heads"
                ),
                "AdaptiveDiseaseFusion": (
                    "trained-checkpoint"
                    if self.ready
                    else "unknown"
                ),
                "DiseaseRelationshipModeling": (
                    f"{arch.get('relationship_heads', 8)} heads"
                ),
                "MultiLabelClassificationHeads": (
                    "trained-checkpoint"
                ),
            },
            "classes": list(ml_config.CLASS_NAMES),
            "models": self.available_models(),
            "research_evaluation": self.research_metrics,
        }


def get_manager() -> ModelManager:
    """Return the global singleton ModelManager."""
    return ModelManager.instance()  