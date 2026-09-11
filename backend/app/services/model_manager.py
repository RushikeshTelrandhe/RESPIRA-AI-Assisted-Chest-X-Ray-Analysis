"""ModelManager: load the REAL Respira checkpoints exactly once.

Wraps app.backend.pipeline.RespiraPipeline + GradCAMGenerator (which in turn
import src.models / src.fusion logic). No mock models, no fake warmup.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch

from app.backend import config as ml_config


# Selectable prediction models (keys accepted by POST /analyze).
# fusion/efficientnet/vit share the RespiraPipeline backbones;
# densenet loads its own checkpoint. Standalone backbones were trained
# with CrossEntropy (softmax); fusion uses sigmoid multi-label heads.
MODEL_REGISTRY: list[dict] = [
    {"key": "fusion", "name": "Respira Fusion",
     "version": "respira-fusion-1.0",
     "description": "Full EfficientNet-B0 + ViT-B/16 fusion pipeline (recommended)"},
    {"key": "efficientnet", "name": "EfficientNet-B0",
     "version": "efficientnet-b0-1.0",
     "description": "CNN baseline, fast single-model prediction"},
    {"key": "vit", "name": "ViT-B/16",
     "version": "vit-b16-1.0",
     "description": "Transformer baseline, attention-based prediction"},
    {"key": "densenet", "name": "DenseNet-121",
     "version": "densenet121-1.0",
     "description": "Research baseline CNN"},
]

MODEL_KEYS = [m["key"] for m in MODEL_REGISTRY]


class ModelManager:
    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- loading -----------------------------------------------------
    def _load(self) -> None:
        try:
            from app.backend.pipeline import RespiraPipeline
            from app.backend.gradcam import GradCAMGenerator

            self.pipeline = RespiraPipeline(device=self.device)
            self.pipeline.load()
            self.gradcam = GradCAMGenerator(device=self.device)
            self.gradcam.load()
            self.ready = True
        except Exception as exc:  # surfaced via /models/status, never a traceback to UI
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"
        try:
            from src.models.densenet import DenseNet121Model

            self.densenet = DenseNet121Model(num_classes=6, pretrained=False, dropout=0.2)
            ckpt_path = ml_config.PROJECT_ROOT / "outputs" / "densenet" / "checkpoints" / "best_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing DenseNet-121 checkpoint: {ckpt_path}")
            obj = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
            self.densenet.load_state_dict(state, strict=True)
            self.densenet.to(self.device)
            self.densenet.eval()
        except Exception as exc:
            self.densenet = None
            self.model_errors["densenet"] = f"{type(exc).__name__}: {exc}"
        self.research_metrics = self._read_research_metrics()

    @staticmethod
    def _read_research_metrics() -> dict:
        """Verified test-set metrics from the research pipeline output files."""
        candidates = [
            ml_config.OUTPUTS_DIR / "final_prediction" / "reports" / "final_metrics.json",
            ml_config.OUTPUTS_DIR / "final_prediction" / "final_summary.json",
        ]
        for path in candidates:
            try:
                if Path(path).exists():
                    return json.loads(Path(path).read_text())
            except Exception:
                continue
        return {}

    def is_model_ready(self, key: str) -> bool:
        if key in ("fusion", "efficientnet", "vit"):
            return bool(self.ready and self.pipeline is not None)
        if key == "densenet":
            return self.densenet is not None
        return False

    def available_models(self) -> list[dict]:
        out = []
        for m in MODEL_REGISTRY:
            err = None
            if m["key"] in ("fusion", "efficientnet", "vit"):
                if not (self.ready and self.pipeline is not None):
                    err = self.error or "Fusion pipeline is not loaded"
            else:
                err = self.model_errors.get(m["key"])
            out.append({**m, "loaded": err is None, "error": err})
        return out

    # -- status ------------------------------------------------------
    def status(self) -> dict:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        arch = {}
        if self.pipeline is not None:
            try:
                arch = self.pipeline.model_info().get("architecture", {})
            except Exception:
                arch = {}
        return {
            "respira_ai": "loaded" if self.ready else ("error" if self.error else "loading"),
            "backend": "ok",
            "device": str(self.device),
            "gpu_name": gpu_name,
            "model_loaded": self.ready,
            "error": self.error,
            "architecture": {
                "DenseNet121": "research-baseline",
                "EfficientNet-B0": arch.get("cnn", "EfficientNet-B0"),
                "ViT-B/16": arch.get("transformer", "ViT-B/16"),
                "BidirectionalCrossAttention": f"{arch.get('cross_attention_heads', 8)} heads",
                "DiseaseConditionedAttention": f"{arch.get('disease_attention_heads', 8)} heads",
                "AdaptiveDiseaseFusion": "trained-checkpoint" if self.ready else "unknown",
                "DiseaseRelationshipModeling": f"{arch.get('relationship_heads', 8)} heads",
                "MultiLabelClassificationHeads": "trained-checkpoint",
            },
            "classes": list(ml_config.CLASS_NAMES),
            "models": self.available_models(),
            "research_evaluation": self.research_metrics,
        }


def get_manager() -> ModelManager:
    return ModelManager.instance()
