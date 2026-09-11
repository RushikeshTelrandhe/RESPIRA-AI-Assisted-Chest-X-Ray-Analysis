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


class ModelManager:
    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.gradcam = None
        self.ready = False
        self.error: str | None = None
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
            "research_evaluation": self.research_metrics,
        }


def get_manager() -> ModelManager:
    return ModelManager.instance()
