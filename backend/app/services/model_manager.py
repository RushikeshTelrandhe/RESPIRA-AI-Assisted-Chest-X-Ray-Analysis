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
# fusion/efficientnet/vit share the RespiraPipeline backbones (224x224).
# efficientnet-512/vit-512 are high-resolution 512x512 backbones.
# fusion-512 is a weighted probability fusion of the two 512 backbones
# (weights from outputs/fusion/fusion_config.json).
# densenet loads its own standalone checkpoint.
#
# Fusion (224) uses sigmoid multi-label heads.
# Standalone baselines use softmax, as trained with CrossEntropy.
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
    {
        "key": "efficientnet-512",
        "name": "EfficientNet-B0 (512)",
        "version": "efficientnet-b0-512-1.0",
        "description": "High-resolution 512x512 CNN (92.4% test accuracy)",
    },
    {
        "key": "vit-512",
        "name": "ViT-B/16 (512)",
        "version": "vit-b16-512-1.0",
        "description": "High-resolution 512x512 transformer (91.3% test accuracy)",
    },
    {
        "key": "fusion-512",
        "name": "Respira Fusion (512)",
        "version": "respira-fusion-512-1.0",
        "description": "Weighted 512x512 EfficientNet + ViT fusion (92.9% val accuracy)",
    },
]

MODEL_KEYS = [m["key"] for m in MODEL_REGISTRY]

# High-resolution 512x512 checkpoint candidates (first existing wins).
def _candidates(*paths: Path) -> list[Path]:
    return list(paths)


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


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

        # High-resolution 512x512 backbones + fusion weights.
        self.efficientnet_512 = None
        self.vit_512 = None
        self.fusion_512_weights: dict = {}

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
        # High-resolution 512x512 backbones
        # (EfficientNet-B0 512 + ViT-B/16 512 + weighted fusion)
        # --------------------------------------------------------------
        try:
            self._load_512_models()
        except Exception as exc:
            self.model_errors["efficientnet-512"] = f"{type(exc).__name__}: {exc}"
            self.model_errors["vit-512"] = f"{type(exc).__name__}: {exc}"
            self.model_errors["fusion-512"] = f"{type(exc).__name__}: {exc}"

        # --------------------------------------------------------------
        # Research evaluation metrics
        # --------------------------------------------------------------
        self.research_metrics = self._read_research_metrics()

    def _load_512_models(self) -> None:
        """Load 512x512 EfficientNet + ViT backbones and fusion weights."""
        from src.models.efficientnet import EfficientNetB0Classifier
        from src.models.vit import VisionTransformerModel

        root = ml_config.PROJECT_ROOT

        eff_candidates = _candidates(
            root / "outputs" / "efficientnet_b0_512" / "checkpoints" / "best_model.pth",
            root / ".hf_cache" / "efficientnet_b0_512" / "best_model.pth",
        )
        vit_candidates = _candidates(
            root / "outputs" / "vit_512" / "checkpoints" / "best_model.pth",
            root / ".hf_cache" / "vit_512" / "best_model.pth",
        )
        fusion_cfg_candidates = _candidates(
            root / "outputs" / "fusion" / "fusion_config.json",
            root / ".hf_cache" / "fusion_512" / "fusion_config.json",
        )

        # --- EfficientNet-B0 512 ---
        try:
            ckpt_path = _first_existing(eff_candidates)
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Missing EfficientNet-B0 512 checkpoint: "
                    + ", ".join(str(p) for p in eff_candidates)
                )
            model = EfficientNetB0Classifier(
                num_classes=6, pretrained=False, dropout=0.3
            )
            obj = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            self.efficientnet_512 = model
        except Exception as exc:
            self.efficientnet_512 = None
            self.model_errors["efficientnet-512"] = f"{type(exc).__name__}: {exc}"

        # --- ViT-B/16 512 ---
        try:
            ckpt_path = _first_existing(vit_candidates)
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Missing ViT-B/16 512 checkpoint: "
                    + ", ".join(str(p) for p in vit_candidates)
                )
            model = VisionTransformerModel(
                num_classes=6, pretrained=False
            )
            obj = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            self.vit_512 = model
        except Exception as exc:
            self.vit_512 = None
            self.model_errors["vit-512"] = f"{type(exc).__name__}: {exc}"

        # --- Fusion-512 weights (weighted probability fusion) ---
        try:
            cfg_path = _first_existing(fusion_cfg_candidates)
            if cfg_path is not None:
                cfg = json.loads(cfg_path.read_text())
                eff_w = float(cfg.get("efficientnet_weight", 0.35))
                vit_w = float(cfg.get("vit_weight", 0.65))
            else:
                eff_w, vit_w = 0.35, 0.65
            total = eff_w + vit_w
            self.fusion_512_weights = {
                "efficientnet_weight": eff_w / total,
                "vit_weight": vit_w / total,
            }
            if self.efficientnet_512 is None or self.vit_512 is None:
                missing = [
                    k for k, v in
                    (("efficientnet-512", self.efficientnet_512), ("vit-512", self.vit_512))
                    if v is None
                ]
                raise RuntimeError(
                    "Fusion-512 unavailable, missing backbones: " + ", ".join(missing)
                )
        except Exception as exc:
            if "fusion-512" not in self.model_errors:
                self.model_errors["fusion-512"] = f"{type(exc).__name__}: {exc}"

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

        if key == "efficientnet-512":
            return self.efficientnet_512 is not None

        if key == "vit-512":
            return self.vit_512 is not None

        if key == "fusion-512":
            return (
                self.efficientnet_512 is not None
                and self.vit_512 is not None
            )

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

            elif key in ("densenet", "efficientnet-512", "vit-512", "fusion-512"):
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
                "EfficientNet-B0-512": (
                    "trained-checkpoint"
                    if self.efficientnet_512 is not None
                    else "missing"
                ),
                "ViT-B/16-512": (
                    "trained-checkpoint"
                    if self.vit_512 is not None
                    else "missing"
                ),
                "Fusion-512": (
                    f"weighted-{self.fusion_512_weights.get('efficientnet_weight', 0.35):.2f}/"
                    f"{self.fusion_512_weights.get('vit_weight', 0.65):.2f}"
                    if self.fusion_512_weights
                    else "missing"
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