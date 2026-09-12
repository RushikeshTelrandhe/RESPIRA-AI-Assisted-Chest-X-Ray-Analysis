"""
RESPIRA - Application Configuration
Central configuration for the backend service.
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Research outputs remain in outputs/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Deployment model checkpoints are stored separately.
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

EFFICIENTNET_CHECKPOINT = (
    MODELS_DIR
    / "efficientnet_b0"
    / "best_model.pth"
)

VIT_CHECKPOINT = (
    MODELS_DIR
    / "vit"
    / "best_model.pth"
)

ADAPTIVE_FUSION_CHECKPOINT = (
    MODELS_DIR
    / "adaptive_fusion"
    / "best_model.pth"
)

CLASSIFICATION_CHECKPOINT = (
    MODELS_DIR
    / "classification"
    / "best_model.pth"
)

DENSENET_CHECKPOINT = MODELS_DIR / "densenet" / "best_model.pth"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 6

FEATURE_DIM = 512

HIDDEN_DIM = 256

EMBED_DIM = 512

NUM_HEADS = 8

THRESHOLD = 0.5

SEED = 42

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

# Clinical guidance shown alongside each prediction.
CLASS_DESCRIPTIONS = {
    "Atelectasis": (
        "Collapse or closure of lung tissue resulting in "
        "reduced or absent gas exchange. Often appears as "
        "increased opacity in the affected region."
    ),
    "Bacterial Pneumonia": (
        "Lung infection caused by bacteria. Typically presents "
        "with lobar consolidation and air bronchograms on "
        "chest X-ray."
    ),
    "Normal": (
        "No abnormal findings detected. The lungs display "
        "normal aeration without focal opacities, effusions, "
        "or consolidation."
    ),
    "Pulmonary Edema": (
        "Accumulation of fluid in the lung airspaces. "
        "Characteristic bilateral perihilar opacities and "
        "Kerley B lines may be present."
    ),
    "Tuberculosis": (
        "Bacterial infection commonly affecting the upper lung "
        "zones. May present with cavitation, fibronodular "
        "opacities, or hilar lymphadenopathy."
    ),
    "Viral Pneumonia": (
        "Lung infection caused by viral pathogens. Often shows "
        "diffuse bilateral interstitial or reticular patterns "
        "rather than lobar consolidation."
    ),
}

# Uncertainty categories (from final_prediction.py thresholds).
LOW_UNCERTAINTY = 0.20
MODERATE_UNCERTAINTY = 0.40

# CLAHE and mask preprocessing (from the fixed preprocessing notebook).
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "RESPIRA"

APP_TAGLINE = "AI-Assisted Chest X-Ray Analysis"

API_PREFIX = "/api"