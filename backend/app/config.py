"""Respira backend configuration.

Environment variables (see .env.example):
  DATABASE_URL   - SQLAlchemy URL. Defaults to local SQLite fallback.
                   Use postgresql+psycopg://user:pass@host:5432/respira for Postgres.
  JWT_SECRET     - HS256 signing secret (REQUIRED in production).
  BACKEND_HOST/PORT, FRONTEND_URL (CORS), STORAGE_DIR, RETAIN_IMAGES.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# PROJECT PATHS
# ============================================================================

# C:\Users\rushi\Desktop\Respira\app\backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# C:\Users\rushi\Desktop\Respira
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# DEPLOYMENT MODEL CHECKPOINTS
# ============================================================================

# All application/deployment checkpoints are stored here:
#
# Respira/
# └── models/
#     ├── densenet/
#     │   └── best_model.pth
#     ├── efficientnet_b0/
#     │   └── best_model.pth
#     ├── vit/
#     │   └── best_model.pth
#     ├── adaptive_fusion/
#     │   └── best_model.pth
#     └── classification/
#         └── best_model.pth

MODELS_DIR = PROJECT_ROOT / "models"

# Standalone / backbone checkpoints
EFFICIENTNET_CHECKPOINT = (
    MODELS_DIR / "efficientnet_b0" / "best_model.pth"
)

VIT_CHECKPOINT = (
    MODELS_DIR / "vit" / "best_model.pth"
)

DENSENET_CHECKPOINT = (
    MODELS_DIR / "densenet" / "best_model.pth"
)

# Fusion pipeline checkpoints
ADAPTIVE_FUSION_CHECKPOINT = (
    MODELS_DIR / "adaptive_fusion" / "best_model.pth"
)

CLASSIFICATION_CHECKPOINT = (
    MODELS_DIR / "classification" / "best_model.pth"
)
DENSENET_CHECKPOINT = (
    MODELS_DIR
    / "densenet"
    / "best_model.pth"
)

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "RESPIRA"
    APP_TAGLINE: str = "AI-Assisted Chest X-Ray Analysis"
    API_PREFIX: str = "/api/v1"
    APP_VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = ""

    # Authentication
    JWT_SECRET: str = "dev-only-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 8

    # Backend
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000

    # Frontend / CORS
    FRONTEND_URL: str = "http://localhost:5173"

    # Storage
    STORAGE_DIR: str = ""
    RETAIN_IMAGES: bool = True
    MAX_UPLOAD_MB: int = 15


# ============================================================================
# SETTINGS
# ============================================================================

@lru_cache
def get_settings() -> Settings:
    """Load application settings and apply local defaults."""

    s = Settings()

    # Default local SQLite database
    if not s.DATABASE_URL:
        s.DATABASE_URL = (
            "sqlite:///"
            f"{(PROJECT_ROOT / 'backend_storage' / 'respira.db').as_posix()}"
        )

    # Default local application storage
    if not s.STORAGE_DIR:
        s.STORAGE_DIR = (
            PROJECT_ROOT / "backend_storage"
        ).as_posix()

    return s


settings = get_settings()


# ============================================================================
# STORAGE DIRECTORIES
# ============================================================================

STORAGE_ROOT = Path(settings.STORAGE_DIR)

STORAGE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

(STORAGE_ROOT / "temp").mkdir(
    parents=True,
    exist_ok=True,
)

(STORAGE_ROOT / "analyses").mkdir(
    parents=True,
    exist_ok=True,
)

(STORAGE_ROOT / "reports").mkdir(
    parents=True,
    exist_ok=True,
)