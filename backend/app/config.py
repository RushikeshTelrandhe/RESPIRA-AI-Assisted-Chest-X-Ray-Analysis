"""Respira backend configuration.

Environment variables (see .env.example):
  DATABASE_URL   - SQLAlchemy URL. Defaults to local SQLite fallback.
                   Use postgresql+psycopg://user:pass@host:5432/respira for Postgres.
  JWT_SECRET     - HS256 signing secret (REQUIRED in production).
  BACKEND_HOST/PORT, FRONTEND_URL (CORS), STORAGE_DIR, RETAIN_IMAGES.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]  # backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Respira/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "RESPIRA"
    APP_TAGLINE: str = "AI-Assisted Chest X-Ray Analysis"
    API_PREFIX: str = "/api/v1"
    APP_VERSION: str = "1.0.0"

    DATABASE_URL: str = ""
    JWT_SECRET: str = "dev-only-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 8

    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:5173"

    STORAGE_DIR: str = ""
    RETAIN_IMAGES: bool = True
    MAX_UPLOAD_MB: int = 15


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.DATABASE_URL:
        s.DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'backend_storage' / 'respira.db').as_posix()}"
    if not s.STORAGE_DIR:
        s.STORAGE_DIR = (PROJECT_ROOT / "backend_storage").as_posix()
    return s


settings = get_settings()
STORAGE_ROOT = Path(settings.STORAGE_DIR)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
(STORAGE_ROOT / "temp").mkdir(parents=True, exist_ok=True)
(STORAGE_ROOT / "analyses").mkdir(parents=True, exist_ok=True)
(STORAGE_ROOT / "reports").mkdir(parents=True, exist_ok=True)
