"""Pytest bootstrap: isolated SQLite DB per session, backend imports after env setup."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_DB = ROOT / "backend_storage" / "test_pytest.db"
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["JWT_SECRET"] = "pytest-secret"

from backend.app.database import init_db  # noqa: E402

init_db()
