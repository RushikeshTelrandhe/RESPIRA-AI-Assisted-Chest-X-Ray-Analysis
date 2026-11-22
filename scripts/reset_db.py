"""RESPIRA - Reset Database Script
================================
Safely resets the local database (or drops and recreates all tables)
and optionally clears stored patient X-rays and PDF reports.

Usage:
  # Reset database tables only:
  python scripts/reset_db.py

  # Reset database tables AND clear uploaded X-rays/reports:
  python scripts/reset_db.py --clean-storage

  # Non-interactive (bypass confirmation prompt):
  python scripts/reset_db.py --yes
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.database import Base, engine, init_db


def reset_database(clean_storage: bool = False, skip_confirm: bool = False) -> None:
    print("=" * 65)
    print("                RESPIRA — DATABASE RESET TOOL")
    print("=" * 65)
    print(f"Target Database : {settings.DATABASE_URL}")
    print(f"Storage Dir     : {settings.STORAGE_DIR}")
    print(f"Clean Storage   : {'YES (X-rays & PDFs will be deleted)' if clean_storage else 'NO (DB records only)'}")
    print("=" * 65)

    if not skip_confirm:
        confirm = input("\n[WARNING] This will permanently delete all doctors, patients,\nstudies, and analysis records. Proceed? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Operation aborted. No changes were made.")
            return

    print("\n[1/3] Dropping all database tables...")
    try:
        from backend.app import models_db  # noqa: F401
        Base.metadata.drop_all(bind=engine)
        print("  ✓ All existing tables dropped.")
    except Exception as exc:
        print(f"  [!] Note during table drop: {exc}")

    # If SQLite file exists, remove to ensure clean schema rebuild
    db_file = Path(PROJECT_ROOT / "backend_storage" / "respira.db")
    if settings.DATABASE_URL.startswith("sqlite") and db_file.exists():
        try:
            engine.dispose()
            db_file.unlink()
            print(f"  ✓ Deleted SQLite database file: {db_file.name}")
        except Exception as exc:
            print(f"  [!] Could not delete db file directly (recreating via DDL): {exc}")

    print("\n[2/3] Re-creating fresh database schema...")
    try:
        init_db()
        print("  ✓ Fresh database schema created successfully.")
    except Exception as exc:
        print(f"  [ERROR] Failed to recreate schema: {exc}")
        sys.exit(1)

    if clean_storage:
        print("\n[3/3] Clearing stored files in backend_storage/...")
        storage_path = Path(settings.STORAGE_DIR)
        for folder_name in ("analyses", "reports", "temp"):
            folder = storage_path / folder_name
            if folder.exists():
                try:
                    for item in folder.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print(f"  ✓ Cleared {folder_name}/")
                except Exception as exc:
                    print(f"  [!] Error clearing {folder_name}/: {exc}")
    else:
        print("\n[3/3] Skipped storage folder cleanup (files preserved).")

    print("\n" + "=" * 65)
    print("  ✓ Database reset complete! The system is now fresh.")
    print("    You can now register a new doctor account at /signup.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Reset RESPIRA database.")
    parser.add_argument(
        "--clean-storage",
        action="store_true",
        help="Also delete uploaded X-ray images, Grad-CAM overlays, and generated PDF reports in backend_storage/.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()
    reset_database(clean_storage=args.clean_storage, skip_confirm=args.yes)


if __name__ == "__main__":
    main()
