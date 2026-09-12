"""
RESPIRA - Model Downloader

Downloads the required Respira model checkpoints from Hugging Face
when they are missing from the local models/ directory.

Models are intentionally NOT stored in GitHub.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

REPOSITORY_ID = "rushikeshtelrandhe/respira-models"

REQUIRED_MODELS = {
    "adaptive_fusion/best_model.pth": (
        MODELS_DIR / "adaptive_fusion" / "best_model.pth"
    ),
    "classification/best_model.pth": (
        MODELS_DIR / "classification" / "best_model.pth"
    ),
    "densenet/best_model.pth": (
        MODELS_DIR / "densenet" / "best_model.pth"
    ),
    "efficientnet_b0/best_model.pth": (
        MODELS_DIR / "efficientnet_b0" / "best_model.pth"
    ),
    "vit/best_model.pth": (
        MODELS_DIR / "vit" / "best_model.pth"
    ),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Format bytes as MB/GB."""

    mb = size_bytes / (1024 * 1024)

    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"

    return f"{mb:.2f} MB"


def check_models() -> tuple[list[str], list[str]]:
    """Return existing and missing model paths."""

    existing = []
    missing = []

    for remote_path, local_path in REQUIRED_MODELS.items():
        if local_path.is_file() and local_path.stat().st_size > 0:
            existing.append(remote_path)
        else:
            missing.append(remote_path)

    return existing, missing


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------

def download_models() -> bool:
    """Download all missing Respira checkpoints."""

    print()
    print("=" * 70)
    print("RESPIRA MODEL CHECK")
    print("=" * 70)
    print(f"Repository : {REPOSITORY_ID}")
    print(f"Models dir : {MODELS_DIR}")
    print()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing, missing = check_models()

    if existing:
        print("Already available:")
        for model in existing:
            print(f"  [OK] {model}")
        print()

    if not missing:
        print("All required Respira models are already available.")
        print("=" * 70)
        return True

    print("Missing models:")
    for model in missing:
        print(f"  [DOWNLOAD] {model}")

    print()
    print(
        "The download may require approximately 1.2 GB "
        "if all models are missing."
    )
    print()

    try:
        for remote_path in missing:
            local_path = REQUIRED_MODELS[remote_path]

            local_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("-" * 70)
            print(f"Downloading: {remote_path}")
            print(f"Destination: {local_path}")
            print("-" * 70)

            cached_file = hf_hub_download(
                repo_id=REPOSITORY_ID,
                filename=remote_path,
                repo_type="model",
            )

            cached_path = Path(cached_file)

            # Copy from Hugging Face cache to our deployment directory.
            import shutil

            shutil.copy2(
                cached_path,
                local_path,
            )

            size = local_path.stat().st_size

            print(
                f"[OK] Downloaded {remote_path} "
                f"({format_size(size)})"
            )
            print()

    except Exception as exc:
        print()
        print("=" * 70)
        print("MODEL DOWNLOAD FAILED")
        print("=" * 70)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print(
            "Respira cannot start until the required model "
            "checkpoints are available."
        )
        return False

    # Final verification
    _, still_missing = check_models()

    print("=" * 70)

    if still_missing:
        print("MODEL VERIFICATION FAILED")
        print()

        for model in still_missing:
            print(f"  [MISSING] {model}")

        print("=" * 70)

        return False

    print("ALL RESPIRA MODELS READY")
    print("=" * 70)
    print()

    return True


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    success = download_models()

    sys.exit(0 if success else 1)