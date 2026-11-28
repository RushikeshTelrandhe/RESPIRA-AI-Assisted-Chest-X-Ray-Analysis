"""
RESPIRA - Download Model Checkpoints
====================================
Downloads model checkpoints from Hugging Face Hub:
  1. EfficientNet-B0 (512x512)
  2. Vision Transformer ViT-B/16 (512x512)
  3. Weighted Fusion 512 configuration
  4. Finetuned Respira Fusion (Dual-Backbone Cross-Attention + GCN, 512x512)

Usage:
  python scripts/download_models.py
"""

from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "rushikeshtelrandhe/respira-models"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    # 1. Finetuned Methodology Model (Primary Model: 96.4% ROC-AUC, 96.2% Top-2)
    "methodology_fusion/best_model.pth": [
        PROJECT_ROOT / "models/methodology_fusion/best_model.pth",
        PROJECT_ROOT / "outputs/methodology_fusion/checkpoints/best_model.pth",
    ],
    "methodology_fusion/metrics.json": [
        PROJECT_ROOT / "outputs/methodology_fusion/evaluation/metrics.json",
    ],

    # 2. Standalone 512 Backbones (Optional / Baselines)
    "efficientnet_b0_512/best_model.pth": [
        PROJECT_ROOT / "outputs/efficientnet_b0_512/checkpoints/best_model.pth",
    ],
    "vit_512/best_model.pth": [
        PROJECT_ROOT / "outputs/vit_512/checkpoints/best_model.pth",
    ],
    "efficientnet_b0_512/metrics.json": [
        PROJECT_ROOT / "outputs/efficientnet_b0_512/evaluation/metrics.json",
    ],
    "vit_512/metrics.json": [
        PROJECT_ROOT / "outputs/vit_512/evaluation/metrics.json",
    ],
    "fusion_512/fusion_config.json": [
        PROJECT_ROOT / "outputs/fusion/fusion_config.json",
    ],
}


def main():
    print("=" * 70)
    print("               RESPIRA — DOWNLOAD MODEL CHECKPOINTS")
    print("=" * 70)
    print(f"Hugging Face repository: https://huggingface.co/{REPO_ID}")
    print()

    primary_model_ok = False
    methodology_path = PROJECT_ROOT / "models" / "methodology_fusion" / "best_model.pth"
    methodology_alt = PROJECT_ROOT / "outputs" / "methodology_fusion" / "checkpoints" / "best_model.pth"

    # Fast sync if already locally present in either location
    if methodology_path.exists() and not methodology_alt.exists():
        methodology_alt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(methodology_path, methodology_alt)
    elif methodology_alt.exists() and not methodology_path.exists():
        methodology_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(methodology_alt, methodology_path)

    if methodology_path.exists() or methodology_alt.exists():
        primary_model_ok = True

    for repo_file, target_paths in FILES.items():
        # Check if already present locally
        already_all_present = all(p.exists() and p.stat().st_size > 0 for p in target_paths)
        if already_all_present:
            print(f"Already present locally: {repo_file}")
            for p in target_paths:
                print(f"  ✓ Found: {p.relative_to(PROJECT_ROOT)}")
            print()
            continue

        print(f"Downloading: {repo_file} from {REPO_ID}...")

        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file,
                local_dir=PROJECT_ROOT / ".hf_cache",
            )
            downloaded_path = Path(downloaded)

            for target_path in target_paths:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_path, target_path)
                print(f"  ✓ Saved: {target_path.relative_to(PROJECT_ROOT)}")

            if "methodology_fusion" in repo_file:
                primary_model_ok = True

        except Exception as exc:
            print(f"  [!] Skipped {repo_file} (not yet available in {REPO_ID}): {exc}")
        print()

    print("=" * 70)
    if primary_model_ok or methodology_path.exists() or methodology_alt.exists():
        print("✓ SUCCESS: Primary RESPIRA Methodology Model is ready.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("[!] ACTION REQUIRED: Methodology model checkpoint is missing!")
        print(f"    Expected: {methodology_path.relative_to(PROJECT_ROOT)}")
        print(f"    Target repo: https://huggingface.co/{REPO_ID}")
        print()
        print("    To upload the model from a machine where it exists, run:")
        print("      python scripts/create_hf_pull_request.py")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
