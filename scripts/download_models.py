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
from huggingface_hub import hf_hub_download

REPO_ID = "rushikeshtelrandhe/respira-models"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    # 1. Finetuned Methodology Model (Primary Model: 96.4% ROC-AUC, 96.2% Top-2)
    "methodology_fusion/best_model.pth": [
        PROJECT_ROOT / "models/methodology_fusion/best_model.pth",
        PROJECT_ROOT / "outputs/methodology_fusion/checkpoints/best_model.pth",
    ],

    # 2. Standalone 512 Backbones
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
    print(f"Hugging Face repository: {REPO_ID}")
    print()

    for repo_file, target_paths in FILES.items():
        print(f"Downloading: {repo_file}")

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
                print(f"  ✓ Saved: {target_path}")

        except Exception as exc:
            print(f"  [!] Skipped {repo_file} (or not yet uploaded to HF repo): {exc}")
        print()

    print("=" * 70)
    print("MODEL DOWNLOAD PROCESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
