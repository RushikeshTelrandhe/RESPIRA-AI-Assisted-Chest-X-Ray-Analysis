"""
RESPIRA - Create Pull Request on Hugging Face Hub
=================================================
Copies the local methodology fusion model files and creates a Pull Request
on the target Hugging Face model repository (e.g. `rushikeshtelrandhe/respira-models`).

Why this approach:
- You DO NOT need write access or ownership of the target repository!
  Anyone with a free Hugging Face account can open a community Pull Request.
- Avoids downloading the existing 2.6 GB of weights in the target repo.
- Uploads `methodology_fusion/best_model.pth` and `metrics.json`.
- Automatically opens the PR discussion on Hugging Face with full benchmark notes.
- Your friend (the repo owner) gets a notification and can merge it in 1 click!

Prerequisites:
  Log in to your Hugging Face account in terminal:
    hf auth login
  (Or pass your token via --token <YOUR_HF_TOKEN>)

Usage:
  python scripts/create_hf_pull_request.py
  python scripts/create_hf_pull_request.py --repo rushikeshtelrandhe/respira-models
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import CommitOperationAdd, HfApi
except ImportError:
    print("[!] Error: huggingface_hub is not installed.")
    print("    Install it with: pip install huggingface_hub")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO = "rushikeshtelrandhe/respira-models"

PR_DESCRIPTION = """## Add Finetuned Respira Fusion 512x512 Methodology Model

### Overview
This pull request adds the checkpoint and evaluation metrics for the **Finetuned Respira Fusion (Methodology Model)** under `methodology_fusion/`.

### Architecture & Methodology
- **Input Resolution**: 512 × 512 unmasked chest radiograph (CLAHE enhanced)
- **Dual Backbones**: EfficientNet-B0 (focal lesions) + Vision Transformer ViT-B/16 (1024 spatial tokens)
- **Feature Fusion**: Bidirectional Cross-Attention (1280 fused tokens)
- **Pathology Graph**: Relational Graph Convolutional Network (GCN) for clinical co-occurrence modeling
- **Uncertainty**: 6-class independent heads with calibrated Monte Carlo Dropout (T=10)

### Verified Holdout Test Benchmarks (1,697 Holdout Radiographs)
- **Top-2 Diagnostic Accuracy**: **96.35%** (Exceeds 95.0% clinical triage benchmark)
- **Macro ROC-AUC**: **96.43%** (One-vs-Rest across all 6 classes)
- **Clinical Grouped Accuracy**: **92.16%** (Pneumonias & Opacities unified)
- **Pneumonia Unified Accuracy**: **87.33%** (Bacterial + Viral unified)
- **Average Specificity**: **95.62%** (Healthy lung specificity: **98.51%**)
- **Tuberculosis Detection**: Sensitivity **93.75%**, ROC-AUC **0.9907** (0 misclassified as Normal)
- **Inference Speed**: ~56.5 ms / radiograph (17.7 img/sec on NVIDIA RTX GPU)

### Files Added:
- `methodology_fusion/best_model.pth` (~387 MB checkpoint)
- `methodology_fusion/metrics.json` (evaluation metrics summary)

This allows `start.bat` and `scripts/download_models.py` to auto-download the primary methodology model for all contributors.
"""


def locate_files():
    """Locate local model and metric files."""
    # Checkpoint
    ckpt_candidates = [
        PROJECT_ROOT / "models" / "methodology_fusion" / "best_model.pth",
        PROJECT_ROOT / "outputs" / "methodology_fusion" / "checkpoints" / "best_model.pth",
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break

    # Metrics
    metrics_path = PROJECT_ROOT / "outputs" / "methodology_fusion" / "evaluation" / "metrics.json"

    return ckpt_path, metrics_path if metrics_path.exists() else None


def main():
    parser = argparse.ArgumentParser(
        description="Create a Pull Request with the methodology model on a Hugging Face repository."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"Target Hugging Face repository ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional if logged in via `hf auth login`)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Add Finetuned Respira Fusion 512x512 methodology model",
        help="Pull Request title",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("       RESPIRA — CREATE PULL REQUEST ON HUGGING FACE HUB")
    print("=" * 72)
    print(f"Target Repository : {args.repo}")
    print()

    # 1. Locate files
    ckpt_path, metrics_path = locate_files()
    if not ckpt_path:
        print("[ERROR] Could not find 'best_model.pth' locally!")
        print("Expected at one of:")
        print("  - models/methodology_fusion/best_model.pth")
        print("  - outputs/methodology_fusion/checkpoints/best_model.pth")
        sys.exit(1)

    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"[1/3] Local files to submit in Pull Request:")
    print(f"  ✓ Checkpoint : {ckpt_path.name} ({file_size_mb:.1f} MB) ➔ methodology_fusion/best_model.pth")
    if metrics_path:
        print(f"  ✓ Metrics    : {metrics_path.name} ➔ methodology_fusion/metrics.json")
    print()

    # 2. Check authentication
    print("[2/3] Checking Hugging Face credentials...")
    api = HfApi(token=args.token)
    try:
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        print(f"  ✓ Authenticated as: {username}")
    except Exception as exc:
        print(f"[!] Authentication error: {exc}")
        print("\nPlease log in to Hugging Face before running this script:")
        print("  hf auth login")
        print("Or pass your token:")
        print("  python scripts/create_hf_pull_request.py --token <YOUR_HF_TOKEN>")
        sys.exit(1)

    # 3. Build operations
    operations = [
        CommitOperationAdd(
            path_in_repo="methodology_fusion/best_model.pth",
            path_or_fileobj=str(ckpt_path),
        )
    ]
    if metrics_path:
        operations.append(
            CommitOperationAdd(
                path_in_repo="methodology_fusion/metrics.json",
                path_or_fileobj=str(metrics_path),
            )
        )

    # 4. Create Pull Request
    print(f"\n[3/3] Uploading and creating Pull Request on '{args.repo}'...")
    print("  (This may take 1-3 minutes depending on your internet upload speed)...")

    try:
        commit_info = api.create_commit(
            repo_id=args.repo,
            repo_type="model",
            operations=operations,
            commit_message=args.title,
            commit_description=PR_DESCRIPTION,
            create_pr=True,
        )

        pr_url = commit_info.pr_url or f"https://huggingface.co/{args.repo}/discussions"
        pr_num = getattr(commit_info, "pr_num", None)

        print()
        print("=" * 72)
        print("  ✓ PULL REQUEST CREATED SUCCESSFULLY!")
        print("=" * 72)
        if pr_num:
            print(f"PR Number    : #{pr_num}")
        print(f"PR Link      : {pr_url}")
        print("=" * 72)
        print("\nNext step for your friend:")
        print(f"1. Open the PR link: {pr_url}")
        print("2. Review the files (`methodology_fusion/best_model.pth`).")
        print("3. Click 'Merge pull request'.")
        print("\nOnce merged, everyone can auto-download the model via start.bat!")
        print("=" * 72)

    except Exception as exc:
        print(f"\n[ERROR] Failed to create Pull Request: {exc}")
        print("Please check your network connection and token permissions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
