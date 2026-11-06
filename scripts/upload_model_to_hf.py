"""
RESPIRA - Upload Finetuned Methodology Model to Hugging Face Hub
===============================================================
Uploads the local trained checkpoint (`models/methodology_fusion/best_model.pth`)
to a Hugging Face model repository so that any teammate who clones the repo
can download it using `python scripts/download_models.py`.

Prerequisites:
  1. pip install huggingface_hub
  2. Log in with your Hugging Face write token:
       huggingface-cli login
     OR pass your token via --token <HF_TOKEN>

Usage:
  python scripts/upload_model_to_hf.py --repo rushikeshtelrandhe/respira-models
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_file
except ImportError:
    print("[!] Error: huggingface_hub is not installed.")
    print("    Install it with: pip install huggingface_hub")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Upload trained RESPIRA checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        type=str,
        default="rushikeshtelrandhe/respira-models",
        help="Hugging Face repo ID (e.g. 'username/repo-name')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models/methodology_fusion/best_model.pth"),
        help="Path to local .pth checkpoint to upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token with write permission (optional if logged in via huggingface-cli)",
    )
    args = parser.parse_args()

    local_ckpt = Path(args.checkpoint)
    if not local_ckpt.exists():
        # Fallback to outputs folder
        fallback = PROJECT_ROOT / "outputs/methodology_fusion/checkpoints/best_model.pth"
        if fallback.exists():
            local_ckpt = fallback
        else:
            print(f"[!] Checkpoint not found at: {local_ckpt}")
            sys.exit(1)

    file_size_mb = local_ckpt.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print("        RESPIRA — UPLOAD CHECKPOINT TO HUGGING FACE HUB")
    print("=" * 70)
    print(f"Local Checkpoint  : {local_ckpt.resolve()} ({file_size_mb:.1f} MB)")
    print(f"Target Repository : {args.repo}")
    print(f"Destination Path  : methodology_fusion/best_model.pth")
    print()

    api = HfApi(token=args.token)

    try:
        print("[1/2] Verifying repository access...")
        api.repo_info(repo_id=args.repo, repo_type="model")
    except Exception as exc:
        print(f"[!] Warning: Could not find repository '{args.repo}' or no access.")
        print(f"    Creating model repository '{args.repo}' (if permitted)...")
        try:
            api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
            print(f"✓ Repository ready: https://huggingface.co/{args.repo}")
        except Exception as create_exc:
            print(f"[!] Failed to access or create repository: {create_exc}")
            print("    Please run `huggingface-cli login` or specify a repo you own: --repo <your-username>/<repo-name>")
            sys.exit(1)

    print("\n[2/2] Uploading model checkpoint (this may take 1-3 minutes depending on internet upload speed)...")
    try:
        res = upload_file(
            path_or_fileobj=str(local_ckpt),
            path_in_repo="methodology_fusion/best_model.pth",
            repo_id=args.repo,
            repo_type="model",
            token=args.token,
            commit_message="Add Finetuned Respira Fusion 512x512 best checkpoint (96.4% ROC-AUC)",
        )
        print("=" * 70)
        print("✓ UPLOAD COMPLETE!")
        print(f"Model URL: https://huggingface.co/{args.repo}/blob/main/methodology_fusion/best_model.pth")
        print("Anyone can now download it with: python scripts/download_models.py")
        print("=" * 70)
    except Exception as exc:
        print(f"[!] Upload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
