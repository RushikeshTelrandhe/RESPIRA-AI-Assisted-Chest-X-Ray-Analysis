from pathlib import Path
from huggingface_hub import hf_hub_download


REPO_ID = "rushikeshtelrandhe/respira-models"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "efficientnet_b0_512/best_model.pth":
        PROJECT_ROOT / "outputs/efficientnet_b0_512/checkpoints/best_model.pth",

    "vit_512/best_model.pth":
        PROJECT_ROOT / "outputs/vit_512/checkpoints/best_model.pth",

    "efficientnet_b0_512/metrics.json":
        PROJECT_ROOT / "outputs/efficientnet_b0_512/evaluation/metrics.json",

    "vit_512/metrics.json":
        PROJECT_ROOT / "outputs/vit_512/evaluation/metrics.json",

    "fusion_512/fusion_config.json":
        PROJECT_ROOT / "outputs/fusion/fusion_config.json",
}


def main():
    print("=" * 70)
    print("RESPIRA — DOWNLOAD 512x512 MODELS")
    print("=" * 70)
    print(f"Hugging Face repository: {REPO_ID}")
    print()

    for repo_file, local_path in FILES.items():
        print(f"Downloading: {repo_file}")

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            local_dir=PROJECT_ROOT / ".hf_cache"
        )

        downloaded = Path(downloaded)

        local_path.write_bytes(
            downloaded.read_bytes()
        )

        print(f"✓ Saved: {local_path}")
        print()

    print("=" * 70)
    print("ALL 512x512 MODEL FILES DOWNLOADED")
    print("=" * 70)


if __name__ == "__main__":
    main()