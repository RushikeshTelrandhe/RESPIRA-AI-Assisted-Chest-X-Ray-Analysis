"""
RESPIRA - Master Dataset Ingestion Script
=========================================
Runs all three filtering and ingestion scripts in sequence:
  1. NIH Dataset       -> Atelectasis & Pulmonary Edema (and Normal)
  2. Pneumonia Dataset -> Bacterial Pneumonia, Viral Pneumonia, Normal
  3. TB Database       -> Tuberculosis (and Normal)

Usage:
------
1. Dry-run simulation of all three datasets:
   python scripts/ingest_all_raw_data.py --dry_run

2. Execute full ingestion:
   python scripts/ingest_all_raw_data.py
"""

import sys
import subprocess
import argparse


def run_step(description: str, cmd: list) -> bool:
    print("\n" + "=" * 75)
    print(f" STEP: {description}")
    print("=" * 75)
    print("Executing:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[!] Step failed with exit code: {res.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Master ingestion pipeline for all raw chest X-ray datasets.")
    parser.add_argument("--dry_run", action="store_true", help="Simulate all steps without copying files")
    parser.add_argument("--nih_max", type=int, default=1500, help="Max images per class from NIH (default: 1500)")
    parser.add_argument("--pneumonia_max", type=int, default=1000, help="Max images per class from Pneumonia dataset (default: 1000)")
    parser.add_argument("--tb_max", type=int, default=None, help="Max images per class from TB dataset (default: all)")
    args = parser.parse_args()

    py_exe = sys.executable
    dry_flag = ["--dry_run"] if args.dry_run else []

    # 1. NIH Ingestion
    nih_cmd = [
        py_exe, "scripts/filter_nih_dataset.py",
        "--classes", "Atelectasis", "Pulmonary Edema",
        "--max_per_class", str(args.nih_max)
    ] + dry_flag
    if not run_step("NIH ChestX-ray14 (Atelectasis & Pulmonary Edema)", nih_cmd):
        return

    # 2. Pneumonia Ingestion
    pneu_cmd = [
        py_exe, "scripts/filter_pneumonia_dataset.py",
        "--classes", "Bacterial Pneumonia", "Viral Pneumonia", "Normal",
        "--max_per_class", str(args.pneumonia_max)
    ] + dry_flag
    if not run_step("Kaggle Pneumonia Dataset (Bacterial, Viral, Normal)", pneu_cmd):
        return

    # 3. TB Ingestion
    tb_max_args = ["--max_per_class", str(args.tb_max)] if args.tb_max else []
    tb_cmd = [
        py_exe, "scripts/filter_tb_dataset.py",
        "--classes", "Tuberculosis"
    ] + tb_max_args + dry_flag
    if not run_step("Tuberculosis Database (Tuberculosis)", tb_cmd):
        return

    print("\n" + "=" * 75)
    print("  ALL INGESTION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print("Destination directories ready in data/raw_downloads/:")
    print("  - data/raw_downloads/nih/")
    print("  - data/raw_downloads/pneumonia/")
    print("  - data/raw_downloads/tb/")
    print("=" * 75)


if __name__ == "__main__":
    main()
