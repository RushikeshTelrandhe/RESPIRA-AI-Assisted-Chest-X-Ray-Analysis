"""
RESPIRA - Tuberculosis (TB) Dataset Filtering and Ingestion Script
==================================================================
This script parses the Kaggle Tuberculosis (TB) Chest X-ray Database,
filters Tuberculosis (and optionally Normal) images, deduplicates against
existing dataset samples, and copies them to data/raw_downloads/tb/
organized by class subfolders.

Folder Structure Expected:
--------------------------
TB_Chest_Radiography_Database/
├── Tuberculosis/ (*.png)
└── Normal/       (*.png)

Usage examples:
---------------
1. Dry run to inspect counts:
   python scripts/filter_tb_dataset.py --dry_run

2. Copy all new Tuberculosis images and up to 1,000 Normal images:
   python scripts/filter_tb_dataset.py

3. Copy ONLY Tuberculosis images:
   python scripts/filter_tb_dataset.py --classes Tuberculosis

4. Limit maximum images per class:
   python scripts/filter_tb_dataset.py --max_per_class 500
"""

import os
import sys
import csv
import glob
import random
import shutil
import argparse
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


TARGET_CLASSES = ["Tuberculosis", "Normal"]


def find_tb_source(preferred_path: str) -> str:
    """Finds the unzipped TB_Chest_Radiography_Database directory."""
    candidate_paths = [
        preferred_path,
        os.path.join(preferred_path, "TB_Chest_Radiography_Database"),
        "data/raw_downloads/TB_Chest_Radiography_Database",
        "data/raw_downloads/tb/TB_Chest_Radiography_Database",
        "data/raw_downloads/tb",
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            sub = [d.lower() for d in os.listdir(p)]
            if "tuberculosis" in sub or "normal" in sub:
                return p
    return ""


def get_existing_dataset_images(existing_dir: str) -> Set[str]:
    """Collects set of image filenames already present in preprocessed dataset."""
    existing = set()
    if os.path.exists(existing_dir):
        for root, _, files in os.walk(existing_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    existing.add(f.lower())
    return existing


def scan_tb_images(source_dir: str, existing_images: Set[str]) -> Dict[str, List[dict]]:
    """
    Scans Tuberculosis and Normal subdirectories in the TB database.
    """
    print(f"[1/3] Scanning TB database in: {source_dir}")
    categorized: Dict[str, List[dict]] = {
        "Tuberculosis": [],
        "Normal": [],
    }

    scanned_total = 0
    duplicate_count = 0

    for cls in ["Tuberculosis", "Normal"]:
        cls_dir = os.path.join(source_dir, cls)
        # Case insensitive match for directory
        if not os.path.exists(cls_dir):
            for d in os.listdir(source_dir):
                if d.lower() == cls.lower():
                    cls_dir = os.path.join(source_dir, d)
                    break

        if not os.path.exists(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            scanned_total += 1
            if fname.lower() in existing_images:
                duplicate_count += 1
                continue

            categorized[cls].append({
                "image_name": fname,
                "target_class": cls,
                "source_path": os.path.join(cls_dir, fname),
            })

    print(f"      Total files scanned: {scanned_total:,}")
    print(f"      Duplicates skipped : {duplicate_count:,}")
    return categorized


def copy_single_file(task: Tuple[str, str]) -> Tuple[bool, str]:
    """Copies a single file to target destination."""
    src, dst = task
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True, dst
    except Exception as e:
        return False, f"Error copying {src} -> {dst}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Filter and sample Tuberculosis and Normal images from TB database."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/raw_downloads/TB_Chest_Radiography_Database",
        help="Path to TB_Chest_Radiography_Database folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_downloads/tb",
        help="Destination directory for categorized images",
    )
    parser.add_argument(
        "--existing_dir",
        type=str,
        default="data/Lung_Disease_Preprocessed_512",
        help="Directory with existing dataset to avoid duplicates",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Tuberculosis", "Normal"],
        help="Classes to copy (default: Tuberculosis Normal)",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Target maximum images to copy per class (default: None = copy all available new)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads for copying files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simulate selection without copying files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      RESPIRA - Tuberculosis (TB) Dataset Ingestion Pipeline")
    print("=" * 70)
    print(f"Target Output Dir  : {args.output_dir}")
    print(f"Requested Classes  : {args.classes}")
    print(f"Max Per Class      : {args.max_per_class if args.max_per_class else 'All available'}")
    print(f"Random Seed        : {args.seed}")
    print(f"Dry Run            : {args.dry_run}")
    print("=" * 70)

    # 1. Locate source directory
    source_dir = find_tb_source(args.source_dir)
    if not source_dir:
        print(f"\n[!] ERROR: TB database folder not found in '{args.source_dir}'.")
        print("Please place 'TB_Chest_Radiography_Database' into 'data/raw_downloads/'.")
        sys.exit(1)

    print(f"Source Directory   : {source_dir}")

    # 2. Check existing dataset for deduplication
    existing_images = get_existing_dataset_images(args.existing_dir)
    print(f"\n[Deduplication Check]")
    print(f"      Found {len(existing_images):,} existing images in '{args.existing_dir}' to avoid duplicating.")

    # 3. Scan and categorize
    categorized = scan_tb_images(source_dir, existing_images)

    # 4. Sample and prepare copy tasks
    random.seed(args.seed)

    print("\n[2/3] Available Candidates & Sampling Plan:")
    print("-" * 70)
    print(f"{'Class Name':<25} | {'Available (New)':<18} | {'Selected to Copy':<18}")
    print("-" * 70)

    copy_tasks = []
    manifest_rows = []

    for cls in args.classes:
        candidates = categorized.get(cls, [])
        random.shuffle(candidates)

        if args.max_per_class and len(candidates) > args.max_per_class:
            selected = candidates[:args.max_per_class]
        else:
            selected = candidates

        print(f"{cls:<25} | {len(candidates):<18} | {len(selected):<18}")

        for rec in selected:
            dst_path = os.path.join(args.output_dir, cls, rec["image_name"])
            copy_tasks.append((rec["source_path"], dst_path))
            manifest_rows.append({
                "image_name": rec["image_name"],
                "target_class": cls,
                "source_path": rec["source_path"],
                "destination_path": dst_path,
            })

    print("-" * 70)
    print(f"{'TOTAL':<25} | {'-':<18} | {len(copy_tasks):<18}")
    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No files were copied. To copy, run without --dry_run.")
        return

    if not copy_tasks:
        print("\n[!] No matching images found to copy.")
        return

    # 5. Execute copy
    print(f"\n[3/3] Copying {len(copy_tasks):,} images using {args.workers} worker threads...")
    os.makedirs(args.output_dir, exist_ok=True)

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(copy_single_file, task): task for task in copy_tasks}
        for future in as_completed(futures):
            success, msg = future.result()
            if success:
                completed += 1
            else:
                errors += 1
                print(f"[!] {msg}")

            if completed % 500 == 0 or completed == len(copy_tasks):
                progress = (completed / len(copy_tasks)) * 100
                print(f"      Progress: {completed:,} / {len(copy_tasks):,} ({progress:.1f}%)")

    # 6. Save Manifest CSV
    manifest_csv = os.path.join(args.output_dir, "copied_tb_manifest.csv")
    with open(manifest_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_name", "target_class", "source_path", "destination_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print("                      PROCESS COMPLETE")
    print("=" * 70)
    print(f"Successfully copied : {completed:,} images")
    if errors > 0:
        print(f"Errors encountered  : {errors}")
    print(f"Destination folder  : {os.path.abspath(args.output_dir)}")
    print(f"Manifest saved to   : {os.path.abspath(manifest_csv)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
