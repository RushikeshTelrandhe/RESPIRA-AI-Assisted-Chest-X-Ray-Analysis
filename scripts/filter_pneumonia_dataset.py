"""
RESPIRA - Pneumonia Dataset Filtering and Ingestion Script
=========================================================
This script parses the Kaggle Chest X-Ray Images (Pneumonia) dataset,
differentiates Bacterial Pneumonia vs Viral Pneumonia vs Normal based
on directory and filename patterns, and copies sampled images into
data/raw_downloads/pneumonia/ organized by class subfolders.

Category Mapping & Etiology:
----------------------------
- PNEUMONIA/*bacteria* or BACTERIA-* -> Bacterial Pneumonia
- PNEUMONIA/*virus* or VIRUS-*       -> Viral Pneumonia
- NORMAL/*                           -> Normal

Usage examples:
---------------
1. Dry run to inspect counts:
   python scripts/filter_pneumonia_dataset.py --dry_run

2. Sample up to 1,000 images per class (default):
   python scripts/filter_pneumonia_dataset.py --max_per_class 1000

3. Sample only Bacterial Pneumonia and Viral Pneumonia:
   python scripts/filter_pneumonia_dataset.py --classes "Bacterial Pneumonia" "Viral Pneumonia"

4. Automatically extract zip if archive is present:
   python scripts/filter_pneumonia_dataset.py --unzip
"""

import os
import sys
import csv
import glob
import random
import shutil
import zipfile
import argparse
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


TARGET_CLASSES = [
    "Bacterial Pneumonia",
    "Viral Pneumonia",
    "Normal"
]


def find_pneumonia_source(preferred_path: str) -> str:
    """Finds the unzipped chest_xray directory or returns empty string."""
    candidate_paths = [
        preferred_path,
        os.path.join(preferred_path, "chest_xray"),
        "data/raw_downloads/Full Pneumonia Data",
        "data/raw_downloads/Full Pneumonia Data/chest_xray",
        "data/raw_downloads/pneumonia/chest_xray",
        "data/raw_downloads/pneumonia",
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            # Check if train or NORMAL exists
            sub = [d.upper() for d in os.listdir(p)]
            if "TRAIN" in sub or "NORMAL" in sub or "PNEUMONIA" in sub:
                return p
            # Check nested chest_xray
            nested = os.path.join(p, "chest_xray")
            if os.path.exists(nested) and os.path.isdir(nested):
                return nested
    return ""


def find_zip_archive() -> str:
    """Searches for the downloaded pneumonia zip archive."""
    home_downloads = os.path.expanduser("~/Downloads")
    candidates = [
        "data/raw_downloads/Full Pneumonia Data/chest-xray-pneumonia.zip",
        "data/raw_downloads/chest-xray-pneumonia.zip",
        os.path.join(home_downloads, "archive (2).zip"),
        os.path.join(home_downloads, "chest-xray-pneumonia.zip"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c
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


def scan_pneumonia_images(source_dir: str, existing_images: Set[str]) -> Dict[str, List[dict]]:
    """
    Scans the pneumonia directory structure and categorizes images into:
      - Bacterial Pneumonia
      - Viral Pneumonia
      - Normal
    """
    print(f"[1/3] Scanning pneumonia images in: {source_dir}")
    categorized: Dict[str, List[dict]] = {
        "Bacterial Pneumonia": [],
        "Viral Pneumonia": [],
        "Normal": [],
    }

    scanned_total = 0
    duplicate_count = 0

    for root, _, files in os.walk(source_dir):
        # Ignore macOS metadata artifacts
        if "__MACOSX" in root:
            continue

        for fname in files:
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue

            scanned_total += 1
            if fname.lower() in existing_images:
                duplicate_count += 1
                continue

            full_path = os.path.join(root, fname)
            fname_lower = fname.lower()
            rel_path_upper = root.upper()

            assigned_class = None
            etiology = "Unknown"

            if "NORMAL" in rel_path_upper:
                assigned_class = "Normal"
                etiology = "Normal"
            elif "PNEUMONIA" in rel_path_upper or "bacteria" in fname_lower or "virus" in fname_lower:
                if "bacteria" in fname_lower:
                    assigned_class = "Bacterial Pneumonia"
                    etiology = "Bacteria"
                elif "virus" in fname_lower:
                    assigned_class = "Viral Pneumonia"
                    etiology = "Virus"

            if assigned_class in categorized:
                # Deduplicate by filename within the current batch
                categorized[assigned_class].append({
                    "image_name": fname,
                    "target_class": assigned_class,
                    "etiology": etiology,
                    "source_path": full_path,
                })

    # Deduplicate entries with identical filename in same class
    for cls in categorized:
        unique = {}
        for item in categorized[cls]:
            if item["image_name"] not in unique:
                unique[item["image_name"]] = item
        categorized[cls] = list(unique.values())

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
        description="Filter and sample Bacterial, Viral, and Normal images from Kaggle Pneumonia dataset."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/raw_downloads/Full Pneumonia Data",
        help="Path to unzipped chest_xray folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_downloads/pneumonia",
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
        default=["Bacterial Pneumonia", "Viral Pneumonia", "Normal"],
        help="Classes to sample (default: all 3)",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=1000,
        help="Target number of images to sample per class (default: 1000)",
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
        "--unzip",
        action="store_true",
        help="Automatically extract archive (2).zip if source directory is empty",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simulate selection without copying files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      RESPIRA - Pneumonia Dataset Ingestion Pipeline")
    print("=" * 70)
    print(f"Target Output Dir  : {args.output_dir}")
    print(f"Requested Classes  : {args.classes}")
    print(f"Max Per Class      : {args.max_per_class if args.max_per_class else 'All available'}")
    print(f"Random Seed        : {args.seed}")
    print(f"Dry Run            : {args.dry_run}")
    print("=" * 70)

    # 1. Locate source directory
    source_dir = find_pneumonia_source(args.source_dir)

    if not source_dir:
        # Check if zip exists and can be extracted
        zip_path = find_zip_archive()
        if zip_path:
            print(f"\n[!] Unzipped pneumonia folder not found, but archive detected at:")
            print(f"    {zip_path}")
            if args.unzip:
                extract_target = os.path.abspath("data/raw_downloads/Full Pneumonia Data")
                print(f"[*] Extracting archive to: {extract_target} ...")
                os.makedirs(extract_target, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_target)
                print("[*] Extraction complete.")
                source_dir = find_pneumonia_source(extract_target)
            else:
                print("\nTo automatically extract this archive and continue, re-run with:")
                print("  python scripts/filter_pneumonia_dataset.py --unzip")
                return
        else:
            print(f"\n[!] ERROR: No pneumonia dataset found in '{args.source_dir}' or Downloads folder.")
            print("Please extract chest_xray into 'data/raw_downloads/Full Pneumonia Data' or run with --unzip.")
            sys.exit(1)

    print(f"Source Directory   : {source_dir}")

    # 2. Check existing dataset for deduplication
    existing_images = get_existing_dataset_images(args.existing_dir)
    print(f"\n[Deduplication Check]")
    print(f"      Found {len(existing_images):,} existing images in '{args.existing_dir}' to avoid duplicating.")

    # 3. Scan and categorize
    categorized = scan_pneumonia_images(source_dir, existing_images)

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
        # Shuffle deterministically
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
                "etiology": rec["etiology"],
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
    manifest_csv = os.path.join(args.output_dir, "copied_pneumonia_manifest.csv")
    with open(manifest_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_name", "target_class", "etiology", "source_path", "destination_path"]
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
