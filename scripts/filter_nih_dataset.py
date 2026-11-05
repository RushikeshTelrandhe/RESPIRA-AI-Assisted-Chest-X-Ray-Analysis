"""
RESPIRA - NIH Chest X-Ray Filtering and Ingestion Script
=======================================================
This script parses Data_Entry_2017.csv from the NIH ChestX-ray14 dataset,
filters images based on the target disease categories, and copies them to the
target directory (data/raw_downloads/nih/) organized by class subfolders.

Mapping of NIH Findings to RESPIRA Target Classes:
--------------------------------------------------
- "Atelectasis"      -> Atelectasis
- "Edema"            -> Pulmonary Edema
- "No Finding"       -> Normal
- "Pneumonia"        -> Generic Pneumonia (NIH does not distinguish Bacterial vs Viral)
- "Tuberculosis"     -> Not present in NIH (sourced from dedicated TB database)

Usage examples:
---------------
1. Dry run to inspect how many images will be selected:
   python scripts/filter_nih_dataset.py --dry_run

2. Copy up to 1,500 images per class (pure single-label first, then multi-label if needed):
   python scripts/filter_nih_dataset.py --max_per_class 1500

3. Copy strictly pure single-label images only:
   python scripts/filter_nih_dataset.py --pure_only

4. Copy specific classes (e.g., Atelectasis and Pulmonary Edema only):
   python scripts/filter_nih_dataset.py --classes "Pulmonary Edema" Atelectasis --max_per_class 1200
"""

import os
import sys
import csv
import glob
import shutil
import argparse
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# Standard disease mappings from NIH ChestX-ray14
NIH_TO_TARGET_MAPPING = {
    "Atelectasis": "Atelectasis",
    "Edema": "Pulmonary Edema",
    "No Finding": "Normal",
}

TARGET_TO_NIH_MAPPING = {
    "Atelectasis": "Atelectasis",
    "Pulmonary Edema": "Edema",
    "Normal": "No Finding",
}

ALL_RESPIRA_CLASSES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia"
]


def index_nih_images(source_dir: str) -> Dict[str, str]:
    """
    Scans images_001 to images_012 subdirectories and builds a fast lookup dictionary.
    Returns:
        Dict mapping image filename (e.g. '00000001_000.png') -> full path on disk.
    """
    print(f"[1/4] Indexing NIH image files in: {source_dir}")
    index = {}
    
    # Check if images are inside images_xxx/images or directly inside source_dir
    subfolders = sorted(glob.glob(os.path.join(source_dir, "images_*")))
    
    if not subfolders:
        # Fallback: search source_dir directly
        subfolders = [source_dir]

    for folder in subfolders:
        img_dir = os.path.join(folder, "images")
        if not os.path.exists(img_dir):
            img_dir = folder
            
        if os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    index[fname] = os.path.join(img_dir, fname)

    print(f"      Total NIH images indexed: {len(index):,}")
    return index


def get_existing_dataset_images(existing_dir: str) -> Set[str]:
    """Collects set of image filenames already present in the preprocessed dataset."""
    existing = set()
    if os.path.exists(existing_dir):
        for root, _, files in os.walk(existing_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    existing.add(f)
    return existing


def parse_nih_csv(
    csv_path: str,
    target_classes: List[str],
    image_index: Dict[str, str],
    existing_images: Set[str],
    pure_only: bool = False,
    include_pneumonia: bool = False,
) -> Dict[str, Tuple[List[dict], List[dict]]]:
    """
    Parses Data_Entry_2017.csv and groups candidate images into:
      - pure single-label list
      - multi-label list
    for each target class.
    """
    print(f"[2/4] Parsing CSV metadata: {csv_path}")
    
    target_to_nih = TARGET_TO_NIH_MAPPING
    
    # Containers: target_class -> (pure_list, multi_list)
    candidates: Dict[str, Tuple[List[dict], List[dict]]] = {
        cls: ([], []) for cls in target_classes
    }
    if include_pneumonia and "Pneumonia" not in candidates:
        candidates["Pneumonia"] = ([], [])

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["Image Index"]
            
            # Skip if image file does not exist in the unzipped images
            if img_name not in image_index:
                continue

            # Skip if already part of existing preprocessed dataset
            if img_name in existing_images:
                continue

            findings_raw = row["Finding Labels"]
            labels = [lbl.strip() for lbl in findings_raw.split("|")]
            is_single = len(labels) == 1

            record = {
                "image_name": img_name,
                "finding_labels": findings_raw,
                "patient_id": row.get("Patient ID", ""),
                "view_position": row.get("View Position", ""),
                "source_path": image_index[img_name],
            }

            for cls in target_classes:
                nih_key = target_to_nih.get(cls)
                if not nih_key:
                    continue

                if is_single and labels[0] == nih_key:
                    candidates[cls][0].append(record)
                elif (not pure_only) and (not is_single) and (nih_key in labels):
                    candidates[cls][1].append(record)

            if include_pneumonia:
                if is_single and labels[0] == "Pneumonia":
                    candidates["Pneumonia"][0].append(record)
                elif (not pure_only) and (not is_single) and ("Pneumonia" in labels):
                    candidates["Pneumonia"][1].append(record)

    return candidates


def copy_single_file(task: Tuple[str, str]) -> Tuple[bool, str]:
    """Worker function to copy a single file."""
    src, dst = task
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True, dst
    except Exception as e:
        return False, f"Error copying {src} -> {dst}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Filter and copy NIH chest X-ray images matching RESPIRA categories."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/raw_downloads/Full NIH Data/Data_Entry_2017.csv",
        help="Path to Data_Entry_2017.csv",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/raw_downloads/Full NIH Data",
        help="Path containing unzipped images_001 to images_012 folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_downloads/nih",
        help="Destination directory for copied images",
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
        default=["Atelectasis", "Pulmonary Edema", "Normal"],
        help="Target classes to extract (e.g. 'Atelectasis' 'Pulmonary Edema' 'Normal')",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum images to copy per class (default: None = copy all candidates)",
    )
    parser.add_argument(
        "--pure_only",
        action="store_true",
        help="Only extract pure single-label images (zero co-occurring diseases)",
    )
    parser.add_argument(
        "--include_pneumonia",
        action="store_true",
        help="Also extract generic 'Pneumonia' images into a separate folder",
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
        help="Perform filtering and print summary without copying files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      RESPIRA - NIH Dataset Filtering & Ingestion Pipeline")
    print("=" * 70)
    print(f"Source Directory   : {args.source_dir}")
    print(f"CSV Metadata       : {args.csv_path}")
    print(f"Output Directory   : {args.output_dir}")
    print(f"Requested Classes  : {args.classes}")
    print(f"Max Per Class      : {args.max_per_class if args.max_per_class else 'All available'}")
    print(f"Purity Mode        : {'Pure single-label ONLY' if args.pure_only else 'Pure first, then multi-label'}")
    print(f"Dry Run            : {args.dry_run}")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(args.csv_path):
        print(f"[!] ERROR: CSV file not found at: {args.csv_path}")
        sys.exit(1)
    if not os.path.exists(args.source_dir):
        print(f"[!] ERROR: Source directory not found at: {args.source_dir}")
        sys.exit(1)

    # Explain unrepresented classes
    unsupported = [c for c in args.classes if c not in TARGET_TO_NIH_MAPPING]
    if unsupported:
        print("\n[NOTE on Class Availability]:")
        for c in unsupported:
            if "Pneumonia" in c:
                print(f"  * '{c}': NIH dataset only has generic 'Pneumonia', not Bacterial vs Viral.")
                print("    (Use --include_pneumonia if you want generic pneumonia from NIH, or use Kaggle pediatric pneumonia dataset)")
            elif "Tuberculosis" in c:
                print(f"  * '{c}': Tuberculosis is NOT part of the NIH 14 diseases.")
                print("    (Tuberculosis is sourced from dedicated TB Chest Radiography database)")

    # 1. Index images on disk
    image_index = index_nih_images(args.source_dir)

    # 2. Check existing dataset for deduplication
    print(f"\n[Deduplication Check]")
    existing_images = get_existing_dataset_images(args.existing_dir)
    print(f"      Found {len(existing_images):,} existing images in '{args.existing_dir}' to avoid duplicating.")

    # 3. Parse CSV and group candidates
    candidates = parse_nih_csv(
        csv_path=args.csv_path,
        target_classes=[c for c in args.classes if c in TARGET_TO_NIH_MAPPING],
        image_index=image_index,
        existing_images=existing_images,
        pure_only=args.pure_only,
        include_pneumonia=args.include_pneumonia,
    )

    # 4. Determine final selection
    print("\n[3/4] Candidate Summary & Selection Plan:")
    print("-" * 75)
    print(f"{'Class Name':<20} | {'Pure Single':<12} | {'Multi-label':<12} | {'Selected to Copy':<16}")
    print("-" * 75)

    copy_tasks = []
    manifest_rows = []

    for cls, (pure_list, multi_list) in candidates.items():
        total_avail = len(pure_list) + len(multi_list)
        
        # Selection logic: prioritize pure single-label images first
        if args.max_per_class:
            selected_pure = pure_list[:args.max_per_class]
            needed = args.max_per_class - len(selected_pure)
            selected_multi = multi_list[:needed] if needed > 0 else []
            selected = selected_pure + selected_multi
        else:
            selected = pure_list + multi_list

        print(f"{cls:<20} | {len(pure_list):<12} | {len(multi_list):<12} | {len(selected):<16}")

        for rec in selected:
            dst_path = os.path.join(args.output_dir, cls, rec["image_name"])
            copy_tasks.append((rec["source_path"], dst_path))
            manifest_rows.append({
                "image_name": rec["image_name"],
                "target_class": cls,
                "finding_labels": rec["finding_labels"],
                "patient_id": rec["patient_id"],
                "view_position": rec["view_position"],
                "source_path": rec["source_path"],
                "destination_path": dst_path,
            })

    print("-" * 75)
    print(f"{'TOTAL':<20} | {'-':<12} | {'-':<12} | {len(copy_tasks):<16}")
    print("-" * 75)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No files were copied. To execute copy, run without --dry_run.")
        return

    if not copy_tasks:
        print("\n[!] No matching images found to copy.")
        return

    # 5. Execute copy
    print(f"\n[4/4] Copying {len(copy_tasks):,} images using {args.workers} worker threads...")
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
    manifest_csv = os.path.join(args.output_dir, "copied_nih_manifest.csv")
    with open(manifest_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_name",
            "target_class",
            "finding_labels",
            "patient_id",
            "view_position",
            "source_path",
            "destination_path",
        ]
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
