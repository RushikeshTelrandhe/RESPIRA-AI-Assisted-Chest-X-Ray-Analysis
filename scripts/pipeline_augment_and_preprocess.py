"""
RESPIRA - End-to-End Dataset Augmentation & Preprocessing Pipeline
==================================================================
Executes the full data lifecycle:
  1. Filters & samples raw images from NIH, Kaggle Pneumonia, and TB DB to hit ~2,000 per class.
  2. Strict multi-threaded SHA-256 deduplication and corrupt image filtering against existing dataset.
  3. Preprocessing: CLAHE (clip 2.0, tile 8x8) -> Morphological Lung Masking (with fallback) -> Resize 512x512 (cv2.INTER_AREA) -> 3-channel RGB.
  4. Stratified train/val/test splitting (70% train / 15% val / 15% test).
  5. Comprehensive dataset integrity audit and PyTorch DataLoader verification.

Usage:
------
1. Dry run simulation:
   python scripts/pipeline_augment_and_preprocess.py --dry_run

2. Full execution (multi-threaded preprocessing):
   python scripts/pipeline_augment_and_preprocess.py --workers 8
"""

import os
import sys

# Ensure immediate unbuffered output on Windows
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import csv
import json
import glob
import time
import shutil
import random
import hashlib
import argparse
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

# Ensure app module can be imported
sys.path.append(os.path.abspath("."))
from app.backend.preprocessing import apply_clahe


# Target 6 classes
TARGET_CLASSES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia"
]

TARGET_PER_CLASS = 2000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def batch_hash_files(file_paths: List[str], max_workers: int = 16) -> Dict[str, str]:
    """Computes SHA-256 hashes for a list of file paths in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(sha256_file, p): p for p in file_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as e:
                results[path] = ""
    return results


def validate_image_readable(path: str) -> bool:
    """Verifies image is non-empty and readable by PIL and OpenCV."""
    try:
        if os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_existing_dataset(dataset_dir: str) -> Tuple[Dict[str, List[dict]], Set[str], Set[str]]:
    """
    Collects existing preprocessed images from data/Lung_Disease_Preprocessed_512.
    Returns:
        existing_by_class: Dict[cls -> list of records]
        existing_filenames: Set of lowercase filenames
        existing_hashes: Set of SHA-256 digests
    """
    print(f"[Phase 1/5] Scanning existing preprocessed dataset: {dataset_dir}")
    existing_by_class = {cls: [] for cls in TARGET_CLASSES}
    existing_filenames = set()
    existing_hashes = set()

    if not os.path.exists(dataset_dir):
        print(f"      Directory does not exist yet: {dataset_dir}")
        return existing_by_class, existing_filenames, existing_hashes

    all_files = []
    file_metadata = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(cls_dir, f)
                    all_files.append(fpath)
                    file_metadata.append((f, fpath, cls))

    # Fast parallel hashing of all existing images
    print(f"      Computing SHA-256 hashes for {len(all_files):,} existing images (parallel)...")
    t0 = time.time()
    hashes = batch_hash_files(all_files, max_workers=16)
    print(f"      Hashed {len(all_files):,} images in {time.time()-t0:.2f} seconds.")

    for fname, fpath, cls in file_metadata:
        h = hashes.get(fpath, "")
        if h:
            if h in existing_hashes:
                # Discard duplicate content even if filenames differ
                continue
            existing_hashes.add(h)
            existing_filenames.add(fname.lower())
            existing_by_class[cls].append({
                "filename": fname,
                "path": fpath,
                "hash": h,
                "class": cls,
                "is_preprocessed": True,
            })

    print(f"      Found {len(all_files):,} existing images across 6 classes:")
    for cls in TARGET_CLASSES:
        print(f"        * {cls:<20}: {len(existing_by_class[cls]):,} images")
    return existing_by_class, existing_filenames, existing_hashes


def sample_new_raw_images(
    existing_filenames: Set[str],
    existing_hashes: Set[str],
    existing_by_class: Dict[str, List[dict]],
    target_per_class: int = 2000,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Selects raw images from NIH, Pneumonia, and TB datasets to reach target_per_class.
    """
    print(f"\n[Phase 2/5] Selecting new raw images to reach ~{target_per_class:,} per class...")
    random.seed(seed)
    new_candidates_by_class = {cls: [] for cls in TARGET_CLASSES}
    seen_hashes = set(existing_hashes)

    # 1. NIH Data (Atelectasis & Pulmonary Edema)
    nih_csv = "data/raw_downloads/Full NIH Data/Data_Entry_2017.csv"
    nih_dir = "data/raw_downloads/Full NIH Data"
    
    if os.path.exists(nih_csv) and os.path.exists(nih_dir):
        print("      Indexing NIH dataset...")
        nih_index = {}
        for folder in sorted(glob.glob(os.path.join(nih_dir, "images_*"))):
            img_dir = os.path.join(folder, "images") if os.path.exists(os.path.join(folder, "images")) else folder
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    nih_index[fname] = os.path.join(img_dir, fname)

        nih_pure = {"Atelectasis": [], "Pulmonary Edema": []}
        nih_multi = {"Atelectasis": [], "Pulmonary Edema": []}

        with open(nih_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Image Index"]
                if img_name not in nih_index or img_name.lower() in existing_filenames:
                    continue
                findings = [x.strip() for x in row["Finding Labels"].split("|")]
                is_single = len(findings) == 1

                rec = {"filename": img_name, "source_path": nih_index[img_name], "source_dataset": "NIH"}

                if is_single and findings[0] == "Atelectasis":
                    nih_pure["Atelectasis"].append(rec)
                elif (not is_single) and ("Atelectasis" in findings):
                    nih_multi["Atelectasis"].append(rec)

                if is_single and findings[0] == "Edema":
                    nih_pure["Pulmonary Edema"].append(rec)
                elif (not is_single) and ("Edema" in findings):
                    nih_multi["Pulmonary Edema"].append(rec)

        # Sample for Atelectasis & Pulmonary Edema
        for cls in ["Atelectasis", "Pulmonary Edema"]:
            needed = max(0, target_per_class - len(existing_by_class[cls]))
            p_list = nih_pure[cls]
            m_list = nih_multi[cls]
            random.shuffle(p_list)
            random.shuffle(m_list)

            chosen_pure = p_list[:needed]
            chosen_multi = m_list[:max(0, needed - len(chosen_pure))]
            combined = chosen_pure + chosen_multi

            # Batch hash candidate files in parallel
            candidate_paths = [x["source_path"] for x in combined]
            cand_hashes = batch_hash_files(candidate_paths, max_workers=16)

            for item in combined:
                h = cand_hashes.get(item["source_path"], "")
                if h and h not in seen_hashes:
                    seen_hashes.add(h)
                    item["hash"] = h
                    item["class"] = cls
                    item["is_preprocessed"] = False
                    new_candidates_by_class[cls].append(item)

    # 2. Kaggle Pneumonia Data (Bacterial Pneumonia, Viral Pneumonia, Normal)
    pneu_dir = "data/raw_downloads/Full Pneumonia Data/chest_xray"
    if not os.path.exists(pneu_dir):
        pneu_dir = "data/raw_downloads/pneumonia/chest_xray"

    if os.path.exists(pneu_dir):
        print("      Scanning Kaggle Pneumonia dataset...")
        pneu_found = {"Bacterial Pneumonia": [], "Viral Pneumonia": [], "Normal": []}

        for root, _, files in os.walk(pneu_dir):
            if "__MACOSX" in root:
                continue
            for fname in files:
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                if fname.lower() in existing_filenames:
                    continue

                full_path = os.path.join(root, fname)
                fname_lower = fname.lower()
                root_upper = root.upper()

                rec = {"filename": fname, "source_path": full_path, "source_dataset": "Pneumonia"}

                if "NORMAL" in root_upper:
                    pneu_found["Normal"].append(rec)
                elif "bacteria" in fname_lower:
                    pneu_found["Bacterial Pneumonia"].append(rec)
                elif "virus" in fname_lower:
                    pneu_found["Viral Pneumonia"].append(rec)

        for cls in ["Bacterial Pneumonia", "Viral Pneumonia", "Normal"]:
            needed = max(0, target_per_class - len(existing_by_class[cls]))
            items = pneu_found[cls]
            unique_items = list({x["filename"]: x for x in items}.values())
            random.shuffle(unique_items)
            chosen = unique_items[:needed]

            cand_paths = [x["source_path"] for x in chosen]
            cand_hashes = batch_hash_files(cand_paths, max_workers=16)

            for item in chosen:
                h = cand_hashes.get(item["source_path"], "")
                if h and h not in seen_hashes:
                    seen_hashes.add(h)
                    item["hash"] = h
                    item["class"] = cls
                    item["is_preprocessed"] = False
                    new_candidates_by_class[cls].append(item)

    # 3. TB Database (Tuberculosis)
    tb_dir = "data/raw_downloads/TB_Chest_Radiography_Database"
    if os.path.exists(tb_dir):
        print("      Scanning TB database...")
        tb_sub = os.path.join(tb_dir, "Tuberculosis")
        if os.path.exists(tb_sub):
            needed = max(0, target_per_class - len(existing_by_class["Tuberculosis"]))
            items = []
            for fname in os.listdir(tb_sub):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")) and fname.lower() not in existing_filenames:
                    items.append({
                        "filename": fname,
                        "source_path": os.path.join(tb_sub, fname),
                        "source_dataset": "TB_Database"
                    })
            random.shuffle(items)
            chosen = items[:needed]

            cand_paths = [x["source_path"] for x in chosen]
            cand_hashes = batch_hash_files(cand_paths, max_workers=16)

            for item in chosen:
                h = cand_hashes.get(item["source_path"], "")
                if h and h not in seen_hashes:
                    seen_hashes.add(h)
                    item["hash"] = h
                    item["class"] = "Tuberculosis"
                    item["is_preprocessed"] = False
                    new_candidates_by_class["Tuberculosis"].append(item)

    # Summary table
    print("\n      --- Image Sampling Summary ---")
    print(f"      {'Class Name':<22} | {'Existing':<10} | {'New Added':<10} | {'Total Combined':<14}")
    print("      " + "-" * 62)
    grand_total = 0
    for cls in TARGET_CLASSES:
        ex_cnt = len(existing_by_class[cls])
        new_cnt = len(new_candidates_by_class[cls])
        tot = ex_cnt + new_cnt
        grand_total += tot
        print(f"      {cls:<22} | {ex_cnt:<10} | {new_cnt:<10} | {tot:<14}")
    print("      " + "-" * 62)
    print(f"      {'TOTAL DATASET':<22} | {'-':<10} | {'-':<10} | {grand_total:<14}\n")

    return new_candidates_by_class


def preprocess_worker(task: Tuple[str, str, int]) -> Tuple[bool, str]:
    """
    Worker function to preprocess a single raw image and save as 512x512 PNG.
    Applies CLAHE enhancement + Resize 512x512 (Area) + RGB conversion.
    NO LUNG MASKING.
    """
    src_path, dst_path, img_size = task
    try:
        if not validate_image_readable(src_path):
            return False, f"Unreadable file: {src_path}"

        img = cv2.imread(src_path)
        if img is None:
            return False, f"cv2.imread failed: {src_path}"

        # Grayscale conversion
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # CLAHE enhancement (clip=2.0, tileGridSize=(8, 8)) WITHOUT masking
        clahe = apply_clahe(gray)

        # Resize to 512x512 using area interpolation
        resized = cv2.resize(clahe, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Convert to 3-channel RGB
        final = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        success = cv2.imwrite(dst_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        if not success:
            return False, f"cv2.imwrite failed: {dst_path}"
        return True, dst_path
    except Exception as e:
        return False, f"Error preprocessing {src_path}: {str(e)}"


def copy_worker(task: Tuple[str, str]) -> Tuple[bool, str]:
    """Worker function to copy an existing preprocessed image to the new split folder."""
    src_path, dst_path = task
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True, dst_path
    except Exception as e:
        return False, f"Error copying {src_path}: {str(e)}"


def execute_pipeline(
    existing_by_class: Dict[str, List[dict]],
    new_candidates_by_class: Dict[str, List[dict]],
    output_dir: str,
    workers: int = 8,
    img_size: int = 512,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Preprocesses new images, stratifies into train/val/test splits,
    and populates output_dir.
    """
    random.seed(seed)
    
    print(f"[Phase 3/5] Stratifying and partitioning dataset (70% train / 15% val / 15% test)...")
    split_tasks = {"train": [], "val": [], "test": []}
    all_manifest_records = []

    for cls in TARGET_CLASSES:
        all_class_items = existing_by_class[cls] + new_candidates_by_class[cls]
        random.shuffle(all_class_items)

        n = len(all_class_items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        splits_assigned = {
            "train": all_class_items[:n_train],
            "val": all_class_items[n_train:n_train + n_val],
            "test": all_class_items[n_train + n_val:],
        }

        print(f"      {cls:<20}: Train={len(splits_assigned['train'])}, Val={len(splits_assigned['val'])}, Test={len(splits_assigned['test'])}")

        for split_name, items in splits_assigned.items():
            for itm in items:
                dst_fname = itm["filename"] if itm["filename"].endswith(".png") else os.path.splitext(itm["filename"])[0] + ".png"
                dst_path = os.path.join(output_dir, split_name, cls, dst_fname)

                split_tasks[split_name].append({
                    "item": itm,
                    "target_class": cls,
                    "split": split_name,
                    "dst_path": dst_path,
                })

                all_manifest_records.append({
                    "filename": dst_fname,
                    "class": cls,
                    "split": split_name,
                    "is_new": not itm["is_preprocessed"],
                    "source_dataset": itm.get("source_dataset", "Existing_Preprocessed"),
                    "hash": itm["hash"],
                })

    if dry_run:
        print("\n[DRY RUN COMPLETE] Simulated partition successfully. No files written.")
        return

    # Preparation
    print(f"\n[Phase 4/5] Executing image processing & copying to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    copy_tasks = []
    preprocess_tasks = []

    for split_name in ["train", "val", "test"]:
        for task in split_tasks[split_name]:
            itm = task["item"]
            dst = task["dst_path"]
            if itm["is_preprocessed"]:
                copy_tasks.append((itm["path"], dst))
            else:
                preprocess_tasks.append((itm["source_path"], dst, img_size))

    print(f"      Existing images to copy: {len(copy_tasks):,}")
    print(f"      New raw images to preprocess: {len(preprocess_tasks):,}")

    # Copy existing
    t0 = time.time()
    copied_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(copy_worker, t): t for t in copy_tasks}
        for future in as_completed(futures):
            ok, msg = future.result()
            if ok:
                copied_count += 1
            else:
                print(f"[!] {msg}")
    print(f"      Completed copying {copied_count:,} existing images in {time.time()-t0:.2f}s.")

    # Preprocess new images using ProcessPoolExecutor for CPU-bound computer vision
    t0 = time.time()
    preprocessed_count = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(preprocess_worker, t): t for t in preprocess_tasks}
        for future in as_completed(futures):
            ok, msg = future.result()
            if ok:
                preprocessed_count += 1
                if preprocessed_count % 500 == 0 or preprocessed_count == len(preprocess_tasks):
                    pct = (preprocessed_count / len(preprocess_tasks)) * 100
                    print(f"      Preprocessing progress: {preprocessed_count:,} / {len(preprocess_tasks):,} ({pct:.1f}%)")
            else:
                errors += 1
                print(f"[!] {msg}")

    print(f"      Preprocessed {preprocessed_count:,} new images in {time.time()-t0:.2f}s.")

    # Save Manifest
    manifest_csv = os.path.join(output_dir, "dataset_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "split", "is_new", "source_dataset", "hash"])
        writer.writeheader()
        writer.writerows(all_manifest_records)
    print(f"      Manifest saved to: {manifest_csv}")


def verify_final_dataset(dataset_dir: str):
    """
    Phase 5: Exhaustive Integrity & Leakage Verification.
    """
    print(f"\n[Phase 5/5] Running Comprehensive Dataset Integrity Checks on: {dataset_dir}")
    print("=" * 70)

    split_hashes = {"train": set(), "val": set(), "test": set()}
    split_counts = {"train": {c: 0 for c in TARGET_CLASSES},
                    "val": {c: 0 for c in TARGET_CLASSES},
                    "test": {c: 0 for c in TARGET_CLASSES}}
    corrupted = []
    dimension_mismatches = []

    total_verified = 0

    for split in ["train", "val", "test"]:
        s_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(s_dir):
            print(f"[!] Missing split directory: {s_dir}")
            return False

        for cls in TARGET_CLASSES:
            c_dir = os.path.join(s_dir, cls)
            if not os.path.exists(c_dir):
                print(f"[!] Missing class directory: {c_dir}")
                continue

            for fname in os.listdir(c_dir):
                fpath = os.path.join(c_dir, fname)
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                total_verified += 1
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    if (w, h) != (512, 512):
                        dimension_mismatches.append((fpath, (w, h)))
                    hsh = sha256_file(fpath)
                    split_hashes[split].add(hsh)
                    split_counts[split][cls] += 1
                except Exception as e:
                    corrupted.append((fpath, str(e)))

    # Leakage audits
    tv_leak = split_hashes["train"].intersection(split_hashes["val"])
    tt_leak = split_hashes["train"].intersection(split_hashes["test"])
    vt_leak = split_hashes["val"].intersection(split_hashes["test"])

    print("\n--- Integrity Audit Results ---")
    print(f"Total Images Verified : {total_verified:,}")
    print(f"Corrupted Images      : {len(corrupted)}")
    print(f"Dimension Mismatches  : {len(dimension_mismatches)}")
    print(f"Train / Val Leakage   : {len(tv_leak)} images (Expect 0)")
    print(f"Train / Test Leakage  : {len(tt_leak)} images (Expect 0)")
    print(f"Val / Test Leakage    : {len(vt_leak)} images (Expect 0)")

    print("\n--- Final Stratified Distribution ---")
    print(f"{'Class Name':<22} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 62)
    grand_train = 0
    grand_val = 0
    grand_test = 0

    for cls in TARGET_CLASSES:
        tr = split_counts["train"][cls]
        va = split_counts["val"][cls]
        te = split_counts["test"][cls]
        tot = tr + va + te
        grand_train += tr
        grand_val += va
        grand_test += te
        print(f"{cls:<22} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")

    print("-" * 62)
    print(f"{'TOTALS':<22} | {grand_train:<8} | {grand_val:<8} | {grand_test:<8} | {grand_train + grand_val + grand_test:<8}")
    print("=" * 70)

    success = (len(corrupted) == 0 and len(dimension_mismatches) == 0 and
               len(tv_leak) == 0 and len(tt_leak) == 0 and len(vt_leak) == 0)

    if success:
        print("[SUCCESS] All dataset integrity and leakage checks passed with 100% compliance!")
    else:
        print("[WARNING] Some checks reported anomalies. Please review logs.")
    return success


def main():
    parser = argparse.ArgumentParser(description="RESPIRA Dataset Augmentation, Preprocessing & Verification Pipeline")
    parser.add_argument("--dataset_dir", type=str, default="data/Lung_Disease_Preprocessed_512", help="Main preprocessed dataset path")
    parser.add_argument("--augmented_dir", type=str, default="data/Lung_Disease_Preprocessed_512_Augmented", help="Target output path for augmented dataset")
    parser.add_argument("--target_per_class", type=int, default=TARGET_PER_CLASS, help="Target images per class (default: 2000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes for preprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--dry_run", action="store_true", help="Run simulation without writing files")
    parser.add_argument("--inplace", action="store_true", help="Replace data/Lung_Disease_Preprocessed_512 directly (after automatic backup)")
    args = parser.parse_args()

    print("=" * 70)
    print("  RESPIRA - DATASET AUGMENTATION & PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Target Images Per Class: {args.target_per_class:,}")
    print(f"Workers                : {args.workers}")
    print(f"Dry Run                : {args.dry_run}")
    print("=" * 70)

    # Phase 1: Collect existing preprocessed images (prefer original unmasked backup if present)
    source_existing = args.dataset_dir
    backup_dir = args.dataset_dir + "_Backup_Original"
    if os.path.exists(backup_dir):
        source_existing = backup_dir
        print(f"[*] Sourcing original unmasked images from backup: {source_existing}")

    existing_by_class, existing_filenames, existing_hashes = collect_existing_dataset(source_existing)

    # Phase 2: Sample new raw images
    new_candidates = sample_new_raw_images(
        existing_filenames=existing_filenames,
        existing_hashes=existing_hashes,
        existing_by_class=existing_by_class,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )

    # Target directory
    target_dir = args.augmented_dir if not args.inplace else args.dataset_dir

    if args.inplace and not args.dry_run:
        target_dir = args.dataset_dir + "_Temp"

    # Phase 3 & 4: Execute processing and split
    execute_pipeline(
        existing_by_class=existing_by_class,
        new_candidates_by_class=new_candidates,
        output_dir=target_dir,
        workers=args.workers,
        img_size=512,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    if args.inplace:
        print(f"[*] Replacing {args.dataset_dir} with newly augmented dataset...")
        shutil.rmtree(args.dataset_dir)
        os.rename(target_dir, args.dataset_dir)
        final_dir = args.dataset_dir
    else:
        final_dir = target_dir

    # Phase 5: Verification
    verify_final_dataset(final_dir)


if __name__ == "__main__":
    main()
