"""
============================================================
RESPIRA - CHEST X-RAY DATASET LOADER
============================================================

Phase:
    DenseNet121 - Phase 1

Step:
    Step 2 - Dataset Loading & Validation

Purpose:
    Load the FINAL preprocessed dataset and verify that it is
    suitable for DenseNet121 feature extraction.

IMPORTANT:
    This script DOES NOT perform preprocessing.

Already completed:
    1. Image acquisition
    2. Cleaning
    3. Balancing
    4. CLAHE
    5. Lung masking
    6. 224 x 224 resizing

Expected dataset:

    data/
    └── processed/
        ├── train/
        ├── val/
        └── test/

Classes:

    Atelectasis
    Bacterial Pneumonia
    Normal
    Pulmonary Edema
    Tuberculosis
    Viral Pneumonia

Task:
    6-class chest X-ray classification

============================================================
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# This script is inside:
#
# Respira/
# └── scripts/
#     └── distinction.py
#
# Therefore parents[1] = Respira

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# Final preprocessed dataset

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

VALID_CLASSES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


NUM_CLASSES = len(VALID_CLASSES)

IMAGE_SIZE = 224

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Class -> integer index

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
    in enumerate(VALID_CLASSES)
}


# ============================================================
# 3. IMAGE TRANSFORMATION
# ============================================================

def get_transform():
    """
    Transform already-preprocessed 224x224 RGB images
    into tensors suitable for DenseNet121.

    No resizing is performed here.

    No CLAHE is performed here.

    No masking is performed here.
    """

    return transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# ============================================================
# 4. CHEST X-RAY DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split,
        transform=None,
    ):

        self.root_dir = Path(root_dir)

        self.split = split

        self.transform = (
            transform
            if transform is not None
            else get_transform()
        )

        # ----------------------------------------------------
        # Validate split
        # ----------------------------------------------------

        if split not in {
            "train",
            "val",
            "test",
        }:

            raise ValueError(
                f"Invalid split: {split}\n"
                f"Allowed: train, val, test"
            )

        # ----------------------------------------------------
        # Split directory
        # ----------------------------------------------------

        self.split_dir = (
            self.root_dir / split
        )

        if not self.split_dir.exists():

            raise FileNotFoundError(
                f"\nDataset split does not exist:\n"
                f"{self.split_dir}"
            )

        # ----------------------------------------------------
        # Collect samples
        # ----------------------------------------------------

        self.samples = []

        self._collect_samples()

        # ----------------------------------------------------
        # Empty dataset check
        # ----------------------------------------------------

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.split_dir}"
            )

    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    def _collect_samples(self):

        # ----------------------------------------------------
        # Check for unexpected directories
        # ----------------------------------------------------

        existing_directories = {
            directory.name
            for directory
            in self.split_dir.iterdir()
            if directory.is_dir()
        }

        unexpected = (
            existing_directories
            - set(VALID_CLASSES)
        )

        if unexpected:

            raise RuntimeError(
                "\nUnexpected class directories found:\n"
                f"{sorted(unexpected)}\n\n"
                "These should not be present in the "
                "final processed dataset."
            )

        # ----------------------------------------------------
        # Check all required classes
        # ----------------------------------------------------

        for class_name in VALID_CLASSES:

            class_dir = (
                self.split_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    "\nRequired class directory missing:\n"
                    f"{class_dir}"
                )

            # ------------------------------------------------
            # Find images recursively
            # ------------------------------------------------

            image_paths = sorted(
                [
                    path
                    for path
                    in class_dir.rglob("*")
                    if (
                        path.is_file()
                        and
                        path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                ]
            )

            # ------------------------------------------------
            # Empty class check
            # ------------------------------------------------

            if len(image_paths) == 0:

                raise RuntimeError(
                    f"\nClass contains no images:\n"
                    f"{class_dir}"
                )

            # ------------------------------------------------
            # Create sample entries
            # ------------------------------------------------

            for image_path in image_paths:

                self.samples.append({
                    "path": image_path,
                    "class_name": class_name,
                    "class_index":
                        CLASS_TO_INDEX[class_name],
                })

    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    @staticmethod
    def _load_image(image_path):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise ValueError(
                f"\nUnable to read image:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Verify image size
        # ----------------------------------------------------

        height, width = image.shape[:2]

        if (
            height != IMAGE_SIZE
            or width != IMAGE_SIZE
        ):

            raise ValueError(
                "\nInvalid image size!\n"
                f"Image : {image_path}\n"
                f"Expected: "
                f"{IMAGE_SIZE} x {IMAGE_SIZE}\n"
                f"Actual: "
                f"{width} x {height}"
            )

        # ----------------------------------------------------
        # Verify channels
        # ----------------------------------------------------

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):

            raise ValueError(
                "\nInvalid image channels!\n"
                f"Image: {image_path}\n"
                f"Shape: {image.shape}"
            )

        return image

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["path"]

        image = self._load_image(
            image_path
        )

        # ----------------------------------------------------
        # Apply tensor transformation
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # Verify tensor
        # ----------------------------------------------------

        if not isinstance(
            image,
            torch.Tensor,
        ):

            raise TypeError(
                "Transform did not return "
                "torch.Tensor."
            )

        if image.shape != (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            raise ValueError(
                "\nInvalid tensor shape:\n"
                f"{image.shape}"
            )

        # ----------------------------------------------------
        # Class label
        # ----------------------------------------------------

        label = torch.tensor(
            sample["class_index"],
            dtype=torch.long,
        )

        return image, label

    # ========================================================
    # SAMPLE INFORMATION
    # ========================================================

    def get_sample_info(self, index):

        sample = self.samples[index]

        return {
            "path": str(
                sample["path"]
            ),
            "class_name":
                sample["class_name"],
            "class_index":
                sample["class_index"],
        }

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self):

        counts = {
            class_name: 0
            for class_name
            in VALID_CLASSES
        }

        for sample in self.samples:

            counts[
                sample["class_name"]
            ] += 1

        return {
            "split": self.split,
            "total_images":
                len(self.samples),
            "class_distribution":
                counts,
        }


# ============================================================
# 5. DATASET VALIDATION
# ============================================================

def validate_dataset(dataset):

    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(
        "\nSplit:",
        dataset.split,
    )

    print(
        "Directory:",
        dataset.split_dir,
    )

    print(
        "Total images:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    summary = dataset.summary()

    print(
        "\nClass distribution:"
    )

    for class_name, count in (
        summary[
            "class_distribution"
        ].items()
    ):

        print(
            f"  {class_name:<22}"
            f"{count:>8}"
        )

    # --------------------------------------------------------
    # Random folder protection
    # --------------------------------------------------------

    random_found = any(

        sample["class_name"].lower()
        == "random"

        for sample
        in dataset.samples
    )

    if random_found:

        raise RuntimeError(
            "\nCRITICAL ERROR:\n"
            "'random' class found in "
            "final processed dataset."
        )

    # --------------------------------------------------------
    # Check first sample
    # --------------------------------------------------------

    image, label = dataset[0]

    print(
        "\nFirst sample:"
    )

    print(
        "  Image shape :",
        tuple(image.shape),
    )

    print(
        "  Image dtype :",
        image.dtype,
    )

    print(
        "  Label       :",
        label.item(),
    )

    print(
        "  Class       :",
        VALID_CLASSES[
            label.item()
        ],
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert image.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert label.item() in range(
        NUM_CLASSES
    )

    print(
        "\n✓ Image dimensions valid."
    )

    print(
        "✓ RGB channels valid."
    )

    print(
        "✓ Class label valid."
    )

    print(
        "✓ No random class detected."
    )

    print(
        "✓ Dataset validation successful."
    )


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RESPIRA - CHEST X-RAY DATASET LOADER")
    print("=" * 70)

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nProcessed dataset:"
    )

    print(
        PROCESSED_DATA
    )

    print(
        "\nClasses:"
    )

    for index, class_name in enumerate(
        VALID_CLASSES
    ):

        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------
    # Validate dataset root
    # --------------------------------------------------------

    if not PROCESSED_DATA.exists():

        raise FileNotFoundError(
            "\nProcessed dataset directory "
            "does not exist:\n"
            f"{PROCESSED_DATA}"
        )

    # --------------------------------------------------------
    # Test all splits
    # --------------------------------------------------------

    datasets = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        dataset = ChestXrayDataset(
            root_dir=PROCESSED_DATA,
            split=split,
        )

        datasets[split] = dataset

        validate_dataset(
            dataset
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    total = 0

    for split, dataset in (
        datasets.items()
    ):

        count = len(dataset)

        total += count

        print(
            f"{split.upper():<10}"
            f"{count:>10} images"
        )

    print(
        "-" * 70
    )

    print(
        f"{'TOTAL':<10}"
        f"{total:>10} images"
    )

    print()
    print(
        "✓ ALL DATASET SPLITS PASSED."
    )

    print(
        "✓ READY FOR DENSENET121."
    )

    print(
        "=" * 70
    )