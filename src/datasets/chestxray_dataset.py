"""
============================================================
RESPIRA
Chest X-Ray Dataset Module
============================================================

STEP 2: DATASET LOADING

Purpose
-------
This module provides a reusable PyTorch Dataset for the
already-preprocessed Chest X-Ray dataset.

The preprocessing has already been completed in:

    data/processed/

Expected structure:

    data/
    └── processed/
        ├── train/
        │   ├── Atelectasis/
        │   ├── Bacterial Pneumonia/
        │   ├── Normal/
        │   ├── Pulmonary Edema/
        │   ├── Tuberculosis/
        │   └── Viral Pneumonia/
        │
        ├── val/
        │   ├── Atelectasis/
        │   ├── Bacterial Pneumonia/
        │   ├── Normal/
        │   ├── Pulmonary Edema/
        │   ├── Tuberculosis/
        │   └── Viral Pneumonia/
        │
        └── test/
            ├── Atelectasis/
            ├── Bacterial Pneumonia/
            ├── Normal/
            ├── Pulmonary Edema/
            ├── Tuberculosis/
            └── Viral Pneumonia/

Preprocessing already performed:
    1. Cleaning
    2. Balancing training data
    3. CLAHE
    4. Lung masking
    5. Resize to 224 x 224
    6. RGB conversion

This file DOES NOT perform preprocessing.

It only:
    - discovers images
    - validates classes
    - loads images
    - converts them to tensors
    - provides labels

============================================================
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# 1. PROJECT PATH
# ============================================================

# chestxray_dataset.py
#     ↓
# src/
#     ↓
# project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
    in enumerate(VALID_CLASSES)
}


INDEX_TO_CLASS = {
    index: class_name
    for class_name, index
    in CLASS_TO_INDEX.items()
}


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


# ============================================================
# 3. DATASET SPLITS
# ============================================================

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


# ============================================================
# 4. DATA SAMPLE TYPE
# ============================================================

Sample = Dict[str, object]


# ============================================================
# 5. DATASET CLASS
# ============================================================

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the processed Chest X-Ray dataset.

    Parameters
    ----------
    root_dir:
        Root directory containing train/val/test.

    split:
        One of:
            train
            val
            test

    transform:
        Optional torchvision transformation.

    verify_images:
        If True, images are opened once during dataset
        initialization to detect corrupted files.
    """

    def __init__(
        self,
        root_dir: str | Path = PROCESSED_DATA,
        split: str = "train",
        transform: Optional[Callable] = None,
        verify_images: bool = False,
    ):

        self.root_dir = Path(root_dir)

        self.split = split

        self.transform = transform

        self.verify_images = verify_images

        # ----------------------------------------------------
        # Validate split
        # ----------------------------------------------------

        if self.split not in VALID_SPLITS:

            raise ValueError(
                f"Invalid split: '{self.split}'.\n"
                f"Expected one of: {sorted(VALID_SPLITS)}"
            )

        # ----------------------------------------------------
        # Split directory
        # ----------------------------------------------------

        self.split_dir = (
            self.root_dir
            / self.split
        )

        if not self.split_dir.exists():

            raise FileNotFoundError(
                "\nDataset split does not exist:\n"
                f"{self.split_dir}\n\n"
                "Expected structure:\n"
                f"{self.root_dir}/train/\n"
                f"{self.root_dir}/val/\n"
                f"{self.root_dir}/test/"
            )

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        self.samples: List[Sample] = []

        self._collect_samples()

        # ----------------------------------------------------
        # Optional image verification
        # ----------------------------------------------------

        if self.verify_images:

            self._verify_images()

    # ========================================================
    # COLLECT SAMPLES
    # ========================================================

    def _collect_samples(self) -> None:

        """
        Discover all images and assign class labels.
        """

        for class_name in VALID_CLASSES:

            class_dir = (
                self.split_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    "\nRequired class directory missing:\n"
                    f"{class_dir}\n\n"
                    f"Expected class: {class_name}"
                )

            # ------------------------------------------------
            # Find images recursively
            # ------------------------------------------------

            image_paths = sorted(
                [
                    path
                    for path in class_dir.rglob("*")
                    if (
                        path.is_file()
                        and
                        path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                ]
            )

            # ------------------------------------------------
            # Empty class protection
            # ------------------------------------------------

            if len(image_paths) == 0:

                raise RuntimeError(
                    "\nClass directory contains no images:\n"
                    f"{class_dir}"
                )

            # ------------------------------------------------
            # Create samples
            # ------------------------------------------------

            label = CLASS_TO_INDEX[class_name]

            for image_path in image_paths:

                self.samples.append({

                    "path": image_path,

                    "class_name": class_name,

                    "label": label,

                    "split": self.split,
                })

        # ----------------------------------------------------
        # Dataset cannot be empty
        # ----------------------------------------------------

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.split_dir}"
            )

    # ========================================================
    # VERIFY IMAGES
    # ========================================================

    def _verify_images(self) -> None:

        """
        Open every image once to detect corrupted files.
        """

        corrupted = []

        print(
            f"\nVerifying {len(self.samples):,} images..."
        )

        for sample in self.samples:

            image_path = sample["path"]

            try:

                with Image.open(image_path) as image:

                    image.verify()

            except Exception as exc:

                corrupted.append(
                    (
                        image_path,
                        str(exc)
                    )
                )

        if corrupted:

            print(
                "\nCorrupted images detected:"
            )

            for path, error in corrupted[:20]:

                print(
                    f"  {path}"
                )

                print(
                    f"    Error: {error}"
                )

            raise RuntimeError(
                f"\nFound {len(corrupted)} "
                "corrupted image(s)."
            )

        print(
            "✓ All images verified successfully."
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self) -> int:

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, int]:

        sample = self.samples[index]

        image_path = sample["path"]

        label = sample["label"]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                "\nFailed to load image:\n"
                f"{image_path}\n"
                f"Error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Apply transform
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        return image, label

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    def class_distribution(self) -> Dict[str, int]:

        """
        Return number of images per class.
        """

        distribution = {
            class_name: 0
            for class_name in VALID_CLASSES
        }

        for sample in self.samples:

            class_name = sample["class_name"]

            distribution[class_name] += 1

        return distribution

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self) -> Dict[str, object]:

        """
        Return basic dataset information.
        """

        return {

            "split": self.split,

            "total_images": len(self),

            "num_classes": NUM_CLASSES,

            "classes": VALID_CLASSES.copy(),

            "class_distribution":
                self.class_distribution(),

            "root":
                str(self.split_dir),
        }


# ============================================================
# 6. DENSENET TRANSFORM
# ============================================================

def get_densenet_transform(
    image_size: int = IMAGE_SIZE,
) -> transforms.Compose:

    """
    Transform used for DenseNet121.

    The dataset is already resized to 224x224.

    We therefore do NOT resize again.

    Only:
        PIL RGB
            ↓
        Tensor
            ↓
        ImageNet normalization
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
# 7. EFFICIENTNET TRANSFORM
# ============================================================

def get_efficientnet_transform(
    image_size: int = IMAGE_SIZE,
) -> transforms.Compose:

    """
    EfficientNet-B0 preprocessing.

    Uses the same ImageNet normalization.

    Kept separate so that the architecture can later be
    changed without modifying the Dataset class.
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
# 8. ViT TRANSFORM
# ============================================================

def get_vit_transform(
    image_size: int = IMAGE_SIZE,
) -> transforms.Compose:

    """
    Vision Transformer preprocessing.

    The processed dataset already contains 224x224 images.
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
# 9. DATASET FACTORY
# ============================================================

def create_dataset(
    split: str,
    model: str = "densenet",
    root_dir: str | Path = PROCESSED_DATA,
    verify_images: bool = False,
) -> ChestXrayDataset:

    """
    Create a dataset using the correct model transform.

    Example
    -------
    dataset = create_dataset(
        split="train",
        model="densenet"
    )
    """

    model = model.lower().strip()

    if model == "densenet":

        transform = get_densenet_transform()

    elif model == "efficientnet":

        transform = get_efficientnet_transform()

    elif model == "vit":

        transform = get_vit_transform()

    else:

        raise ValueError(
            f"Unsupported model: {model}\n"
            "Supported models:\n"
            "  densenet\n"
            "  efficientnet\n"
            "  vit"
        )

    return ChestXrayDataset(

        root_dir=root_dir,

        split=split,

        transform=transform,

        verify_images=verify_images,
    )


# ============================================================
# 10. DEBUG / MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RESPIRA DATASET MODULE")
    print("=" * 70)

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

    print(
        "\nCreating training dataset..."
    )

    dataset = create_dataset(

        split="train",

        model="densenet",

        verify_images=False,
    )

    print(
        "\nDataset summary:"
    )

    summary = dataset.summary()

    print(
        f"  Split: {summary['split']}"
    )

    print(
        f"  Images: {summary['total_images']:,}"
    )

    print(
        f"  Classes: {summary['num_classes']}"
    )

    print(
        "\nClass distribution:"
    )

    for class_name, count in (
        summary["class_distribution"].items()
    ):

        print(
            f"  {class_name:<25}"
            f"{count:>8}"
        )

    # --------------------------------------------------------
    # Test one sample
    # --------------------------------------------------------

    image, label = dataset[0]

    print(
        "\nFirst sample:"
    )

    print(
        "  Tensor shape:",
        tuple(image.shape)
    )

    print(
        "  Tensor dtype:",
        image.dtype
    )

    print(
        "  Label:",
        label
    )

    print(
        "  Class:",
        INDEX_TO_CLASS[label]
    )

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    expected_shape = (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(image.shape) != expected_shape:

        raise RuntimeError(
            "\nUnexpected tensor shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual: {tuple(image.shape)}"
        )

    print(
        "\n✓ Tensor shape verified."
    )

    print(
        "\nDataset module working correctly."
    )

    print(
        "=" * 70
    )