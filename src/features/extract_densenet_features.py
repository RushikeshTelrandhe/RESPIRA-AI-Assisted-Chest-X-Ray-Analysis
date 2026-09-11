# ============================================================
# RESPIRA
# STEP 5 — DENSENET121 FEATURE EXTRACTION
# ============================================================
#
# Purpose:
#   Extract trained DenseNet121 CNN features for the
#   later CNN + ViT fusion architecture.
#
# Extracted representations:
#
#   1. Spatial feature map
#      Shape: (1024, 7, 7)
#
#   2. Global pooled feature vector
#      Shape: (1024,)
#
# These features will later be used by:
#
#   DenseNet CNN Branch
#          ↓
#   Spatial Feature Map
#          ↓
#   Bidirectional Cross-Attention
#          ↑
#   ViT Token Features
#
# ============================================================

from pathlib import Path
import sys
import json
import csv

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT DENSENET MODEL
# ============================================================

from src.models.densenet import DenseNet121Model


# ============================================================
# 3. CONFIGURATION
# ============================================================

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
    / "features"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_CLASSES = 6

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESPIRA — DENSENET121 FEATURE EXTRACTION")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. VALIDATE PATHS
# ============================================================

if not PROCESSED_DATA.exists():

    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n"
        f"{PROCESSED_DATA}"
    )

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nDenseNet checkpoint not found:\n"
        f"{CHECKPOINT}"
    )


# ============================================================
# 6. TRANSFORM
# ============================================================
#
# Your processed images are already:
#
#   224 × 224
#   3 channels
#
# Therefore we do NOT resize again unnecessarily.
#
# DenseNet pretrained on ImageNet expects normalized RGB
# input.
#
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),

])


# ============================================================
# 7. DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split,
        class_names,
        transform=None
    ):

        self.root_dir = Path(root_dir)

        self.split = split

        self.class_names = class_names

        self.transform = transform

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        self.samples = []

        split_dir = (
            self.root_dir
            / split
        )

        if not split_dir.exists():

            raise FileNotFoundError(
                f"Dataset split does not exist:\n"
                f"{split_dir}"
            )

        # ----------------------------------------------------
        # Collect samples
        # ----------------------------------------------------

        for class_name in class_names:

            class_dir = (
                split_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Required class directory missing:\n"
                    f"{class_dir}"
                )

            label = self.class_to_idx[class_name]

            for image_path in sorted(
                class_dir.rglob("*")
            ):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                ]:
                    continue

                self.samples.append(
                    (
                        image_path,
                        label,
                        class_name
                    )
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{split_dir}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, class_name = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        return (
            image,
            label,
            str(image_path),
            class_name
        )


# ============================================================
# 8. LOAD MODEL
# ============================================================

print("\nLoading DenseNet121...")

model = DenseNet121Model(
    num_classes=NUM_CLASSES,
    pretrained=False,
)


# ============================================================
# 9. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# Your checkpoint contains:
#
# model_state_dict
# optimizer_state_dict
# scheduler_state_dict
# etc.
#
# ------------------------------------------------------------

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()


# ============================================================
# 10. FREEZE MODEL
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


print("✓ DenseNet121 checkpoint loaded.")

print(
    "✓ DenseNet121 frozen for feature extraction."
)


# ============================================================
# 11. FEATURE EXTRACTION FUNCTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    images
):

    outputs = model(
        images
    )

    # --------------------------------------------------------
    # Your DenseNet model returns a dictionary.
    #
    # Expected keys normally include:
    #
    #   feature_map
    #   pooled_features
    #   logits
    #
    # --------------------------------------------------------

    if not isinstance(outputs, dict):

        raise RuntimeError(
            "DenseNet model output is not a dictionary."
        )

    # --------------------------------------------------------
    # Locate spatial feature map
    # --------------------------------------------------------

    feature_map = None

    possible_feature_keys = [
        "feature_map",
        "features",
        "cnn_features",
        "spatial_features",
    ]

    for key in possible_feature_keys:

        if key in outputs:

            feature_map = outputs[key]

            break

    if feature_map is None:

        raise KeyError(
            "Could not find DenseNet spatial "
            "feature map in model output.\n"
            f"Available keys: {list(outputs.keys())}"
        )

    # --------------------------------------------------------
    # Global pooling
    # --------------------------------------------------------

    pooled_features = torch.mean(
        feature_map,
        dim=(2, 3)
    )

    return (
        feature_map,
        pooled_features
    )


# ============================================================
# 12. EXTRACT EACH SPLIT
# ============================================================

all_metadata = []


for split in SPLITS:

    print("\n")
    print("=" * 70)
    print(f"EXTRACTING FEATURES — {split.upper()}")
    print("=" * 70)

    dataset = ChestXrayDataset(
        root_dir=PROCESSED_DATA,
        split=split,
        class_names=CLASS_NAMES,
        transform=transform,
    )

    print(
        f"\nImages: {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    split_output_dir = (
        OUTPUT_DIR
        / split
    )

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    spatial_features = []

    pooled_features = []

    labels = []

    image_paths = []

    class_labels = []

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    for (
        images,
        batch_labels,
        batch_paths,
        batch_class_names
    ) in tqdm(
        loader,
        desc=f"Extracting {split}",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        feature_map, pooled = (
            extract_features(
                model,
                images
            )
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        spatial_features.append(
            feature_map.cpu().half()
        )

        pooled_features.append(
            pooled.cpu().float()
        )

        labels.extend(
            batch_labels.numpy().tolist()
        )

        image_paths.extend(
            list(batch_paths)
        )

        class_labels.extend(
            list(batch_class_names)
        )

    # ========================================================
    # CONCATENATE
    # ========================================================

    spatial_features = torch.cat(
        spatial_features,
        dim=0
    )

    pooled_features = torch.cat(
        pooled_features,
        dim=0
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nFeature shapes:")

    print(
        "  Spatial:",
        tuple(spatial_features.shape)
    )

    print(
        "  Pooled :",
        tuple(pooled_features.shape)
    )

    print(
        "  Labels :",
        labels.shape
    )

    # Expected:
    #
    # spatial:
    #     N × 1024 × 7 × 7
    #
    # pooled:
    #     N × 1024
    #
    # --------------------------------------------------------

    if spatial_features.ndim != 4:

        raise RuntimeError(
            "Invalid spatial feature shape."
        )

    if spatial_features.shape[1:] != (
        1024,
        7,
        7
    ):

        raise RuntimeError(
            "Unexpected DenseNet feature map shape:\n"
            f"{tuple(spatial_features.shape)}"
        )

    if pooled_features.shape[1] != 1024:

        raise RuntimeError(
            "Unexpected pooled feature dimension:\n"
            f"{tuple(pooled_features.shape)}"
        )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    spatial_file = (
        split_output_dir
        / "spatial_features.pt"
    )

    pooled_file = (
        split_output_dir
        / "pooled_features.pt"
    )

    labels_file = (
        split_output_dir
        / "labels.npy"
    )

    metadata_file = (
        split_output_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # Spatial features
    #
    # float16 saves significant disk space.
    #
    # --------------------------------------------------------

    torch.save(
        spatial_features,
        spatial_file
    )

    # --------------------------------------------------------
    # Pooled features
    # --------------------------------------------------------

    torch.save(
        pooled_features,
        pooled_file
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    np.save(
        labels_file,
        labels
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_df = pd.DataFrame({

        "index": range(
            len(image_paths)
        ),

        "split": [
            split
        ] * len(image_paths),

        "image_path": image_paths,

        "label_index": labels,

        "label": class_labels,

    })

    metadata_df.to_csv(
        metadata_file,
        index=False
    )

    # ========================================================
    # RECORD METADATA
    # ========================================================

    split_info = {

        "split": split,

        "num_images": len(dataset),

        "spatial_feature_shape":
            list(
                spatial_features.shape
            ),

        "pooled_feature_shape":
            list(
                pooled_features.shape
            ),

        "spatial_dtype":
            str(
                spatial_features.dtype
            ),

        "pooled_dtype":
            str(
                pooled_features.dtype
            ),

        "spatial_features":
            str(spatial_file),

        "pooled_features":
            str(pooled_file),

        "labels":
            str(labels_file),

        "metadata":
            str(metadata_file),

    }

    all_metadata.append(
        split_info
    )

    print(
        "\n✓ Saved spatial features:"
    )

    print(
        " ",
        spatial_file
    )

    print(
        "\n✓ Saved pooled features:"
    )

    print(
        " ",
        pooled_file
    )

    print(
        "\n✓ Saved labels:"
    )

    print(
        " ",
        labels_file
    )

    print(
        "\n✓ Saved metadata:"
    )

    print(
        " ",
        metadata_file
    )


# ============================================================
# 13. SAVE GLOBAL EXTRACTION METADATA
# ============================================================

summary = {

    "model": "DenseNet121",

    "checkpoint": str(
        CHECKPOINT
    ),

    "image_size": IMAGE_SIZE,

    "num_classes": NUM_CLASSES,

    "classes": CLASS_NAMES,

    "device": str(DEVICE),

    "feature_channels": 1024,

    "feature_map_size": [
        7,
        7
    ],

    "spatial_feature_shape": [
        1024,
        7,
        7
    ],

    "pooled_feature_shape": [
        1024
    ],

    "splits": all_metadata,

}


summary_file = (
    OUTPUT_DIR
    / "feature_extraction_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("DENSENET121 FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    "\nFeatures saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSpatial feature map:"
)

print(
    "  (N, 1024, 7, 7)"
)

print(
    "\nGlobal feature vector:"
)

print(
    "  (N, 1024)"
)

print(
    "\n✓ Train features extracted."
)

print(
    "✓ Validation features extracted."
)

print(
    "✓ Test features extracted."
)

print(
    "✓ Labels saved."
)

print(
    "✓ Image metadata saved."
)

print(
    "✓ Extraction summary saved."
)

print("=" * 70)