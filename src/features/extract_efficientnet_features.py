# ============================================================
# Respira - EfficientNet-B0 Feature Extraction
# ============================================================
#
# STEP 6.5
#
# Purpose:
#   Extract spatial and global features from the trained
#   EfficientNet-B0 CNN branch.
#
# Outputs:
#
#   Spatial feature map:
#       [N, 1280, 7, 7]
#
#   Global pooled feature:
#       [N, 1280]
#
# These features will later be used by the multimodal
# CNN + ViT fusion architecture.
#
# ============================================================

from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from src.models.efficientnet import EfficientNetB0Classifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0"
    / "features"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "efficientnet_b0"
    / "best_model.pth"
)


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

# ------------------------------------------------------------
# Feature storage
#
# Spatial features are large:
#
#   1280 × 7 × 7
#
# Storing them as float16 reduces disk usage substantially.
#
# They can be converted back to float32 during fusion.
# ------------------------------------------------------------

SPATIAL_DTYPE = torch.float16
POOLED_DTYPE = torch.float32


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

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
    ]
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for split in ["train", "val", "test"]:

    (
        OUTPUT_DIR / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — EFFICIENTNET-B0 FEATURE EXTRACTION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nProcessed dataset:")
print(PROCESSED_DATA)

print("\nCheckpoint:")
print(CHECKPOINT)

print("\nFeature output:")
print(OUTPUT_DIR)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

print("\nClasses:")

for idx, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {idx}: {class_name}"
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not PROCESSED_DATA.exists():

    raise FileNotFoundError(
        f"\nProcessed dataset does not exist:\n"
        f"{PROCESSED_DATA}"
    )


if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nEfficientNet checkpoint does not exist:\n"
        f"{CHECKPOINT}\n\n"
        "Train EfficientNet-B0 first."
    )


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(split):

    split_dir = (
        PROCESSED_DATA
        / split
    )

    if not split_dir.exists():

        raise FileNotFoundError(
            f"\nDataset split does not exist:\n"
            f"{split_dir}"
        )

    dataset = datasets.ImageFolder(
        root=str(split_dir),
        transform=transform,
    )

    # --------------------------------------------------------
    # Verify class order
    # --------------------------------------------------------

    detected_classes = dataset.classes

    if detected_classes != CLASS_NAMES:

        raise RuntimeError(
            "\nClass order mismatch!\n\n"
            f"Expected:\n{CLASS_NAMES}\n\n"
            f"Detected:\n{detected_classes}"
        )

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)


model = EfficientNetB0Classifier(
    num_classes=NUM_CLASSES,
    pretrained=False,
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)


if not isinstance(
    checkpoint,
    dict
):

    raise RuntimeError(
        "Invalid checkpoint format."
    )


if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)

model.eval()


# ------------------------------------------------------------
# Freeze model
# ------------------------------------------------------------

for parameter in model.parameters():

    parameter.requires_grad = False


print("✓ EfficientNet-B0 checkpoint loaded.")
print("✓ EfficientNet-B0 frozen.")
print(
    "✓ Model ready for feature extraction."
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

@torch.no_grad()
def extract_features(
    dataset,
    split
):

    print("\n")
    print("=" * 70)

    print(
        f"EXTRACTING EFFICIENTNET-B0 FEATURES — "
        f"{split.upper()}"
    )

    print("=" * 70)

    print(
        f"\nImages: {len(dataset)}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    spatial_features = []
    pooled_features = []
    labels = []

    metadata_rows = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = time.time()

    sample_offset = 0

    progress = tqdm(
        loader,
        desc=f"Extracting {split}",
    )

    for images, batch_labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        batch_labels = batch_labels.cpu()

        # ----------------------------------------------------
        # EfficientNet feature extractor
        #
        # Output before global average pooling:
        #
        # [B, 1280, 7, 7]
        # ----------------------------------------------------

        features = model.backbone.features(
            images
        )

        # ----------------------------------------------------
        # Global average pooling
        #
        # [B, 1280, 7, 7]
        #        ↓
        # [B, 1280, 1, 1]
        #        ↓
        # [B, 1280]
        # ----------------------------------------------------

        pooled = model.backbone.avgpool(
            features
        )

        pooled = torch.flatten(
            pooled,
            start_dim=1
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        spatial_cpu = (
            features
            .detach()
            .cpu()
            .to(SPATIAL_DTYPE)
        )

        pooled_cpu = (
            pooled
            .detach()
            .cpu()
            .to(POOLED_DTYPE)
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        spatial_features.append(
            spatial_cpu
        )

        pooled_features.append(
            pooled_cpu
        )

        labels.append(
            batch_labels
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        batch_size = len(
            batch_labels
        )

        for local_idx in range(
            batch_size
        ):

            dataset_idx = (
                sample_offset
                + local_idx
            )

            image_path, label_idx = (
                dataset.samples[
                    dataset_idx
                ]
            )

            metadata_rows.append(
                {
                    "index": dataset_idx,
                    "image_path": str(
                        image_path
                    ),
                    "label_index": int(
                        label_idx
                    ),
                    "label": CLASS_NAMES[
                        int(label_idx)
                    ],
                }
            )

        sample_offset += batch_size

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

    labels = torch.cat(
        labels,
        dim=0
    )


    # ========================================================
    # SHAPE VERIFICATION
    # ========================================================

    expected_spatial = (
        len(dataset),
        1280,
        7,
        7,
    )

    expected_pooled = (
        len(dataset),
        1280,
    )

    if tuple(
        spatial_features.shape
    ) != expected_spatial:

        raise RuntimeError(
            "\nUnexpected spatial feature shape.\n"
            f"Expected: {expected_spatial}\n"
            f"Actual:   {tuple(spatial_features.shape)}"
        )


    if tuple(
        pooled_features.shape
    ) != expected_pooled:

        raise RuntimeError(
            "\nUnexpected pooled feature shape.\n"
            f"Expected: {expected_pooled}\n"
            f"Actual:   {tuple(pooled_features.shape)}"
        )


    if len(labels) != len(dataset):

        raise RuntimeError(
            "\nLabel count mismatch.\n"
            f"Expected: {len(dataset)}\n"
            f"Actual:   {len(labels)}"
        )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    split_output = (
        OUTPUT_DIR / split
    )

    spatial_path = (
        split_output
        / "spatial_features.pt"
    )

    pooled_path = (
        split_output
        / "pooled_features.pt"
    )

    labels_path = (
        split_output
        / "labels.npy"
    )

    metadata_path = (
        split_output
        / "metadata.csv"
    )


    # ========================================================
    # SAVE FEATURES
    # ========================================================

    torch.save(
        spatial_features,
        spatial_path
    )

    torch.save(
        pooled_features,
        pooled_path
    )

    np.save(
        labels_path,
        labels.numpy()
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    spatial_size_mb = (
        spatial_features
        .numel()
        * spatial_features.element_size()
        / (1024 ** 2)
    )

    pooled_size_mb = (
        pooled_features
        .numel()
        * pooled_features.element_size()
        / (1024 ** 2)
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print("\nFeature shapes:")

    print(
        "  Spatial:",
        tuple(
            spatial_features.shape
        )
    )

    print(
        "  Pooled :",
        tuple(
            pooled_features.shape
        )
    )

    print(
        "  Labels :",
        tuple(
            labels.shape
        )
    )

    print("\nStorage:")

    print(
        f"  Spatial features: "
        f"{spatial_size_mb:.2f} MB"
    )

    print(
        f"  Pooled features : "
        f"{pooled_size_mb:.2f} MB"
    )

    print(
        f"\nExtraction time: "
        f"{elapsed:.2f} seconds"
    )


    # ========================================================
    # VERIFY SAVED FILES
    # ========================================================

    required_files = [
        spatial_path,
        pooled_path,
        labels_path,
        metadata_path,
    ]

    for path in required_files:

        if not path.exists():

            raise IOError(
                f"Expected output file was not created:\n"
                f"{path}"
            )


    print("\n✓ Saved spatial features:")
    print(f"  {spatial_path}")

    print("\n✓ Saved pooled features:")
    print(f"  {pooled_path}")

    print("\n✓ Saved labels:")
    print(f"  {labels_path}")

    print("\n✓ Saved metadata:")
    print(f"  {metadata_path}")


    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {
        "split": split,
        "num_images": len(dataset),

        "spatial_shape": list(
            spatial_features.shape
        ),

        "pooled_shape": list(
            pooled_features.shape
        ),

        "labels_shape": list(
            labels.shape
        ),

        "spatial_dtype": str(
            spatial_features.dtype
        ),

        "pooled_dtype": str(
            pooled_features.dtype
        ),

        "spatial_file": str(
            spatial_path
        ),

        "pooled_file": str(
            pooled_path
        ),

        "labels_file": str(
            labels_path
        ),

        "metadata_file": str(
            metadata_path
        ),

        "extraction_time_seconds": elapsed,
    }


# ============================================================
# MAIN EXTRACTION
# ============================================================

summary = {}


for split in [
    "train",
    "val",
    "test",
]:

    dataset = create_dataset(
        split
    )

    summary[split] = extract_features(
        dataset,
        split
    )


# ============================================================
# SAVE EXTRACTION SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "extraction_summary.json"
)


summary_data = {

    "model":
        "EfficientNet-B0",

    "checkpoint":
        str(CHECKPOINT),

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "device":
        str(DEVICE),

    "spatial_representation":
        "[N, 1280, 7, 7]",

    "pooled_representation":
        "[N, 1280]",

    "spatial_dtype":
        "torch.float16",

    "pooled_dtype":
        "torch.float32",

    "splits":
        summary,
}


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_data,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print(
    "EFFICIENTNET-B0 FEATURE EXTRACTION COMPLETE"
)
print("=" * 70)

print("\nFeatures saved to:")
print(OUTPUT_DIR)

print("\nExpected representation:")

print(
    "  Spatial feature map : "
    "(N, 1280, 7, 7)"
)

print(
    "  Global feature      : "
    "(N, 1280)"
)

print("\n✓ Train features extracted.")
print("✓ Validation features extracted.")
print("✓ Test features extracted.")
print("✓ Spatial features saved.")
print("✓ Pooled features saved.")
print("✓ Labels saved.")
print("✓ Metadata saved.")
print("✓ Extraction summary saved.")

print("\nSummary:")
print(summary_path)

print("=" * 70)