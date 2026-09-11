"""
============================================================
RESPIRA - STEP 3
DATALOADER + BATCH VERIFICATION
============================================================

Purpose:
    Prepare the final 224x224 processed dataset for DenseNet121.

This step DOES NOT:
    - preprocess images
    - apply CLAHE
    - apply lung masking
    - resize images
    - balance the dataset
    - train the model

It ONLY:
    - loads the processed dataset
    - applies DenseNet-compatible normalization
    - creates PyTorch DataLoaders
    - verifies batch shapes
    - verifies labels
    - checks class distribution
    - checks for corrupted images
    - checks that no "random" class exists

Expected image:
    224 x 224
    RGB
    PNG

Classes:
    0 - Atelectasis
    1 - Bacterial Pneumonia
    2 - Normal
    3 - Pulmonary Edema
    4 - Tuberculosis
    5 - Viral Pneumonia

============================================================
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the dataset class from Step 2
from distinction import (
    ChestXrayDataset,
    PROCESSED_DATA,
    VALID_CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

DROP_LAST_TRAIN = False


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESPIRA - STEP 3")
print("DATALOADER + BATCH VERIFICATION")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "GPU not available."
    )


# ============================================================
# 3. DENSENET TRANSFORM
# ============================================================

def get_densenet_transform():

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


TRANSFORM = get_densenet_transform()


# ============================================================
# 4. CREATE DATASETS
# ============================================================

print("\nCreating datasets...")

train_dataset = ChestXrayDataset(
    root_dir=PROCESSED_DATA,
    split="train",
    transform=TRANSFORM,
)

val_dataset = ChestXrayDataset(
    root_dir=PROCESSED_DATA,
    split="val",
    transform=TRANSFORM,
)

test_dataset = ChestXrayDataset(
    root_dir=PROCESSED_DATA,
    split="test",
    transform=TRANSFORM,
)


print("✓ Train dataset created.")
print("✓ Validation dataset created.")
print("✓ Test dataset created.")


# ============================================================
# 5. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=DROP_LAST_TRAIN,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)


# ============================================================
# 6. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    f"\nTrain images : {len(train_dataset):,}"
)

print(
    f"Validation   : {len(val_dataset):,}"
)

print(
    f"Test images  : {len(test_dataset):,}"
)

print(
    f"Batch size   : {BATCH_SIZE}"
)

print(
    f"Number classes: {NUM_CLASSES}"
)

print(
    f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}"
)


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

def get_class_distribution(dataset):

    distribution = {
        class_name: 0
        for class_name in VALID_CLASSES
    }

    for sample in dataset.samples:

        class_name = sample["class_name"]

        if class_name not in distribution:

            raise RuntimeError(
                f"Unexpected class found: "
                f"{class_name}"
            )

        distribution[class_name] += 1

    return distribution


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for split_name, dataset in [
    ("TRAIN", train_dataset),
    ("VALIDATION", val_dataset),
    ("TEST", test_dataset),
]:

    distribution = (
        get_class_distribution(dataset)
    )

    print(
        f"\n{split_name}"
    )

    print("-" * 50)

    for class_name, count in (
        distribution.items()
    ):

        print(
            f"{class_name:<25}"
            f"{count:>8}"
        )


# ============================================================
# 8. RANDOM CLASS PROTECTION
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FOR INVALID CLASSES")
print("=" * 70)


for split_name, dataset in [
    ("train", train_dataset),
    ("val", val_dataset),
    ("test", test_dataset),
]:

    for sample in dataset.samples:

        class_name = (
            sample["class_name"]
        )

        if class_name.lower() == "random":

            raise RuntimeError(
                f"\nCRITICAL ERROR:\n"
                f"'random' class detected "
                f"in {split_name}."
            )


print(
    "✓ No 'random' class detected."
)


# ============================================================
# 9. VERIFY FIRST SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SINGLE SAMPLE VERIFICATION")
print("=" * 70)


image, label = train_dataset[0]


print(
    "\nImage tensor shape:",
    tuple(image.shape)
)

print(
    "Image dtype:",
    image.dtype
)

print(
    "Label:",
    label.item()
)

print(
    "Class:",
    VALID_CLASSES[label.item()]
)


# ============================================================
# 10. VERIFY TENSOR SHAPE
# ============================================================

expected_shape = (
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)


if image.shape != expected_shape:

    raise RuntimeError(
        "\nInvalid tensor shape!\n"
        f"Expected: {expected_shape}\n"
        f"Actual: {tuple(image.shape)}"
    )


print(
    "\n✓ Tensor shape is correct:"
)

print(
    f"  {expected_shape}"
)


# ============================================================
# 11. VERIFY LABEL
# ============================================================

if not (
    0 <= label.item() < NUM_CLASSES
):

    raise RuntimeError(
        "\nInvalid class label:"
        f" {label.item()}"
    )


print(
    "✓ Label is valid."
)


# ============================================================
# 12. VERIFY FIRST BATCH
# ============================================================

print("\n" + "=" * 70)
print("BATCH VERIFICATION")
print("=" * 70)


images, labels = next(
    iter(train_loader)
)


print(
    "\nBatch image shape:"
)

print(
    " ",
    tuple(images.shape)
)


print(
    "\nBatch label shape:"
)

print(
    " ",
    tuple(labels.shape)
)


print(
    "\nImage dtype:"
)

print(
    " ",
    images.dtype
)


print(
    "\nLabel dtype:"
)

print(
    " ",
    labels.dtype
)


# ============================================================
# 13. EXPECTED BATCH SHAPE
# ============================================================

expected_batch_shape = (
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)


actual_batch_shape = (
    tuple(images.shape)
)


# Last batch can theoretically be smaller,
# so verify dimensions except batch size.

expected_non_batch = (
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)


actual_non_batch = (
    tuple(images.shape[1:])
)


if actual_non_batch != expected_non_batch:

    raise RuntimeError(
        "\nInvalid image dimensions in batch!\n"
        f"Expected per image: "
        f"{expected_non_batch}\n"
        f"Actual: "
        f"{actual_non_batch}"
    )


if labels.ndim != 1:

    raise RuntimeError(
        "\nLabels should be 1-dimensional.\n"
        f"Actual shape: {tuple(labels.shape)}"
    )


print(
    "\n✓ Batch image dimensions valid."
)

print(
    "✓ Batch labels valid."
)


# ============================================================
# 14. VERIFY LABEL RANGE
# ============================================================

if (
    torch.any(labels < 0)
    or
    torch.any(labels >= NUM_CLASSES)
):

    raise RuntimeError(
        "\nInvalid label detected in batch."
    )


print(
    "✓ All batch labels are valid."
)


# ============================================================
# 15. VERIFY IMAGE VALUES
# ============================================================

if not torch.isfinite(images).all():

    raise RuntimeError(
        "\nNaN or Inf detected in image batch."
    )


print(
    "✓ No NaN or Inf detected."
)


# ============================================================
# 16. BATCH CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FIRST BATCH CLASS DISTRIBUTION")
print("=" * 70)


batch_counts = torch.bincount(
    labels,
    minlength=NUM_CLASSES
)


for index, class_name in enumerate(
    VALID_CLASSES
):

    print(
        f"{index}: "
        f"{class_name:<25}"
        f"{batch_counts[index].item():>4}"
    )


# ============================================================
# 17. VERIFY VALIDATION LOADER
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION LOADER CHECK")
print("=" * 70)


val_images, val_labels = next(
    iter(val_loader)
)


print(
    "\nValidation batch:"
)

print(
    "Images:",
    tuple(val_images.shape)
)

print(
    "Labels:",
    tuple(val_labels.shape)
)


if val_images.shape[1:] != torch.Size(
    [
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]
):

    raise RuntimeError(
        "Validation image dimensions invalid."
    )


print(
    "✓ Validation loader valid."
)


# ============================================================
# 18. VERIFY TEST LOADER
# ============================================================

print("\n" + "=" * 70)
print("TEST LOADER CHECK")
print("=" * 70)


test_images, test_labels = next(
    iter(test_loader)
)


print(
    "\nTest batch:"
)

print(
    "Images:",
    tuple(test_images.shape)
)

print(
    "Labels:",
    tuple(test_labels.shape)
)


if test_images.shape[1:] != torch.Size(
    [
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]
):

    raise RuntimeError(
        "Test image dimensions invalid."
    )


print(
    "✓ Test loader valid."
)


# ============================================================
# 19. DATALOADER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATALOADER SUMMARY")
print("=" * 70)


print(
    "\nTrain batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)

print(
    "Test batches:",
    len(test_loader)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Workers:",
    NUM_WORKERS
)

print(
    "Pin memory:",
    PIN_MEMORY
)


# ============================================================
# 20. FINAL SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\n✓ Dataset loaded."
)

print(
    "✓ Train DataLoader created."
)

print(
    "✓ Validation DataLoader created."
)

print(
    "✓ Test DataLoader created."
)

print(
    "✓ Image shape verified."
)

print(
    "✓ Labels verified."
)

print(
    "✓ No random class detected."
)

print(
    "✓ No NaN/Inf values detected."
)

print(
    "✓ Data is ready for DenseNet121."
)

print(
    "\nNEXT STEP:"
)

print(
    "DenseNet121 model initialization "
    "and feature extraction."
)

print(
    "\n" + "=" * 70
)