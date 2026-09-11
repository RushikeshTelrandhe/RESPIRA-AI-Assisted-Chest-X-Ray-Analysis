# ============================================================
# Respira - AlexNet Training
# ============================================================
#
# Experiment:
#   AlexNet
#
# Dataset:
#   Respira processed Chest X-Ray dataset
#
# Input:
#   224 x 224 RGB
#
# Classes:
#   6
#
# ============================================================

from pathlib import Path
import json
import random
import time

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from src.models.alexnet import (
    create_alexnet
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "alexnet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "alexnet"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
    / "alexnet"
)


for directory in [
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LOG_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

NUM_CLASSES = 6

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0


CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("RESPIRA - ALEXNET TRAINING")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATA_DIR)

print("\nDevice:")
print(DEVICE)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    print(
        "WARNING: CUDA unavailable."
    )


# ============================================================
# 5. TRANSFORMS
# ============================================================

# Preprocessing has already been completed:
#
#   Cleaning
#   Balancing
#   CLAHE
#   Lung masking
#   224x224 resizing
#
# These are only training-time augmentations.


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

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


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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
# 6. DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)


# ============================================================
# 7. CLASS VALIDATION
# ============================================================

expected_classes = sorted(
    CLASS_NAMES
)

actual_classes = sorted(
    train_dataset.classes
)

print("\nClasses:")
print(train_dataset.classes)


if actual_classes != expected_classes:

    raise RuntimeError(
        "\nDataset class mismatch.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {actual_classes}"
    )


# ============================================================
# 8. DATA LOADERS
# ============================================================

PIN_MEMORY = torch.cuda.is_available()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


print("\nDataset sizes:")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ============================================================
# 9. MODEL
# ============================================================

model = create_alexnet(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(
    DEVICE
)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\nModel:")
print("AlexNet")

print(
    "Total parameters:",
    total_parameters
)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# 10. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 12. SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 13. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        correct
        / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 14. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    loss = (
        running_loss
        / total
    )

    accuracy = (
        correct
        / total
    )


    return (
        loss,
        accuracy
    )


# ============================================================
# 15. TRAINING
# ============================================================

best_val_accuracy = 0.0

history = []


print("\n")
print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    start_time = time.time()


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        evaluate(
            val_loader
        )
    )


    scheduler.step(
        val_loss
    )


    elapsed = (
        time.time()
        - start_time
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss": val_loss,

        "val_accuracy":
            val_accuracy,

        "learning_rate":
            current_lr,

        "time_seconds":
            elapsed
    })


    print(
        f"\nEpoch "
        f"[{epoch}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )

    print(
        f"Time: "
        f"{elapsed:.2f}s"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,

            "class_names":
                CLASS_NAMES,

            "num_classes":
                NUM_CLASSES,

            "image_size":
                IMAGE_SIZE,

            "model_name":
                "AlexNet"
        }


        checkpoint_path = (
            CHECKPOINT_DIR
            / "best_model.pth"
        )


        torch.save(
            checkpoint,
            checkpoint_path
        )


        print(
            "✓ Best model saved."
        )


# ============================================================
# 16. SAVE HISTORY
# ============================================================

history_path = (
    LOG_DIR
    / "training_history.json"
)


with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# 17. LOAD BEST MODEL
# ============================================================

checkpoint_path = (
    CHECKPOINT_DIR
    / "best_model.pth"
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# 18. TEST
# ============================================================

test_loss, test_accuracy = (
    evaluate(
        test_loader
    )
)


print("\n")
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# 19. EXPERIMENT SUMMARY
# ============================================================

summary = {

    "model":
        "AlexNet",

    "dataset":
        "Respira Chest X-Ray",

    "image_size":
        IMAGE_SIZE,

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        NUM_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "best_validation_accuracy":
        best_val_accuracy,

    "test_accuracy":
        test_accuracy,

    "device":
        str(DEVICE),

    "cuda_available":
        torch.cuda.is_available(),

    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

    "torch_version":
        torch.__version__
}


summary_path = (
    OUTPUT_DIR
    / "experiment_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 20. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("ALEXNET TRAINING COMPLETE")
print("=" * 70)

print(
    "\nCheckpoint:",
    checkpoint_path
)

print(
    "History:",
    history_path
)

print(
    "Summary:",
    summary_path
)