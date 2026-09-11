"""
============================================================
RESPIRA
DenseNet121 Training
============================================================

STEP 4: TRAIN DENSENET121

Purpose
-------
Train the DenseNet121 CNN branch on the preprocessed
Respira chest X-ray dataset.

Dataset structure
----------------

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
    └── test/

Training outputs
----------------

outputs/densenet/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
│
├── logs/
│   └── training_history.csv
│
├── metrics/
│   └── training_summary.json
│
└── plots/
    ├── loss_curve.png
    └── accuracy_curve.png

============================================================
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.models.densenet import (
    DenseNet121Model,
    CLASS_NAMES,
    NUM_CLASSES,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
)


CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)


LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)


METRICS_DIR = (
    OUTPUT_DIR
    / "metrics"
)


PLOT_DIR = (
    OUTPUT_DIR
    / "plots"
)


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    METRICS_DIR,
    PLOT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# 4. TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224


BATCH_SIZE = 16


NUM_EPOCHS = 30


LEARNING_RATE = 1e-4


WEIGHT_DECAY = 1e-4


PATIENCE = 7


NUM_WORKERS = 0


DROPOUT = 0.2


PRETRAINED = True


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)

print("RESPIRA — DENSENET121 TRAINING")

print("=" * 70)

print()

print("Project root:")
print(PROJECT_ROOT)

print()

print("Dataset:")
print(DATA_DIR)

print()

print("Output:")
print(OUTPUT_DIR)

print()

print("Device:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 6. DATASET VALIDATION
# ============================================================

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


for split_dir in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:

    if not split_dir.exists():

        raise FileNotFoundError(
            f"\nDataset split does not exist:\n"
            f"{split_dir}"
        )


# ============================================================
# 7. DATA TRANSFORMS
# ============================================================

# Your preprocessing already performs:
#
#   Cleaning
#   Balancing
#   CLAHE
#   Lung masking
#   224 x 224 conversion
#
# Therefore we do NOT repeat CLAHE or masking here.


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
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
# 8. LOAD DATASETS
# ============================================================

train_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)


val_dataset = ImageFolder(
    root=VAL_DIR,
    transform=eval_transform
)


test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=eval_transform
)


# ============================================================
# 9. VERIFY CLASS ORDER
# ============================================================

print("=" * 70)

print("DATASET CLASSES")

print("=" * 70)

print()

print(
    "Detected:",
    train_dataset.classes
)

print()

print(
    "Expected:",
    CLASS_NAMES
)

print()


if train_dataset.classes != CLASS_NAMES:

    raise RuntimeError(

        "\nClass order mismatch!\n\n"

        f"Detected:\n"
        f"{train_dataset.classes}\n\n"

        f"Expected:\n"
        f"{CLASS_NAMES}\n\n"

        "Check your dataset directory names."
    )


if val_dataset.classes != CLASS_NAMES:

    raise RuntimeError(
        "Validation class order does not match training."
    )


if test_dataset.classes != CLASS_NAMES:

    raise RuntimeError(
        "Test class order does not match training."
    )


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),
)


print("=" * 70)

print("DATASET INFORMATION")

print("=" * 70)

print()

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "Batch size:",
    BATCH_SIZE
)

print()


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = DenseNet121Model(

    num_classes=NUM_CLASSES,

    pretrained=PRETRAINED,

    dropout=DROPOUT,

    freeze_backbone=False,
)


model = model.to(DEVICE)


# ============================================================
# 12. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 14. LEARNING RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=2,
)


# ============================================================
# 15. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress = tqdm(

        train_loader,

        desc="Training",

        leave=False
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(images)


        logits = outputs["logits"]


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )


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
# 16. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        progress = tqdm(

            val_loader,

            desc="Validation",

            leave=False
        )


        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(images)


            logits = outputs["logits"]


            loss = criterion(
                logits,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                logits,
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
# 17. TRAINING HISTORY
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],

    "learning_rate": [],

}


# ============================================================
# 18. TRAINING LOOP
# ============================================================

best_val_loss = float("inf")


best_val_accuracy = 0.0


epochs_without_improvement = 0


training_start = time.time()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print()

    print("=" * 70)

    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    val_loss, val_accuracy = (
        validate()
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["learning_rate"].append(
        current_lr
    )


    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.4f}"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : {val_accuracy:.4f}"
    )

    print(
        f"Learning Rate   : {current_lr:.8f}"
    )


    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if (
        val_loss < best_val_loss
        or
        val_accuracy > best_val_accuracy
    ):

        best_val_loss = min(
            best_val_loss,
            val_loss
        )

        best_val_accuracy = max(
            best_val_accuracy,
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

            "best_val_loss":
                best_val_loss,

            "best_val_accuracy":
                best_val_accuracy,

            "class_names":
                CLASS_NAMES,

            "num_classes":
                NUM_CLASSES,

            "image_size":
                IMAGE_SIZE,

            "model_name":
                "DenseNet121",

            "seed":
                SEED,

        }


        torch.save(

            checkpoint,

            CHECKPOINT_DIR
            / "best_model.pth"
        )


        print()

        print(
            "✓ Best model checkpoint saved."
        )


        epochs_without_improvement = 0


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    torch.save(

        {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "class_names":
                CLASS_NAMES,

        },

        CHECKPOINT_DIR
        / "last_model.pth"
    )


    # --------------------------------------------------------
    # Save CSV after every epoch
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        LOG_DIR
        / "training_history.csv",

        index=False
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# 19. TRAINING TIME
# ============================================================

training_time = (
    time.time()
    - training_start
)


# ============================================================
# 20. SAVE TRAINING SUMMARY
# ============================================================

summary = {

    "model":
        "DenseNet121",

    "pretrained":
        PRETRAINED,

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        NUM_EPOCHS,

    "epochs_completed":
        len(history["epoch"]),

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "best_validation_loss":
        best_val_loss,

    "best_validation_accuracy":
        best_val_accuracy,

    "training_time_seconds":
        training_time,

    "device":
        str(DEVICE),

}


with open(

    METRICS_DIR
    / "training_summary.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        summary,

        file,

        indent=4
    )


# ============================================================
# 21. TRAINING CURVES — LOSS
# ============================================================

history_df = pd.DataFrame(
    history
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    history_df["epoch"],

    history_df["train_loss"],

    label="Train Loss"
)


plt.plot(

    history_df["epoch"],

    history_df["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "DenseNet121 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    PLOT_DIR
    / "loss_curve.png",

    dpi=300
)


plt.close()


# ============================================================
# 22. TRAINING CURVES — ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    history_df["epoch"],

    history_df["train_accuracy"],

    label="Train Accuracy"
)


plt.plot(

    history_df["epoch"],

    history_df["val_accuracy"],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "DenseNet121 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    PLOT_DIR
    / "accuracy_curve.png",

    dpi=300
)


plt.close()


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "DENSENET121 TRAINING COMPLETE"
)

print("=" * 70)

print()

print(
    "Best checkpoint:"
)

print(
    CHECKPOINT_DIR
    / "best_model.pth"
)

print()

print(
    "Training history:"
)

print(
    LOG_DIR
    / "training_history.csv"
)

print()

print(
    "Training summary:"
)

print(
    METRICS_DIR
    / "training_summary.json"
)

print()

print(
    "Plots:"
)

print(
    PLOT_DIR
)

print()

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print()

print(
    "✓ DenseNet121 training finished."
)

print("=" * 70)