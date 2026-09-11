# ============================================================
# RESPIRA - VISION TRANSFORMER TRAINING
# ============================================================
#
# File:
#   src/training/train_vit.py
#
# Purpose:
#   Train pretrained ViT-B/16 on the Respira processed
#   chest X-ray dataset.
#
# Dataset:
#   data/processed/
#
# Classes:
#   1. Atelectasis
#   2. Bacterial Pneumonia
#   3. Normal
#   4. Pulmonary Edema
#   5. Tuberculosis
#   6. Viral Pneumonia
#
# Output:
#   outputs/vit/
#       checkpoints/
#       logs/
#       metrics/
#       plots/
#
# ============================================================

from __future__ import annotations

import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from src.models.vit import (
    VisionTransformerModel,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

NUM_CLASSES = 6

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 7

MODEL_NAME = "ViT-B/16"


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
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

PLOTS_DIR = (
    OUTPUT_DIR
    / "plots"
)


for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    METRICS_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):
    """
    ImageFolder-style dataset for the processed Respira
    dataset.

    Expected structure:

        processed/
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
    """

    EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        root_dir: Path,
        split: str,
        transform=None,
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.split_dir = (
            self.root_dir / split
        )

        if not self.split_dir.exists():

            raise FileNotFoundError(
                f"Dataset split does not exist:\n"
                f"{self.split_dir}"
            )

        self.class_names = CLASS_NAMES

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.class_names
            )
        }

        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.split_dir}"
            )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    def _collect_samples(self):

        for class_name in self.class_names:

            class_dir = (
                self.split_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Required class directory missing:\n"
                    f"{class_dir}"
                )

            label = self.class_to_idx[
                class_name
            ]

            images = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if (
                        p.is_file()
                        and
                        p.suffix.lower()
                        in self.EXTENSIONS
                    )
                ]
            )

            for image_path in images:

                self.samples.append(
                    (
                        image_path,
                        label,
                    )
                )

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------
    # Get item
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_path, label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

# ViT pretrained weights expect ImageNet normalization.

TRAIN_TRANSFORM = transforms.Compose([

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
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


VAL_TRANSFORM = transforms.Compose([

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
])


# ============================================================
# CREATE DATASETS
# ============================================================

print("=" * 70)
print("RESPIRA - VISION TRANSFORMER TRAINING")
print("=" * 70)

print("\nDevice:")

print(
    " ",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("\nProcessed dataset:")
print(
    " ",
    DATA_DIR
)


train_dataset = ChestXrayDataset(
    root_dir=DATA_DIR,
    split="train",
    transform=TRAIN_TRANSFORM,
)


val_dataset = ChestXrayDataset(
    root_dir=DATA_DIR,
    split="val",
    transform=VAL_TRANSFORM,
)


# ============================================================
# DATA LOADERS
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


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset:")
print(
    "  Train images:",
    len(train_dataset)
)

print(
    "  Validation images:",
    len(val_dataset)
)

print(
    "  Classes:",
    CLASS_NAMES
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ViT-B/16...")

model = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=True,
)

model = model.to(DEVICE)


# ============================================================
# PARAMETER INFORMATION
# ============================================================

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
print(
    "  Architecture:",
    MODEL_NAME
)

print(
    "  Total parameters:",
    f"{total_parameters:,}"
)

print(
    "  Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2,
)


# ============================================================
# TRAINING STORAGE
# ============================================================

history = []

best_val_loss = float("inf")

best_val_accuracy = 0.0

epochs_without_improvement = 0

start_time = time.time()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
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

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(images)

        logits = output["logits"]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
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

        output = model(images)

        logits = output["logits"]

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n" + "=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        validate()
    )

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "learning_rate": current_lr,

    })

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\nTrain Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.8f}"
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    is_best = (
        val_loss < best_val_loss
    )

    if is_best:

        best_val_loss = val_loss

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0

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
                MODEL_NAME,

            "seed":
                SEED,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "best_model.pth"
        )

        print(
            "\n✓ Best model checkpoint saved."
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    latest_checkpoint = {

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
            MODEL_NAME,

        "seed":
            SEED,
    }

    torch.save(
        latest_checkpoint,
        CHECKPOINT_DIR
        / "last_model.pth"
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"\nEarly stopping triggered "
            f"after {PATIENCE} epochs "
            f"without validation improvement."
        )

        break


# ============================================================
# TRAINING TIME
# ============================================================

training_time = (
    time.time()
    - start_time
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    LOG_DIR
    / "training_history.csv",
    index=False
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

summary = {

    "model":
        "ViT-B/16",

    "pretrained":
        True,

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(history),

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
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

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

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "ViT-B/16 Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "loss_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# PLOT TRAINING ACCURACY
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

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "ViT-B/16 Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "accuracy_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "VISION TRANSFORMER TRAINING COMPLETE"
)

print("=" * 70)

print("\nBest checkpoint:")

print(
    CHECKPOINT_DIR
    / "best_model.pth"
)

print("\nTraining history:")

print(
    LOG_DIR
    / "training_history.csv"
)

print("\nTraining summary:")

print(
    METRICS_DIR
    / "training_summary.json"
)

print("\nPlots:")

print(
    PLOTS_DIR
)

print(
    f"\nBest validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "\n✓ ViT-B/16 training finished."
)