# ============================================================
# RESPIRA - ViT-B/16 512x512 TRAINING
# ============================================================

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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.vit import VisionTransformerModel


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "Lung_Disease_Preprocessed_512"
)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    EVALUATION_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 512
NUM_CLASSES = 6

# ViT-B/16 at 512 is substantially more memory-intensive
# than EfficientNet because it processes 1024 patch tokens.
BATCH_SIZE = 8

NUM_WORKERS = 0

TOTAL_EPOCHS = 30

# Classifier warm-up
STAGE1_EPOCHS = 3

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 7

SEED = 42

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

def set_seed(seed=42):

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
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

AMP_ENABLED = DEVICE.type == "cuda"

print("=" * 70)
print("RESPIRA - ViT-B/16 512x512 TRAINING")
print("=" * 70)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Image Size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Epochs       : {TOTAL_EPOCHS}")
print(f"Mixed AMP    : {AMP_ENABLED}")

print(f"Dataset      : {DATA_DIR}")
print(f"Output       : {OUTPUT_DIR}")

print("=" * 70)


# ============================================================
# 5. VERIFY DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:

    if not directory.exists():

        raise FileNotFoundError(
            f"Required directory not found:\n{directory}"
        )


# ============================================================
# 6. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5,
        fill=0
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 7. DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

# Loaded only to verify the split/class structure.
# It is NOT used for model selection.
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)


# ============================================================
# 8. CLASS VERIFICATION
# ============================================================

print("\nClass verification:")

print("Train:", train_dataset.classes)
print("Val  :", val_dataset.classes)
print("Test :", test_dataset.classes)

if train_dataset.classes != CLASS_NAMES:

    raise ValueError(
        f"Train class order mismatch.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found: {train_dataset.classes}"
    )

if val_dataset.classes != CLASS_NAMES:

    raise ValueError(
        f"Validation class order mismatch.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found: {val_dataset.classes}"
    )

if test_dataset.classes != CLASS_NAMES:

    raise ValueError(
        f"Test class order mismatch.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found: {test_dataset.classes}"
    )

print("✓ Class order verified")


# ============================================================
# 9. DATASET INFORMATION
# ============================================================

print("\nDataset:")

print(
    f"Train images : "
    f"{len(train_dataset)}"
)

print(
    f"Val images   : "
    f"{len(val_dataset)}"
)

print(
    f"Test images  : "
    f"{len(test_dataset)}"
)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

class_counts = {}

for index, class_name in enumerate(CLASS_NAMES):

    count = sum(
        1
        for label in train_dataset.targets
        if label == index
    )

    class_counts[class_name] = count

    print(
        f"{class_name:<25} : {count}"
    )


# ============================================================
# 11. DATALOADERS
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
# 12. CREATE ViT MODEL
# ============================================================

print("\nLoading pretrained ViT-B/16...")

model = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=True,
)

model = model.to(DEVICE)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)


# ============================================================
# 13. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. MIXED PRECISION
# ============================================================

if AMP_ENABLED:

    try:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True
        )

    except AttributeError:

        scaler = torch.cuda.amp.GradScaler(
            enabled=True
        )

else:

    scaler = None


def autocast_context():

    if AMP_ENABLED:

        try:

            return torch.amp.autocast(
                device_type="cuda",
                enabled=True
            )

        except AttributeError:

            return torch.cuda.amp.autocast(
                enabled=True
            )

    return torch.autocast(
        device_type="cpu",
        enabled=False
    )


# ============================================================
# 15. FREEZE / UNFREEZE
# ============================================================

def freeze_vit_backbone(model):

    """
    Freeze the pretrained ViT encoder and train only
    the final classification head.
    """

    # The modified Respira ViT contains:
    # model.backbone
    # model.classifier

    for parameter in model.backbone.parameters():

        parameter.requires_grad = False

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True

    model.backbone.eval()
    model.classifier.train()


def unfreeze_vit(model):

    """
    Unfreeze the complete ViT for fine-tuning.
    """

    for parameter in model.parameters():

        parameter.requires_grad = True

    model.train()


# ============================================================
# 16. OUTPUT EXTRACTION
# ============================================================

def get_logits(outputs):

    if isinstance(outputs, dict):

        return outputs["logits"]

    return outputs


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    freeze_backbone=False,
):

    model.train()

    if freeze_backbone:

        model.backbone.eval()
        model.classifier.train()

    running_loss = 0.0

    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in loader:

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

        with autocast_context():

            outputs = model(images)

            logits = get_logits(outputs)

            loss = criterion(
                logits,
                labels
            )

        if AMP_ENABLED:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        epoch_loss,
        epoch_accuracy,
        elapsed
    )


# ============================================================
# 18. VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

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

        with autocast_context():

            outputs = model(images)

            logits = get_logits(outputs)

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

    val_loss = (
        running_loss / total
    )

    val_accuracy = (
        correct / total
    )

    return (
        val_loss,
        val_accuracy
    )


# ============================================================
# 19. STAGE 1
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 - CLASSIFIER WARM-UP")
print("=" * 70)

freeze_vit_backbone(model)

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=STAGE1_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 20. TRAINING STATE
# ============================================================

history = []

best_val_accuracy = -1.0
best_val_loss = float("inf")

best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# 21. MAIN TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    TOTAL_EPOCHS + 1
):

    # --------------------------------------------------------
    # Stage transition
    # --------------------------------------------------------

    if epoch == STAGE1_EPOCHS + 1:

        print("\n")
        print("=" * 70)
        print("SWITCHING TO STAGE 2 - FULL ViT FINE-TUNING")
        print("=" * 70)

        unfreeze_vit(model)

        optimizer = AdamW(
            model.parameters(),
            lr=STAGE2_LR,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        epochs_without_improvement = 0


    # --------------------------------------------------------
    # Stage configuration
    # --------------------------------------------------------

    if epoch <= STAGE1_EPOCHS:

        current_stage = (
            "classifier_warmup"
        )

        freeze_backbone = True

    else:

        current_stage = (
            "full_finetuning"
        )

        freeze_backbone = False


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_accuracy, epoch_time = (
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            freeze_backbone=freeze_backbone,
        )
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_accuracy = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
    )


    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "stage": current_stage,

        "learning_rate": current_lr,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "epoch_time_seconds": epoch_time,
    })


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch:02d}/{TOTAL_EPOCHS}] "
        f"| {current_stage:<18} "
        f"| LR {current_lr:.2e} "
        f"| Train Loss {train_loss:.4f} "
        f"| Train Acc {train_accuracy * 100:.2f}% "
        f"| Val Loss {val_loss:.4f} "
        f"| Val Acc {val_accuracy * 100:.2f}% "
        f"| {epoch_time:.1f}s"
    )


    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        best_val_loss = val_loss

        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_accuracy":
                val_accuracy,

            "val_loss":
                val_loss,

            "class_names":
                CLASS_NAMES,

            "class_to_idx":
                train_dataset.class_to_idx,

            "image_size":
                IMAGE_SIZE,

            "num_classes":
                NUM_CLASSES,

            "model_name":
                "ViT-B/16",

            "pretrained":
                True,

            "patch_size":
                16,

            "num_patch_tokens":
                1024,

            "embedding_dimension":
                768,

            "seed":
                SEED,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "best_model.pth"
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Acc: "
            f"{val_accuracy * 100:.2f}%)"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epoch > STAGE1_EPOCHS
        and
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            f"\nEarly stopping triggered "
            f"at epoch {epoch}."
        )

        break


# ============================================================
# 22. SAVE LAST CHECKPOINT
# ============================================================

last_checkpoint = {

    "epoch": epoch,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "val_accuracy":
        val_accuracy,

    "val_loss":
        val_loss,

    "class_names":
        CLASS_NAMES,

    "class_to_idx":
        train_dataset.class_to_idx,

    "image_size":
        IMAGE_SIZE,

    "num_classes":
        NUM_CLASSES,

    "model_name":
        "ViT-B/16",

    "pretrained":
        True,

    "patch_size":
        16,

    "num_patch_tokens":
        1024,

    "embedding_dimension":
        768,

    "seed":
        SEED,
}

torch.save(
    last_checkpoint,
    CHECKPOINT_DIR
    / "last_model.pth"
)


# ============================================================
# 23. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_DIR
    / "training_history.csv",
    index=False
)


# ============================================================
# 24. LOSS PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ViT-B/16 512x512 - Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "loss_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# 25. ACCURACY PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"] * 100,
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"] * 100,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "ViT-B/16 512x512 - Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "accuracy_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# 26. EXPERIMENT SUMMARY
# ============================================================

summary = {

    "model":
        "ViT-B/16",

    "input_size":
        "512x512",

    "patch_size":
        16,

    "patch_grid":
        "32x32",

    "num_patch_tokens":
        1024,

    "embedding_dimension":
        768,

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "dataset":
        str(DATA_DIR),

    "train_images":
        len(train_dataset),

    "validation_images":
        len(val_dataset),

    "test_images":
        len(test_dataset),

    "batch_size":
        BATCH_SIZE,

    "total_epochs_configured":
        TOTAL_EPOCHS,

    "epochs_completed":
        len(history),

    "stage1_epochs":
        STAGE1_EPOCHS,

    "stage1_learning_rate":
        STAGE1_LR,

    "stage2_learning_rate":
        STAGE2_LR,

    "weight_decay":
        WEIGHT_DECAY,

    "pretrained":
        True,

    "mixed_precision":
        AMP_ENABLED,

    "seed":
        SEED,

    "best_epoch":
        best_epoch,

    "best_validation_accuracy":
        best_val_accuracy,

    "best_validation_loss":
        best_val_loss,

    "class_distribution":
        class_counts,

    "checkpoint":
        str(
            CHECKPOINT_DIR
            / "best_model.pth"
        ),
}


with open(
    OUTPUT_DIR
    / "experiment_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("ViT-B/16 512x512 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Epoch          : "
    f"{best_epoch}"
)

print(
    f"Best Validation Acc : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print("\nBest checkpoint:")

print(
    CHECKPOINT_DIR
    / "best_model.pth"
)

print("\nTraining history:")

print(
    OUTPUT_DIR
    / "training_history.csv"
)

print("\nPlots:")

print(
    PLOTS_DIR
    / "loss_curve.png"
)

print(
    PLOTS_DIR
    / "accuracy_curve.png"
)

print("=" * 70)

print(
    "\nIMPORTANT: Test-set accuracy was NOT calculated "
    "during training."
)

print(
    "The test set will be evaluated separately using "
    "the best validation checkpoint."
)

print("=" * 70)