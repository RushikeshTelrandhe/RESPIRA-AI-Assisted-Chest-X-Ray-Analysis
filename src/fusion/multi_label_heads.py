# ============================================================
# Respira - STEP 7.7
# Multi-Label Classification Heads
# ============================================================
#
# Input:
#   Disease features:
#       (N, 6, 512)
#
#   Disease relationship matrix:
#       (N, 6, 6)
#
# Output:
#   Logits:
#       (N, 6)
#
#   Probabilities:
#       (N, 6)
#
#   Uncertainty:
#       (N, 6)
#
# Classes:
#   0: Atelectasis
#   1: Bacterial Pneumonia
#   2: Normal
#   3: Pulmonary Edema
#   4: Tuberculosis
#   5: Viral Pneumonia
#
# ============================================================

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 6
FEATURE_DIM = 512

BATCH_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 7

THRESHOLD = 0.5

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "disease_relationship"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "classification"
)


CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PLOTS_DIR = OUTPUT_ROOT / "plots"
METRICS_DIR = OUTPUT_ROOT / "metrics"
LOGS_DIR = OUTPUT_ROOT / "logs"

TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"
TEST_DIR = OUTPUT_ROOT / "test"


# ============================================================
# CREATE DIRECTORIES AUTOMATICALLY
# ============================================================

for directory in [
    OUTPUT_ROOT,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    LOGS_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("RESPIRA - STEP 7.7")
print("MULTI-LABEL CLASSIFICATION HEADS")
print("=" * 70)

print()
print(f"Project root: {PROJECT_ROOT}")
print(f"Input root  : {INPUT_ROOT}")
print(f"Output root : {OUTPUT_ROOT}")

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


print()
print("Classes:")

for i, name in enumerate(CLASS_NAMES):

    print(f"  {i}: {name}")


# ============================================================
# DATASET
# ============================================================

class DiseaseRelationshipDataset(Dataset):

    def __init__(
        self,
        split
    ):

        self.split = split

        split_dir = INPUT_ROOT / split

        # STEP 7.4 disease-conditioned features
        disease_path = (
            PROJECT_ROOT
            / "outputs"
            / "fusion"
            / "disease_attention"
            / split
            / "disease_features.pt"
        )

        # STEP 7.6 disease relationship matrix
        relationship_path = (
            split_dir
            / "relationship_matrix.pt"
        )

        # STEP 7.6 labels
        labels_path = (
            split_dir
            / "labels.npy"
        )
        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not disease_path.exists():

            raise FileNotFoundError(
                f"\nDisease features not found:\n"
                f"{disease_path}"
            )

        if not relationship_path.exists():

            raise FileNotFoundError(
                f"\nRelationship matrix not found:\n"
                f"{relationship_path}"
            )

        if not labels_path.exists():

            raise FileNotFoundError(
                f"\nLabels not found:\n"
                f"{labels_path}"
            )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        self.disease_features = torch.load(
            disease_path,
            map_location="cpu",
            weights_only=True
        ).float()

        self.relationship = torch.load(
            relationship_path,
            map_location="cpu",
            weights_only=True
        ).float()

        self.labels = np.load(
            labels_path
        )

        self.labels = torch.from_numpy(
            self.labels
        ).long()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print()
        print(f"{split.upper()} dataset")

        print(
            f"  Disease features : "
            f"{tuple(self.disease_features.shape)}"
        )

        print(
            f"  Relationship     : "
            f"{tuple(self.relationship.shape)}"
        )

        print(
            f"  Labels           : "
            f"{tuple(self.labels.shape)}"
        )

        if self.disease_features.ndim != 3:

            raise RuntimeError(
                f"{split}: expected disease "
                f"features [N,6,512]"
            )

        if self.disease_features.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                f"{split}: expected 6 disease features"
            )

        if self.disease_features.shape[2] != FEATURE_DIM:

            raise RuntimeError(
                f"{split}: expected feature dimension 512"
            )

        if self.relationship.ndim != 3:

            raise RuntimeError(
                f"{split}: expected relationship "
                f"matrix [N,6,6]"
            )

        if self.relationship.shape[1:] != (
            NUM_CLASSES,
            NUM_CLASSES,
        ):

            raise RuntimeError(
                f"{split}: relationship matrix "
                f"must be [N,6,6]"
            )

        if len(self.labels) != len(
            self.disease_features
        ):

            raise RuntimeError(
                f"{split}: sample count mismatch"
            )

        if len(self.relationship) != len(
            self.disease_features
        ):

            raise RuntimeError(
                f"{split}: relationship sample "
                f"count mismatch"
            )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not torch.isfinite(
            self.disease_features
        ).all():

            raise RuntimeError(
                f"{split}: NaN/Inf in disease features"
            )

        if not torch.isfinite(
            self.relationship
        ).all():

            raise RuntimeError(
                f"{split}: NaN/Inf in relationship matrix"
            )

        print("  ✓ Shapes valid")
        print("  ✓ Sample counts valid")
        print("  ✓ No NaN / Inf")

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.disease_features[index],
            self.relationship[index],
            self.labels[index],
        )


# ============================================================
# MODEL
# ============================================================

class MultiLabelClassificationHeads(nn.Module):
    """
    Disease-specific classification heads.

    Disease features:
        [B, 6, 512]

    Relationship matrix:
        [B, 6, 6]

    Each disease representation receives
    information from the other disease representations
    using the relationship matrix.

    Final output:
        [B, 6]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        dropout=0.3,
    ):

        super().__init__()

        self.num_classes = num_classes

        self.feature_dim = feature_dim

        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Disease-specific transformation
        # ----------------------------------------------------

        self.disease_projection = nn.Sequential(

            nn.Linear(
                feature_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ----------------------------------------------------
        # Six independent classification heads
        # ----------------------------------------------------

        self.classification_heads = nn.ModuleList()

        for _ in range(num_classes):

            head = nn.Sequential(

                nn.Linear(
                    hidden_dim,
                    hidden_dim // 2
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    hidden_dim // 2,
                    1
                )
            )

            self.classification_heads.append(
                head
            )

    def forward(
        self,
        disease_features,
        relationship
    ):

        # ----------------------------------------------------
        # disease_features
        #
        # [B, 6, 512]
        # ----------------------------------------------------

        projected = self.disease_projection(
            disease_features
        )

        # ----------------------------------------------------
        # Relationship propagation
        #
        # relationship:
        # [B, 6, 6]
        #
        # projected:
        # [B, 6, 256]
        #
        # result:
        # [B, 6, 256]
        # ----------------------------------------------------

        relational_features = torch.bmm(
            relationship,
            projected
        )

        # ----------------------------------------------------
        # Residual combination
        # ----------------------------------------------------

        enhanced = (
            projected
            + relational_features
        )

        # ----------------------------------------------------
        # Disease-specific heads
        # ----------------------------------------------------

        logits = []

        for disease_index in range(
            self.num_classes
        ):

            disease_feature = enhanced[
                :,
                disease_index,
                :
            ]

            disease_logit = (
                self.classification_heads[
                    disease_index
                ](
                    disease_feature
                )
            )

            logits.append(
                disease_logit
            )

        logits = torch.cat(
            logits,
            dim=1
        )

        return logits


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_dataset = DiseaseRelationshipDataset(
    "train"
)

val_dataset = DiseaseRelationshipDataset(
    "val"
)

test_dataset = DiseaseRelationshipDataset(
    "test"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_labels(
    labels
):

    """
    Converts single-class labels:

        [0, 1, 2, 3...]

    into one-hot multi-label targets:

        [1,0,0,0,0,0]
        [0,1,0,0,0,0]
        ...
    """

    one_hot = torch.zeros(
        labels.size(0),
        NUM_CLASSES,
        device=labels.device
    )

    one_hot.scatter_(
        1,
        labels.unsqueeze(1),
        1.0
    )

    return one_hot


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING CLASSIFICATION HEADS")
print("=" * 70)

model = MultiLabelClassificationHeads(
    num_classes=NUM_CLASSES,
    feature_dim=FEATURE_DIM,
    hidden_dim=256,
    dropout=0.3,
).to(DEVICE)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    total_samples = 0

    correct = 0

    for (
        disease_features,
        relationship,
        labels,
    ) in train_loader:

        disease_features = (
            disease_features.to(
                DEVICE,
                non_blocking=True
            )
        )

        relationship = (
            relationship.to(
                DEVICE,
                non_blocking=True
            )
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        targets = convert_labels(
            labels
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            disease_features,
            relationship
        )

        loss = criterion(
            logits,
            targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct
        / total_samples
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    total_samples = 0

    correct = 0

    for (
        disease_features,
        relationship,
        labels,
    ) in val_loader:

        disease_features = (
            disease_features.to(
                DEVICE,
                non_blocking=True
            )
        )

        relationship = (
            relationship.to(
                DEVICE,
                non_blocking=True
            )
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        targets = convert_labels(
            labels
        )

        logits = model(
            disease_features,
            relationship
        )

        loss = criterion(
            logits,
            targets
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct
        / total_samples
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING MULTI-LABEL CLASSIFICATION HEADS")
print("=" * 70)

history = []

best_val_loss = float("inf")

best_val_accuracy = 0.0

epochs_without_improvement = 0

best_checkpoint = (
    CHECKPOINT_DIR
    / "best_model.pth"
)

training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = validate()

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr,
        "time_seconds": epoch_time,
    }

    history.append(
        record
    )

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )
    print("=" * 70)

    print(
        f"Train Loss      : "
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

    print(
        f"Time            : "
        f"{epoch_time:.2f}s"
    )

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    improved = (
        val_loss < best_val_loss
    )

    if improved:

        best_val_loss = val_loss

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "feature_dim": FEATURE_DIM,
                "seed": SEED,
            },
            best_checkpoint
        )

        print(
            "✓ Best checkpoint saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


training_time = (
    time.time()
    - training_start
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    LOGS_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


with open(
    LOGS_DIR
    / "training_history.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING BEST CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    best_checkpoint,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "✓ Best classification-head checkpoint loaded."
)


# ============================================================
# TEST INFERENCE
# ============================================================

@torch.no_grad()
def predict_dataset(
    loader
):

    model.eval()

    all_logits = []

    all_probabilities = []

    all_labels = []

    for (
        disease_features,
        relationship,
        labels,
    ) in loader:

        disease_features = (
            disease_features.to(
                DEVICE,
                non_blocking=True
            )
        )

        relationship = (
            relationship.to(
                DEVICE,
                non_blocking=True
            )
        )

        logits = model(
            disease_features,
            relationship
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_logits.append(
            logits.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_labels.append(
            labels
        )

    logits = torch.cat(
        all_logits
    ).numpy()

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy()

    return (
        logits,
        probabilities,
        labels
    )


print()
print("=" * 70)
print("RUNNING TEST INFERENCE")
print("=" * 70)

test_logits, test_probabilities, test_labels = (
    predict_dataset(
        test_loader
    )
)

print(
    f"\nInference completed: "
    f"{len(test_labels)} images"
)


# ============================================================
# MULTI-LABEL PREDICTIONS
# ============================================================

test_binary = (
    test_probabilities
    >= THRESHOLD
).astype(int)


# ============================================================
# PRIMARY SINGLE-CLASS PREDICTION
# ============================================================

predicted_class = np.argmax(
    test_probabilities,
    axis=1
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    predicted_class
)

precision_macro = precision_score(
    test_labels,
    predicted_class,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    test_labels,
    predicted_class,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    test_labels,
    predicted_class,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    test_labels,
    predicted_class,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    test_labels,
    predicted_class,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    test_labels,
    predicted_class,
    average="weighted",
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

one_hot_labels = np.eye(
    NUM_CLASSES
)[test_labels]

roc_auc_per_class = {}

for i, class_name in enumerate(
    CLASS_NAMES
):

    try:

        auc = roc_auc_score(
            one_hot_labels[:, i],
            test_probabilities[:, i]
        )

        roc_auc_per_class[
            class_name
        ] = float(auc)

    except ValueError:

        roc_auc_per_class[
            class_name
        ] = None


valid_auc = [
    value
    for value in roc_auc_per_class.values()
    if value is not None
]

roc_auc_macro = (
    float(np.mean(valid_auc))
    if valid_auc
    else None
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    predicted_class,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_path = (
    METRICS_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    predicted_class
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    METRICS_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# UNCERTAINTY
# ============================================================

# Binary entropy uncertainty.
#
# High entropy:
#   probability near 0.5
#
# Low entropy:
#   probability near 0 or 1
#

epsilon = 1e-8

uncertainty = -(
    test_probabilities
    * np.log(
        test_probabilities
        + epsilon
    )
    +
    (1 - test_probabilities)
    * np.log(
        1 - test_probabilities
        + epsilon
    )
)

# Normalize entropy to [0,1]

uncertainty = (
    uncertainty
    / np.log(2.0)
)

mean_uncertainty = (
    uncertainty.mean(axis=1)
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_data = {
    "true_class": [
        CLASS_NAMES[i]
        for i in test_labels
    ],

    "predicted_class": [
        CLASS_NAMES[i]
        for i in predicted_class
    ],

    "max_probability":
        test_probabilities.max(
            axis=1
        ),

    "uncertainty":
        mean_uncertainty,
}


for i, class_name in enumerate(
    CLASS_NAMES
):

    safe_name = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    prediction_data[
        f"{safe_name}_probability"
    ] = test_probabilities[:, i]

    prediction_data[
        f"{safe_name}_uncertainty"
    ] = uncertainty[:, i]

    prediction_data[
        f"{safe_name}_predicted"
    ] = test_binary[:, i]


predictions_df = pd.DataFrame(
    prediction_data
)

predictions_path = (
    TEST_DIR
    / "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE PROBABILITIES
# ============================================================

np.save(
    TEST_DIR
    / "probabilities.npy",
    test_probabilities
)

np.save(
    TEST_DIR
    / "uncertainty.npy",
    uncertainty
)

np.save(
    TEST_DIR
    / "predicted_classes.npy",
    predicted_class
)

np.save(
    TEST_DIR
    / "true_labels.npy",
    test_labels
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {

    "model":
        "MultiLabelClassificationHeads",

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "test_images":
        int(len(test_labels)),

    "threshold":
        THRESHOLD,

    "accuracy":
        float(accuracy),

    "precision_macro":
        float(precision_macro),

    "recall_macro":
        float(recall_macro),

    "f1_macro":
        float(f1_macro),

    "precision_weighted":
        float(precision_weighted),

    "recall_weighted":
        float(recall_weighted),

    "f1_weighted":
        float(f1_weighted),

    "roc_auc_macro":
        roc_auc_macro,

    "roc_auc_per_class":
        roc_auc_per_class,

    "mean_uncertainty":
        float(
            mean_uncertainty.mean()
        ),

    "best_validation_loss":
        float(best_val_loss),

    "best_validation_accuracy":
        float(best_val_accuracy),

    "training_time_seconds":
        float(training_time),

    "checkpoint":
        str(best_checkpoint),
}


with open(
    METRICS_DIR
    / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# PLOT 1: TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(9, 6)
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
    "Multi-Label Classification Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "training_loss.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2: TRAINING ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 6)
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
    "Multi-Label Classification Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "training_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3: CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm
)

plt.title(
    "Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 4: ROC CURVES
# ============================================================

from sklearn.metrics import roc_curve


plt.figure(
    figsize=(9, 7)
)

for i, class_name in enumerate(
    CLASS_NAMES
):

    try:

        fpr, tpr, _ = roc_curve(
            one_hot_labels[:, i],
            test_probabilities[:, i]
        )

        auc_value = (
            roc_auc_per_class[
                class_name
            ]
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={auc_value:.3f})"
            )
        )

    except ValueError:

        pass


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Disease Classification"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "roc_curves.png",
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "model":
        "MultiLabelClassificationHeads",

    "architecture":
        {
            "input_disease_features":
                "(N, 6, 512)",

            "relationship_matrix":
                "(N, 6, 6)",

            "output_logits":
                "(N, 6)",

            "output_probabilities":
                "(N, 6)",

            "activation":
                "Sigmoid",

            "classification_heads":
                6,
        },

    "classes":
        CLASS_NAMES,

    "training":
        {
            "epochs_requested":
                EPOCHS,

            "epochs_completed":
                len(history),

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "best_validation_loss":
                float(best_val_loss),

            "best_validation_accuracy":
                float(best_val_accuracy),

            "training_time_seconds":
                float(training_time),
        },

    "test":
        {
            "images":
                int(len(test_labels)),

            "accuracy":
                float(accuracy),

            "precision_macro":
                float(precision_macro),

            "recall_macro":
                float(recall_macro),

            "f1_macro":
                float(f1_macro),

            "precision_weighted":
                float(precision_weighted),

            "recall_weighted":
                float(recall_weighted),

            "f1_weighted":
                float(f1_weighted),

            "roc_auc_macro":
                roc_auc_macro,

            "mean_uncertainty":
                float(
                    mean_uncertainty.mean()
                ),
        },

    "outputs":
        {
            "checkpoint":
                str(best_checkpoint),

            "predictions":
                str(predictions_path),

            "metrics":
                str(
                    METRICS_DIR
                    / "metrics.json"
                ),

            "plots":
                str(PLOTS_DIR),
        },
}


with open(
    OUTPUT_ROOT
    / "classification_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL PRINT
# ============================================================

print()
print("=" * 70)
print("STEP 7.7 — MULTI-LABEL CLASSIFICATION COMPLETE")
print("=" * 70)

print()

print(
    f"Accuracy           : "
    f"{accuracy:.4f}"
)

print(
    f"Precision (Macro)  : "
    f"{precision_macro:.4f}"
)

print(
    f"Recall (Macro)     : "
    f"{recall_macro:.4f}"
)

print(
    f"F1 Score (Macro)   : "
    f"{f1_macro:.4f}"
)

print(
    f"Precision Weighted : "
    f"{precision_weighted:.4f}"
)

print(
    f"Recall Weighted    : "
    f"{recall_weighted:.4f}"
)

print(
    f"F1 Score Weighted  : "
    f"{f1_weighted:.4f}"
)

if roc_auc_macro is not None:

    print(
        f"ROC-AUC (Macro)    : "
        f"{roc_auc_macro:.4f}"
    )

print(
    f"Mean Uncertainty   : "
    f"{mean_uncertainty.mean():.4f}"
)

print()
print("Best checkpoint:")
print(best_checkpoint)

print()
print("Metrics:")
print(
    METRICS_DIR
    / "metrics.json"
)

print()
print("Predictions:")
print(predictions_path)

print()
print("Plots:")
print(PLOTS_DIR)

print()
print("Summary:")
print(
    OUTPUT_ROOT
    / "classification_summary.json"
)

print()
print("=" * 70)
print("✓ STEP 7.7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Next stage:")
print("STEP 7.8 — FINAL PREDICTION + UNCERTAINTY PIPELINE")