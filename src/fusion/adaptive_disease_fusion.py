# ============================================================
# Respira - STEP 7.5
# Adaptive Disease-Specific Fusion
# ============================================================
#
# Input:
#   outputs/fusion/disease_attention/
#
#   disease_features.pt
#       Shape: (N, 6, 512)
#
# Output:
#   outputs/fusion/adaptive_fusion/
#
#   checkpoints/best_model.pth
#   logs/training_history.csv
#   metrics/training_summary.json
#   plots/*.png
#
#   train/
#       fused_features.pt
#       disease_weights.pt
#       labels.npy
#       metadata.csv
#
#   val/
#       ...
#
#   test/
#       ...
#
# Final representation:
#   fused_features: (N, 512)
#
# Disease weights:
#   disease_weights: (N, 6)
#
# ============================================================

from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "disease_attention"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "adaptive_fusion"
)

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

NUM_CLASSES = 6
FEATURE_DIM = 512

BATCH_SIZE = 128
EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
DROPOUT = 0.2

PATIENCE = 7

NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        OUTPUT_ROOT,
        CHECKPOINT_DIR,
        LOG_DIR,
        METRICS_DIR,
        PLOTS_DIR,
    ]

    for split in ["train", "val", "test"]:

        directories.append(
            OUTPUT_ROOT / split
        )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# DATASET
# ============================================================

class DiseaseFeatureDataset(Dataset):

    def __init__(
        self,
        split: str,
    ):

        self.split = split

        split_dir = INPUT_ROOT / split

        feature_path = (
            split_dir /
            "disease_features.pt"
        )

        label_path = (
            split_dir /
            "labels.npy"
        )

        metadata_path = (
            split_dir /
            "metadata.csv"
        )

        if not feature_path.exists():

            raise FileNotFoundError(
                f"\nDisease feature file not found:\n"
                f"{feature_path}\n\n"
                f"Run STEP 7.4 first."
            )

        if not label_path.exists():

            raise FileNotFoundError(
                f"\nLabels not found:\n"
                f"{label_path}"
            )

        # ----------------------------------------------------
        # Load disease features
        # ----------------------------------------------------

        self.features = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True
        )

        # ----------------------------------------------------
        # Load labels
        # ----------------------------------------------------

        self.labels = np.load(
            label_path
        )

        self.labels = torch.from_numpy(
            self.labels
        ).long()

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.metadata = None

        if metadata_path.exists():

            self.metadata = pd.read_csv(
                metadata_path
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if self.features.ndim != 3:

            raise ValueError(
                f"{split}: expected 3D disease features, "
                f"got {self.features.shape}"
            )

        if self.features.shape[1] != NUM_CLASSES:

            raise ValueError(
                f"{split}: expected {NUM_CLASSES} "
                f"disease representations, "
                f"got {self.features.shape[1]}"
            )

        if self.features.shape[2] != FEATURE_DIM:

            raise ValueError(
                f"{split}: expected feature dimension "
                f"{FEATURE_DIM}, "
                f"got {self.features.shape[2]}"
            )

        if len(self.features) != len(self.labels):

            raise ValueError(
                f"{split}: feature/label count mismatch"
            )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not torch.isfinite(
            self.features
        ).all():

            raise ValueError(
                f"{split}: NaN/Inf detected"
            )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index],
        )


# ============================================================
# ADAPTIVE DISEASE FUSION MODEL
# ============================================================

class AdaptiveDiseaseFusion(nn.Module):

    """
    Adaptive Disease-Specific Fusion Network.

    Input:
        (B, 6, 512)

    Produces:

        disease_weights:
            (B, 6)

        fused_features:
            (B, 512)

        logits:
            (B, 6)

    The fusion weights are learned dynamically
    for every image.
    """

    def __init__(
        self,
        feature_dim=512,
        num_classes=6,
        hidden_dim=256,
        dropout=0.2,
    ):

        super().__init__()

        self.feature_dim = feature_dim

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Disease importance network
        # ----------------------------------------------------

        self.weight_network = nn.Sequential(

            nn.LayerNorm(feature_dim),

            nn.Linear(
                feature_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                1
            )
        )

        # ----------------------------------------------------
        # Feature transformation
        # ----------------------------------------------------

        self.feature_transform = nn.Sequential(

            nn.LayerNorm(feature_dim),

            nn.Linear(
                feature_dim,
                feature_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(feature_dim),

            nn.Linear(
                feature_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # x:
        # (B, 6, 512)
        # ----------------------------------------------------

        # Calculate disease-specific importance
        weight_logits = self.weight_network(x)

        # (B, 6, 1) -> (B, 6)
        weight_logits = weight_logits.squeeze(-1)

        # Normalize across diseases
        disease_weights = torch.softmax(
            weight_logits,
            dim=1
        )

        # ----------------------------------------------------
        # Transform each disease representation
        # ----------------------------------------------------

        transformed = self.feature_transform(
            x
        )

        # ----------------------------------------------------
        # Adaptive weighted fusion
        # ----------------------------------------------------

        weighted = (
            transformed *
            disease_weights.unsqueeze(-1)
        )

        # Sum six disease representations
        fused_features = weighted.sum(
            dim=1
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return {
            "fused_features": fused_features,
            "disease_weights": disease_weights,
            "logits": logits,
        }


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(
    logits,
    labels
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    return correct / total


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for features, labels in progress:

        features = features.to(
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

        output = model(
            features
        )

        logits = output["logits"]

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:

        features = features.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        output = model(
            features
        )

        logits = output["logits"]

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    history_path = (
        LOG_DIR /
        "training_history.csv"
    )

    df = pd.DataFrame(
        history
    )

    df.to_csv(
        history_path,
        index=False
    )

    return history_path


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_loss(history):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Adaptive Disease Fusion - Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR /
        "loss_curve.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT ACCURACY
# ============================================================

def plot_accuracy(history):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Adaptive Disease Fusion - Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR /
        "accuracy_curve.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# LOAD DATA
# ============================================================

def create_loaders():

    train_dataset = DiseaseFeatureDataset(
        "train"
    )

    val_dataset = DiseaseFeatureDataset(
        "val"
    )

    test_dataset = DiseaseFeatureDataset(
        "test"
    )

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

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# GENERATE FINAL FEATURES
# ============================================================

@torch.no_grad()
def generate_features(
    model,
    dataset,
    split,
):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_fused = []
    all_weights = []

    print()
    print("=" * 70)
    print(
        f"GENERATING ADAPTIVE FUSION FEATURES — "
        f"{split.upper()}"
    )
    print("=" * 70)

    for features, labels in tqdm(
        loader,
        desc=f"Extracting {split}"
    ):

        features = features.to(
            DEVICE,
            non_blocking=True
        )

        output = model(
            features
        )

        fused = (
            output["fused_features"]
            .cpu()
        )

        weights = (
            output["disease_weights"]
            .cpu()
        )

        all_fused.append(
            fused
        )

        all_weights.append(
            weights
        )

    fused_features = torch.cat(
        all_fused,
        dim=0
    )

    disease_weights = torch.cat(
        all_weights,
        dim=0
    )

    labels = dataset.labels

    print()
    print("Feature shapes:")

    print(
        f"  Fused features : "
        f"{tuple(fused_features.shape)}"
    )

    print(
        f"  Disease weights: "
        f"{tuple(disease_weights.shape)}"
    )

    print(
        f"  Labels         : "
        f"{tuple(labels.shape)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not torch.isfinite(
        fused_features
    ).all():

        raise ValueError(
            f"{split}: fused features contain NaN/Inf"
        )

    if not torch.isfinite(
        disease_weights
    ).all():

        raise ValueError(
            f"{split}: disease weights contain NaN/Inf"
        )

    # --------------------------------------------------------
    # Validate weight normalization
    # --------------------------------------------------------

    weight_sum = (
        disease_weights.sum(
            dim=1
        )
    )

    max_error = torch.abs(
        weight_sum - 1.0
    ).max().item()

    print(
        f"\nMaximum weight normalization error: "
        f"{max_error:.8f}"
    )

    if max_error > 1e-5:

        raise ValueError(
            f"{split}: disease weights are not normalized"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    split_dir = (
        OUTPUT_ROOT /
        split
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fused_path = (
        split_dir /
        "fused_features.pt"
    )

    weights_path = (
        split_dir /
        "disease_weights.pt"
    )

    labels_path = (
        split_dir /
        "labels.npy"
    )

    torch.save(
        fused_features,
        fused_path
    )

    torch.save(
        disease_weights,
        weights_path
    )

    np.save(
        labels_path,
        labels.numpy()
    )

    print(
        f"\n✓ Saved fused features:"
    )

    print(
        f"  {fused_path}"
    )

    print(
        f"\n✓ Saved disease weights:"
    )

    print(
        f"  {weights_path}"
    )

    print(
        f"\n✓ Saved labels:"
    )

    print(
        f"  {labels_path}"
    )

    # --------------------------------------------------------
    # Copy metadata
    # --------------------------------------------------------

    if dataset.metadata is not None:

        metadata_path = (
            split_dir /
            "metadata.csv"
        )

        dataset.metadata.to_csv(
            metadata_path,
            index=False
        )

        print(
            f"\n✓ Saved metadata:"
        )

        print(
            f"  {metadata_path}"
        )

    # --------------------------------------------------------
    # Disease weight statistics
    # --------------------------------------------------------

    mean_weights = (
        disease_weights.mean(
            dim=0
        )
    )

    print(
        "\nAverage adaptive disease weights:"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {i}: "
            f"{name:<22} "
            f"{mean_weights[i].item():.4f}"
        )

    return {
        "samples": len(labels),
        "fused_shape": list(
            fused_features.shape
        ),
        "weights_shape": list(
            disease_weights.shape
        ),
        "max_weight_normalization_error":
            max_error,
        "mean_disease_weights": {
            CLASS_NAMES[i]:
                float(mean_weights[i])
            for i in range(NUM_CLASSES)
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    create_directories()

    print("=" * 70)
    print(
        "RESPIRA — STEP 7.5"
    )

    print(
        "ADAPTIVE DISEASE-SPECIFIC FUSION"
    )

    print("=" * 70)

    print()

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print()

    print(
        f"Input:\n"
        f"{INPUT_ROOT}"
    )

    print()

    print(
        f"Output:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING DISEASE-CONDITIONED FEATURES")
    print("=" * 70)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders()

    print()

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = AdaptiveDiseaseFusion(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    model = model.to(
        DEVICE
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 70)
    print("ADAPTIVE FUSION MODEL")
    print("=" * 70)

    print(
        f"\nFeature dimension: "
        f"{FEATURE_DIM}"
    )

    print(
        f"Disease representations: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Hidden dimension: "
        f"{HIDDEN_DIM}"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING ADAPTIVE FUSION")
    print("=" * 70)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_time_seconds": [],
    }

    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    start_training = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        print()
        print("=" * 70)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("=" * 70)

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
            )
        )

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

        print()

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

        history["epoch_time_seconds"].append(
            epoch_time
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint_path = (
                CHECKPOINT_DIR /
                "best_model.pth"
            )

            torch.save(
                {
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
                    "feature_dim":
                        FEATURE_DIM,
                    "hidden_dim":
                        HIDDEN_DIM,
                    "dropout":
                        DROPOUT,
                    "seed":
                        SEED,
                    "model_name":
                        "AdaptiveDiseaseFusion",
                },
                checkpoint_path,
            )

            print()

            print(
                "✓ New best checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                f"Early stopping triggered "
                f"after {PATIENCE} epochs "
                f"without validation improvement."
            )

            break

    training_time = (
        time.time()
        - start_training
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = save_history(
        history
    )

    plot_loss(
        history
    )

    plot_accuracy(
        history
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint_path = (
        CHECKPOINT_DIR /
        "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print("=" * 70)
    print("BEST ADAPTIVE FUSION MODEL LOADED")
    print("=" * 70)

    print(
        f"\nBest epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )

    # ========================================================
    # GENERATE FEATURES
    # ========================================================

    split_summaries = {}

    split_summaries["train"] = (
        generate_features(
            model,
            train_dataset,
            "train",
        )
    )

    split_summaries["val"] = (
        generate_features(
            model,
            val_dataset,
            "val",
        )
    )

    split_summaries["test"] = (
        generate_features(
            model,
            test_dataset,
            "test",
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "model":
            "AdaptiveDiseaseFusion",

        "stage":
            "STEP 7.5",

        "input_representation":
            "(N, 6, 512)",

        "output_representation":
            "(N, 512)",

        "disease_weights":
            "(N, 6)",

        "num_classes":
            NUM_CLASSES,

        "classes":
            CLASS_NAMES,

        "feature_dim":
            FEATURE_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "batch_size":
            BATCH_SIZE,

        "epochs_requested":
            EPOCHS,

        "epochs_completed":
            len(history["epoch"]),

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,

        "best_validation_accuracy":
            best_val_accuracy,

        "training_time_seconds":
            training_time,

        "device":
            str(DEVICE),

        "checkpoint":
            str(checkpoint_path),

        "training_history":
            str(history_path),

        "splits":
            split_summaries,
    }

    summary_path = (
        OUTPUT_ROOT /
        "fusion_summary.json"
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

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEP 7.5 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()

    print(
        "Adaptive fusion outputs:"
    )

    print(
        OUTPUT_ROOT
    )

    print()

    print(
        "Final representations:"
    )

    print(
        "  Fused features : "
        "(N, 512)"
    )

    print(
        "  Disease weights: "
        "(N, 6)"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()

    print(
        "Training summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        "Plots:"
    )

    print(
        PLOTS_DIR
    )

    print()

    print(
        "✓ Train fusion features generated."
    )

    print(
        "✓ Validation fusion features generated."
    )

    print(
        "✓ Test fusion features generated."
    )

    print(
        "✓ Adaptive disease weights generated."
    )

    print(
        "✓ Best model checkpoint saved."
    )

    print()

    print(
        "Next stage:"
    )

    print(
        "STEP 7.6 — DISEASE RELATIONSHIP MODELING"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()