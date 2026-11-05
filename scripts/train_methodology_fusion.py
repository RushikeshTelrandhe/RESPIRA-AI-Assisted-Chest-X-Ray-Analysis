"""
RESPIRA - Methodology Fusion Training Pipeline (Upgraded: Multi-Class Cross-Entropy + Resume)
============================================================================================
Trains the complete end-to-end multi-modal architecture:
  EfficientNet-B0 + ViT-B/16 (1024 tokens) + Bidirectional Cross-Attention
  + Disease-Conditioned Attention + Relational GCN + MC Dropout

Enhancements:
  - Multi-class Label-Smoothed Cross-Entropy Loss (eliminates uncalibrated BCE inter-class confusion)
  - Real-time Top-1 and Top-2 diagnostic accuracy tracking
  - Seamless resume capability (e.g., from epoch 25 to 55)
  - 4x faster evaluation with batched validation/test DataLoaders
  - Calibrated Monte Carlo (MC) Dropout uncertainty quantification
"""

import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.path.abspath("."))
from src.datasets.chestxray_dataset import ChestXrayDataset
from src.fusion.respira_methodology_model import RespiraMethodologyModel, TARGET_CLASSES, NUM_CLASSES


# Default Hyperparameters
PROJECT_ROOT = Path(os.path.abspath("."))
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Lung_Disease_Preprocessed_512"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "methodology_fusion"


def get_training_transforms(image_size: int = 512):
    """Data augmentation for training chest X-rays."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_validation_transforms(image_size: int = 512):
    """Validation and test normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> Tuple[float, float, float]:
    """Executes one epoch of training with AMP, gradient accumulation, and Top-1/Top-2 tracking."""
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top2 = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images)
            logits = out["logits"]
            loss = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        running_loss += loss.item() * grad_accum_steps * batch_size

        preds = torch.argmax(logits, dim=1)
        correct_top1 += (preds == labels).sum().item()

        top2_preds = torch.topk(logits, k=min(2, NUM_CLASSES), dim=1).indices
        correct_top2 += (top2_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += batch_size

    epoch_loss = running_loss / total
    epoch_top1 = correct_top1 / total
    epoch_top2 = correct_top2 / total
    return epoch_loss, epoch_top1, epoch_top2


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluates validation loss, Top-1 accuracy, Top-2 accuracy, and macro F1."""
    model.eval()
    running_loss = 0.0
    correct_top2 = 0
    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images)
            logits = out["logits"]
            loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)

        top2_preds = torch.topk(logits, k=min(2, NUM_CLASSES), dim=1).indices
        correct_top2 += (top2_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    epoch_top1 = accuracy_score(all_targets, all_preds)
    epoch_top2 = correct_top2 / total
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return epoch_loss, epoch_top1, epoch_top2, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train RESPIRA Methodology Fusion Model")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-tuned if not set)")
    parser.add_argument("--epochs", type=int, default=55, help="Total epochs (default: 55)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Stage 1 warmup epochs (default: 3)")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Backbone fine-tuning LR (default: 1e-5)")
    parser.add_argument("--lr_fusion", type=float, default=8e-5, help="Fusion and heads LR (default: 8e-5)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (default: 0 on Windows)")
    parser.add_argument("--mc_samples", type=int, default=15, help="Number of MC Dropout forward samples")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing best_model.pth")
    parser.add_argument("--resume_from", type=str, default=None, help="Specific checkpoint path to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Explicit epoch number to resume from")
    args = parser.parse_args()

    # 1. Device and Auto-Tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("   RESPIRA - METHODOLOGY FUSION TRAINING PIPELINE (RESUME TO 55 EPOCHS)")
    print("=" * 75)
    print(f"Device              : {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model           : {gpu_name}")
        print(f"VRAM Available      : {vram_gb:.2f} GB")

        if args.batch_size is None:
            if vram_gb < 8.0:
                batch_size = 4
                grad_accum_steps = 4
                eval_batch_size = 16  # Faster evaluation on 6GB VRAM
                print("[*] Local GPU <= 8GB detected: Train batch_size=4, grad_accum=4 (Effective=16), Eval batch=16")
            else:
                batch_size = 16
                grad_accum_steps = 1
                eval_batch_size = 32
                print("[*] High-memory GPU detected: Train batch_size=16, grad_accum=1, Eval batch=32")
        else:
            batch_size = args.batch_size
            grad_accum_steps = 16 // batch_size if batch_size < 16 else 1
            eval_batch_size = max(batch_size * 2, 16)
    else:
        batch_size = args.batch_size if args.batch_size else 2
        grad_accum_steps = 8
        eval_batch_size = 4
        print("[!] Running on CPU: Train batch_size=2, grad_accum=8")

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "evaluation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 2. Stratified DataLoaders
    print("\n[1/4] Initializing Stratified DataLoaders...")
    train_dataset = ChestXrayDataset(
        root_dir=args.data_dir, split="train", transform=get_training_transforms()
    )
    val_dataset = ChestXrayDataset(
        root_dir=args.data_dir, split="val", transform=get_validation_transforms()
    )
    test_dataset = ChestXrayDataset(
        root_dir=args.data_dir, split="test", transform=get_validation_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"      Train Samples : {len(train_dataset):,} ({len(train_loader)} batches)")
    print(f"      Val Samples   : {len(val_dataset):,} ({len(val_loader)} batches)")
    print(f"      Test Samples  : {len(test_dataset):,} ({len(test_loader)} batches)")

    # 3. Model Architecture Instantiation
    print("\n[2/4] Building Methodology Model (Dual Backbone + Cross-Attention + GCN)...")
    model = RespiraMethodologyModel(num_classes=NUM_CLASSES, pretrained=True, dropout=0.3)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total Parameters: {total_params:,}")

    # Label-Smoothed Cross-Entropy Loss to resolve mutual-exclusivity inter-class confusion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Check for Resuming Checkpoint
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume or (checkpoints_dir / "best_model.pth").exists():
        resume_path = checkpoints_dir / "best_model.pth"

    start_epoch = 1
    best_val_top1 = 0.0
    best_val_f1 = 0.0
    patience = 12
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_top1": [], "train_top2": [],
        "val_loss": [], "val_top1": [], "val_top2": [], "val_f1": []
    }

    if resume_path and resume_path.exists():
        print(f"\n[*] RESUMING TRAINING from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_epoch = checkpoint.get("epoch", 25)
        start_epoch = args.resume_epoch if args.resume_epoch is not None else (saved_epoch + 1)
        best_val_top1 = checkpoint.get("val_accuracy", 0.0)
        best_val_f1 = checkpoint.get("val_f1", 0.0)
        print(f"      Loaded Checkpoint: Epoch {saved_epoch} (Val Top-1: {best_val_top1*100:.2f}%, F1: {best_val_f1:.4f})")
        print(f"      Resuming execution from Epoch {start_epoch} up to Epoch {args.epochs}...")

        # When resuming, model is already warmed up: unfreeze backbones
        model.unfreeze_backbones()
        param_groups = [
            {"params": model.cnn_backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.vit_backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.cnn_proj.parameters(), "lr": args.lr_fusion},
            {"params": model.vit_proj.parameters(), "lr": args.lr_fusion},
            {"params": model.cross_attention.parameters(), "lr": args.lr_fusion},
            {"params": model.disease_attention.parameters(), "lr": args.lr_fusion},
            {"params": model.gcn_relationship.parameters(), "lr": args.lr_fusion},
            {"params": model.heads.parameters(), "lr": args.lr_fusion},
        ]
        optimizer = AdamW(param_groups, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    else:
        # Fresh start: Stage 1 Warmup
        model.freeze_backbones()
        trainable_stage1 = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_stage1, lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    print("\n[3/4] Commencing Training Loop:")
    print(f"      Target Epochs        : {start_epoch} -> {args.epochs}")
    print(f"      Loss Function        : CrossEntropyLoss (label_smoothing=0.05)")
    print(f"      Backbone Learning Rate: {args.lr_backbone}")
    print(f"      Fusion Learning Rate  : {args.lr_fusion}")
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Switch to Stage 2 if starting fresh and reaching post-warmup
        if not resume_path and epoch == args.warmup_epochs + 1:
            print("\n" + "=" * 75)
            print(">>> TRANSITIONING TO STAGE 2: Full End-to-End Fine-Tuning")
            print("=" * 75)
            model.unfreeze_backbones()
            param_groups = [
                {"params": model.cnn_backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.vit_backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.cnn_proj.parameters(), "lr": args.lr_fusion},
                {"params": model.vit_proj.parameters(), "lr": args.lr_fusion},
                {"params": model.cross_attention.parameters(), "lr": args.lr_fusion},
                {"params": model.disease_attention.parameters(), "lr": args.lr_fusion},
                {"params": model.gcn_relationship.parameters(), "lr": args.lr_fusion},
                {"params": model.heads.parameters(), "lr": args.lr_fusion},
            ]
            optimizer = AdamW(param_groups, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

        # Train & Evaluate
        train_loss, train_top1, train_top2 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps
        )
        val_loss, val_top1, val_top2, val_f1 = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_top1)

        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_top1)
        history["train_top2"].append(train_top2)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top2"].append(val_top2)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Top-1: {train_top1*100:.2f}% - Top-2: {train_top2*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Top-1: {val_top1*100:.2f}% - Top-2: {val_top2*100:.2f}% - F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model Checkpoint
        if val_top1 > best_val_top1 or (val_top1 == best_val_top1 and val_f1 > best_val_f1):
            best_val_top1 = val_top1
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_top1,
                "val_top2": val_top2,
                "val_f1": val_f1,
                "classes": TARGET_CLASSES,
            }, checkpoints_dir / "best_model.pth")
            print(f"      [*] Best model saved! (Val Top-1: {val_top1*100:.2f}%, Top-2: {val_top2*100:.2f}%, F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered after {epoch} epochs without improvement.")
                break

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy": val_top1,
            "val_top2": val_top2,
            "val_f1": val_f1,
            "classes": TARGET_CLASSES,
        }, checkpoints_dir / "last_model.pth")

    # 5. Final Test Set Evaluation with MC Dropout Uncertainty Analysis
    print("\n[4/4] Executing Test Set Evaluation with MC Dropout Uncertainty...")
    best_checkpoint = torch.load(checkpoints_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_predictions = []
    test_uncertainties = []
    test_confidences = []
    test_probabilities = []
    test_targets = []
    test_top2_matches = 0

    t0 = time.time()
    for images, labels in test_loader:
        images = images.to(device)
        mc_results = model.predict_with_mc_dropout(images, num_samples=args.mc_samples)

        mean_probs = mc_results["mean_probabilities"]
        preds = mc_results["predicted_class"].cpu().numpy()
        unc = mc_results["uncertainty_entropy"].cpu().numpy()
        conf = mc_results["confidence_score"].cpu().numpy()
        probs_np = mean_probs.cpu().numpy()
        lbls_np = labels.numpy()

        # Top-2 calculation
        top2_indices = np.argsort(probs_np, axis=1)[:, -2:]
        for true_idx, top2_pair in zip(lbls_np, top2_indices):
            if true_idx in top2_pair:
                test_top2_matches += 1

        test_predictions.extend(preds)
        test_uncertainties.extend(unc)
        test_confidences.extend(conf)
        test_probabilities.extend(probs_np)
        test_targets.extend(lbls_np)

    test_duration = time.time() - t0
    total_test_samples = len(test_targets)
    test_top1_acc = accuracy_score(test_targets, test_predictions)
    test_top2_acc = test_top2_matches / total_test_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_predictions, average="macro", zero_division=0
    )

    # Multi-class ROC-AUC (One-vs-Rest)
    all_probs_np = np.array(test_probabilities)
    targets_oh = np.eye(NUM_CLASSES)[test_targets]
    try:
        macro_roc_auc = float(roc_auc_score(targets_oh, all_probs_np, multi_class="ovr", average="macro"))
    except Exception:
        macro_roc_auc = 0.0

    print("\n" + "=" * 70)
    print("                     FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Test Top-1 Accuracy   : {test_top1_acc * 100:.2f}%")
    print(f"Test Top-2 Accuracy   : {test_top2_acc * 100:.2f}%")
    print(f"Macro Precision       : {precision:.4f}")
    print(f"Macro Recall          : {recall:.4f}")
    print(f"Macro F1 Score        : {f1:.4f}")
    print(f"Macro ROC-AUC         : {macro_roc_auc:.4f}")
    print(f"Mean Predictive Conf. : {np.mean(test_confidences):.4f}")
    print(f"Mean MC Uncertainty   : {np.mean(test_uncertainties):.4f}")
    print(f"Inference Time ({total_test_samples} samples) : {test_duration:.2f}s")
    print("=" * 70)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(test_targets, test_predictions, target_names=TARGET_CLASSES, digits=4))

    # Save Metrics JSON
    metrics = {
        "test_top1_accuracy": float(test_top1_acc),
        "test_top1_accuracy_percent": float(test_top1_acc * 100),
        "test_top2_accuracy": float(test_top2_acc),
        "test_top2_accuracy_percent": float(test_top2_acc * 100),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_roc_auc": float(macro_roc_auc),
        "mean_confidence": float(np.mean(test_confidences)),
        "mean_uncertainty": float(np.mean(test_uncertainties)),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_accuracy": float(best_checkpoint["val_accuracy"]),
        "best_val_top2": float(best_checkpoint.get("val_top2", 0.0)),
        "best_val_f1": float(best_checkpoint["val_f1"]),
        "classes": TARGET_CLASSES,
    }
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save Test Predictions CSV
    preds_df = pd.DataFrame({
        "true_label": test_targets,
        "true_class": [TARGET_CLASSES[t] for t in test_targets],
        "predicted_label": test_predictions,
        "predicted_class": [TARGET_CLASSES[p] for p in test_predictions],
        "is_correct_top1": [t == p for t, p in zip(test_targets, test_predictions)],
        "confidence_score": test_confidences,
        "mc_uncertainty": test_uncertainties,
    })
    for i, c in enumerate(TARGET_CLASSES):
        preds_df[f"prob_{c}"] = all_probs_np[:, i]
    preds_df.to_csv(eval_dir / "test_predictions.csv", index=False)

    # Plot Convergence Curves
    if len(history["train_loss"]) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(start_epoch, start_epoch + len(history["train_loss"]))
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="orange")
        ax1.set_title("Cross-Entropy Loss Progression")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, [a * 100 for a in history["train_top1"]], label="Train Top-1 (%)", color="blue")
        ax2.plot(epochs_range, [a * 100 for a in history["val_top1"]], label="Val Top-1 (%)", color="orange")
        ax2.plot(epochs_range, [a * 100 for a in history["val_top2"]], label="Val Top-2 (%)", color="purple", linestyle="--")
        ax2.plot(epochs_range, [f * 100 for f in history["val_f1"]], label="Val Macro F1 (%)", color="green")
        ax2.set_title("Accuracy & F1 Progression")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Percentage (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(eval_dir / "training_curves_resumed.png", dpi=200)
        plt.close()

    print(f"\nTraining outputs and evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
