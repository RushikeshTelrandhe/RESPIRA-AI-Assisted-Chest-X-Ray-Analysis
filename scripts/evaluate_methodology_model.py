"""
RESPIRA - Methodology Model Standalone Evaluation Pipeline
==========================================================
Evaluates the trained dual-backbone fusion model:
  EfficientNet-B0 + ViT-B/16 (1024 tokens) + Bidirectional Cross-Attention
  + Disease-Conditioned Attention + Relational GCN + MC Dropout

Outputs:
  - Top-1 and Top-2 Accuracy
  - Macro Precision, Recall, F1, and multi-class ROC-AUC
  - Grouped Clinical Accuracy (Pneumonia subtyping & Opacities)
  - Confusion Matrix plot (normalized & raw)
  - Multi-class ROC curves plot
  - Calibrated MC Dropout Uncertainty analysis (entropy vs accuracy)
  - Predictions CSV and metrics JSON
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
from torchvision import transforms

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.path.abspath("."))
from src.datasets.chestxray_dataset import ChestXrayDataset
from src.fusion.respira_methodology_model import RespiraMethodologyModel, TARGET_CLASSES, NUM_CLASSES


def get_validation_transforms(image_size: int = 512):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path):
    """Plots and saves raw and normalized confusion matrices side-by-side."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold" if i == j else "normal"
            )
    ax1.set_ylabel("True Disease Label", fontsize=11)
    ax1.set_xlabel("Predicted Label", fontsize=11)

    # Normalized percentages
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens)
    ax2.set_title("Normalized Confusion Matrix (Recall %)", fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(classes, fontsize=9)

    thresh_norm = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(
                j, i, f"{cm_norm[i, j]*100:.1f}%",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh_norm else "black",
                fontweight="bold" if i == j else "normal"
            )
    ax2.set_ylabel("True Disease Label", fontsize=11)
    ax2.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: List[str], save_path: Path):
    """Plots multi-class ROC curves (One-vs-Rest)."""
    plt.figure(figsize=(9, 7))
    y_true_oh = np.eye(len(classes))[y_true]

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Multi-Class ROC Curves (Methodology Model)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_uncertainty_analysis(
    is_correct: List[bool], uncertainties: List[float], confidences: List[float], save_path: Path
):
    """Plots uncertainty and confidence distributions for correct vs incorrect predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    corr_unc = [u for u, c in zip(uncertainties, is_correct) if c]
    incorr_unc = [u for u, c in zip(uncertainties, is_correct) if not c]

    ax1.hist(corr_unc, bins=25, alpha=0.6, color="green", label=f"Correct (Mean={np.mean(corr_unc):.3f})", density=True)
    ax1.hist(incorr_unc, bins=25, alpha=0.6, color="red", label=f"Incorrect (Mean={np.mean(incorr_unc):.3f})", density=True)
    ax1.set_title("MC Uncertainty (Normalized Shannon Entropy)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Entropy [0 = Certain, 1 = Maximum Uncertainty]")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    corr_conf = [p for p, c in zip(confidences, is_correct) if c]
    incorr_conf = [p for p, c in zip(confidences, is_correct) if not c]

    ax2.hist(corr_conf, bins=25, alpha=0.6, color="green", label=f"Correct (Mean={np.mean(corr_conf):.3f})", density=True)
    ax2.hist(incorr_conf, bins=25, alpha=0.6, color="red", label=f"Incorrect (Mean={np.mean(incorr_conf):.3f})", density=True)
    ax2.set_title("Predictive Confidence Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Confidence [Max Probability]")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RESPIRA Methodology Fusion Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/methodology_fusion/checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/Lung_Disease_Preprocessed_512",
        help="Path to preprocessed dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/methodology_fusion/evaluation",
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=10,
        help="MC Dropout forward samples (set to 0 for deterministic fast inference)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 on Windows)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      RESPIRA - METHODOLOGY MODEL EVALUATION PIPELINE")
    print("=" * 75)
    print(f"Device              : {device}")
    if device.type == "cuda":
        print(f"GPU Model           : {torch.cuda.get_device_name(0)}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[!] Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n[1/3] Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_epoch = checkpoint.get("epoch", "N/A")
    val_acc = checkpoint.get("val_accuracy", "N/A")
    val_top2 = checkpoint.get("val_top2", "N/A")
    val_f1 = checkpoint.get("val_f1", "N/A")
    print(f"      Trained Epoch : {saved_epoch}")
    if val_acc != "N/A":
        print(f"      Best Val Top-1: {float(val_acc)*100:.2f}%")
    if val_top2 != "N/A":
        print(f"      Best Val Top-2: {float(val_top2)*100:.2f}%")
    if val_f1 != "N/A":
        print(f"      Best Val F1   : {float(val_f1):.4f}")

    # Build model and load weights
    model = RespiraMethodologyModel(num_classes=NUM_CLASSES, pretrained=False, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # DataLoader
    print(f"\n[2/3] Loading '{args.split}' Dataset ({args.data_dir})...")
    dataset = ChestXrayDataset(
        root_dir=args.data_dir, split=args.split, transform=get_validation_transforms()
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"      Total Samples : {len(dataset):,} ({len(loader)} batches)")

    # Inference Loop
    use_mc = args.mc_samples > 0
    print(f"\n[3/3] Running Inference (Mode: {'MC Dropout with ' + str(args.mc_samples) + ' samples' if use_mc else 'Deterministic'})...")
    all_preds = []
    all_targets = []
    all_probs = []
    all_confidences = []
    all_uncertainties = []
    top2_matches = 0

    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)

        if use_mc:
            mc_res = model.predict_with_mc_dropout(images, num_samples=args.mc_samples)
            probs_np = mc_res["mean_probabilities"].cpu().numpy()
            preds_np = mc_res["predicted_class"].cpu().numpy()
            conf_np = mc_res["confidence_score"].cpu().numpy()
            unc_np = mc_res["uncertainty_entropy"].cpu().numpy()
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(images)["logits"]
                    probs = torch.softmax(logits, dim=-1)
                    probs_np = probs.cpu().numpy()
                    preds_np = np.argmax(probs_np, axis=1)
                    conf_np = np.max(probs_np, axis=1)
                    # Normalized entropy
                    p = np.clip(probs_np, 1e-7, 1.0)
                    unc_np = -np.sum(p * np.log2(p), axis=1) / np.log2(NUM_CLASSES)

        lbls_np = labels.numpy()

        # Top-2 calculation
        top2_indices = np.argsort(probs_np, axis=1)[:, -2:]
        for true_idx, top2_pair in zip(lbls_np, top2_indices):
            if true_idx in top2_pair:
                top2_matches += 1

        all_preds.extend(preds_np)
        all_targets.extend(lbls_np)
        all_probs.extend(probs_np)
        all_confidences.extend(conf_np)
        all_uncertainties.extend(unc_np)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            print(f"      Processed {batch_idx + 1}/{len(loader)} batches ({(batch_idx + 1)*args.batch_size} samples)...")

    duration = time.time() - t0
    total_samples = len(all_targets)
    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)

    # 1. Standard Metrics
    top1_acc = accuracy_score(all_targets_arr, all_preds_arr)
    top2_acc = top2_matches / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets_arr, all_preds_arr, average="macro", zero_division=0
    )

    # Multi-class ROC-AUC (OVR)
    targets_oh = np.eye(NUM_CLASSES)[all_targets_arr]
    try:
        macro_roc_auc = float(roc_auc_score(targets_oh, all_probs_arr, multi_class="ovr", average="macro"))
    except Exception:
        macro_roc_auc = 0.0

    # 2. Grouped Clinical Diagnostics
    # Group A: Bacterial + Viral -> Pneumonia
    pred_names = [TARGET_CLASSES[p] for p in all_preds_arr]
    true_names = [TARGET_CLASSES[t] for t in all_targets_arr]
    grouped_pneu_true = ["Pneumonia" if "Pneumonia" in n else n for n in true_names]
    grouped_pneu_pred = ["Pneumonia" if "Pneumonia" in n else n for n in pred_names]
    pneu_grouped_acc = accuracy_score(grouped_pneu_true, grouped_pneu_pred)

    # Group B: Atelectasis + Pulmonary Edema -> Lung Opacities
    grouped_all_true = [
        "Opacities" if n in ["Atelectasis", "Pulmonary Edema"] else n for n in grouped_pneu_true
    ]
    grouped_all_pred = [
        "Opacities" if n in ["Atelectasis", "Pulmonary Edema"] else n for n in grouped_pneu_pred
    ]
    clinical_grouped_acc = accuracy_score(grouped_all_true, grouped_all_pred)

    # Print Final Summary
    print("\n" + "=" * 75)
    print(f"            FINAL EVALUATION RESULTS ({args.split.upper()} SET - {total_samples} SAMPLES)")
    print("=" * 75)
    print(f"Top-1 Accuracy                 : {top1_acc * 100:.2f}%")
    print(f"Top-2 Accuracy                 : {top2_acc * 100:.2f}%")
    print(f"Pneumonia Grouped Accuracy     : {pneu_grouped_acc * 100:.2f}% (Bacterial + Viral unified)")
    print(f"Clinical Grouped Accuracy      : {clinical_grouped_acc * 100:.2f}% (Pneumonias & Opacities unified)")
    print(f"Macro Precision                : {precision:.4f}")
    print(f"Macro Recall                   : {recall:.4f}")
    print(f"Macro F1-Score                 : {f1:.4f}")
    print(f"Macro ROC-AUC (OVR)            : {macro_roc_auc:.4f}")
    print(f"Mean Predictive Confidence     : {np.mean(all_confidences):.4f}")
    print(f"Mean MC Uncertainty (Entropy)  : {np.mean(all_uncertainties):.4f}")
    print(f"Inference Time                 : {duration:.2f}s ({total_samples / duration:.1f} samples/sec)")
    print("=" * 75)

    print("\nPer-Class Classification Report:")
    print(classification_report(all_targets_arr, all_preds_arr, target_names=TARGET_CLASSES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_targets_arr, all_preds_arr)
    print("\nRaw Confusion Matrix (Counts):")
    print(cm)

    # Save Plots
    print("\nSaving evaluation plots and files...")
    plot_confusion_matrix(cm, TARGET_CLASSES, output_dir / "confusion_matrix.png")
    plot_roc_curves(all_targets_arr, all_probs_arr, TARGET_CLASSES, output_dir / "roc_curves.png")
    is_correct = [t == p for t, p in zip(all_targets_arr, all_preds_arr)]
    plot_uncertainty_analysis(is_correct, all_uncertainties, all_confidences, output_dir / "uncertainty_distribution.png")

    # Save Predictions CSV
    df = pd.DataFrame({
        "sample_index": range(total_samples),
        "true_label": all_targets_arr,
        "true_class": true_names,
        "predicted_label": all_preds_arr,
        "predicted_class": pred_names,
        "is_correct_top1": is_correct,
        "confidence_score": all_confidences,
        "mc_uncertainty": all_uncertainties,
    })
    for i, c in enumerate(TARGET_CLASSES):
        df[f"prob_{c}"] = all_probs_arr[:, i]
    df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Save Metrics JSON
    metrics = {
        "split": args.split,
        "total_samples": total_samples,
        "top1_accuracy": float(top1_acc),
        "top1_accuracy_percent": float(top1_acc * 100),
        "top2_accuracy": float(top2_acc),
        "top2_accuracy_percent": float(top2_acc * 100),
        "pneumonia_grouped_accuracy_percent": float(pneu_grouped_acc * 100),
        "clinical_grouped_accuracy_percent": float(clinical_grouped_acc * 100),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_roc_auc": float(macro_roc_auc),
        "mean_confidence": float(np.mean(all_confidences)),
        "mean_uncertainty": float(np.mean(all_uncertainties)),
        "checkpoint_epoch": int(saved_epoch) if saved_epoch != "N/A" else None,
        "classes": TARGET_CLASSES,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"[*] Complete evaluation artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
