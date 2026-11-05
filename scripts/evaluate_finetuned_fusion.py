"""
RESPIRA - Methodology Fusion Model Evaluation Pipeline
======================================================
Comprehensive evaluation suite for the Finetuned Respira Fusion model:
  Dual-Backbone (EfficientNet-B0 + ViT-B/16) + Bidirectional Cross-Attention
  + Disease-Conditioned Attention + Relational GCN + MC Dropout

Usage:
------
# 1. Standard fast evaluation (deterministic, ~1-2 mins on GPU):
python scripts/evaluate_finetuned_fusion.py

# 2. Monte Carlo Dropout evaluation (10 forward passes for calibrated uncertainty):
python scripts/evaluate_finetuned_fusion.py --mc_samples 10

# 3. Custom output directory or batch size:
python scripts/evaluate_finetuned_fusion.py --output_dir evaluation_results --batch_size 32
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
    """Standard ImageNet validation transforms matching 512x512 training."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_default_checkpoint() -> str:
    """Finds the best trained checkpoint from standard candidate locations."""
    candidates = [
        Path("models/methodology_fusion/best_model.pth"),
        Path("outputs/methodology_fusion/checkpoints/best_model.pth"),
        Path("outputs/methodology_fusion/checkpoints/last_model.pth"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path):
    """Plots and saves raw counts and normalized percentage confusion matrices side-by-side."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Raw counts
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
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
    ax1.set_ylabel("True Ground-Truth Diagnosis", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Model Predicted Diagnosis", fontsize=11, fontweight="bold")

    # 2. Normalized percentages
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens)
    ax2.set_title("Normalized Confusion Matrix (Recall / Sensitivity %)", fontsize=13, fontweight="bold")
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
    ax2.set_ylabel("True Ground-Truth Diagnosis", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Model Predicted Diagnosis", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: List[str], save_path: Path):
    """Plots multi-class ROC curves (One-vs-Rest) with AUC values."""
    plt.figure(figsize=(9, 7))
    y_true_oh = np.eye(len(classes))[y_true]

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls_name:20s} (AUC = {roc_auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Multi-Class ROC Curves (Finetuned Respira Fusion)", fontsize=13, fontweight="bold")
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
    ax1.set_title("Calibrated Uncertainty (Normalized Entropy)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Entropy [0 = Certain, 1 = Maximum Uncertainty]")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    corr_conf = [p for p, c in zip(confidences, is_correct) if c]
    incorr_conf = [p for p, c in zip(confidences, is_correct) if not c]

    ax2.hist(corr_conf, bins=25, alpha=0.6, color="green", label=f"Correct (Mean={np.mean(corr_conf):.3f})", density=True)
    ax2.hist(incorr_conf, bins=25, alpha=0.6, color="red", label=f"Incorrect (Mean={np.mean(incorr_conf):.3f})", density=True)
    ax2.set_title("Predictive Softmax Confidence Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Confidence Score [Max Probability]")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Finetuned Respira Fusion Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=find_default_checkpoint(),
        help="Path to trained model checkpoint (*.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/Lung_Disease_Preprocessed_512",
        help="Path to preprocessed dataset root directory",
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
        default="evaluation_results",
        help="Directory to save evaluation reports and plots",
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
        default=0,
        help="Monte Carlo Dropout samples (0 = fast deterministic inference; >0 for MC sampling)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for Windows)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      RESPIRA - FINETUNED FUSION MODEL STANDALONE EVALUATION")
    print("=" * 80)
    print(f"Device              : {device}")
    if device.type == "cuda":
        print(f"GPU Model           : {torch.cuda.get_device_name(0)}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[!] Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n[1/3] Loading Checkpoint: {ckpt_path.resolve()}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_epoch = checkpoint.get("epoch", "N/A")
    val_acc = checkpoint.get("val_accuracy", "N/A")
    val_top2 = checkpoint.get("val_top2", "N/A")
    val_f1 = checkpoint.get("val_f1", "N/A")
    print(f"      Checkpoint Epoch : {saved_epoch}")
    if val_acc != "N/A":
        print(f"      Best Val Top-1   : {float(val_acc)*100:.2f}%")
    if val_top2 != "N/A":
        print(f"      Best Val Top-2   : {float(val_top2)*100:.2f}%")
    if val_f1 != "N/A":
        print(f"      Best Val F1      : {float(val_f1):.4f}")

    # Build model and load state dict
    model = RespiraMethodologyModel(num_classes=NUM_CLASSES, pretrained=False, dropout=0.3)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    # DataLoader
    print(f"\n[2/3] Loading '{args.split}' Dataset ({args.data_dir})...")
    dataset = ChestXrayDataset(
        root_dir=args.data_dir, split=args.split, transform=get_validation_transforms(512)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"      Total Samples    : {len(dataset):,} ({len(loader)} batches)")

    # Inference Loop
    use_mc = args.mc_samples > 0
    mode_str = f"Monte Carlo Dropout ({args.mc_samples} stochastic forward passes)" if use_mc else "Deterministic (Fast single forward pass)"
    print(f"\n[3/3] Running Inference...")
    print(f"      Mode             : {mode_str}")

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
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(images)["logits"]
                    probs = torch.softmax(logits, dim=-1)
                    probs_np = probs.cpu().numpy()
                    preds_np = np.argmax(probs_np, axis=1)
                    conf_np = np.max(probs_np, axis=1)
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

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
            print(f"      Processed {batch_idx + 1}/{len(loader)} batches ({(batch_idx + 1)*args.batch_size if (batch_idx + 1) < len(loader) else len(dataset)} samples)...")

    duration = time.time() - t0
    total_samples = len(all_targets)
    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)

    # 1. Overall Metrics
    top1_acc = accuracy_score(all_targets_arr, all_preds_arr)
    top2_acc = top2_matches / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets_arr, all_preds_arr, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets_arr, all_preds_arr, average="weighted", zero_division=0
    )

    targets_oh = np.eye(NUM_CLASSES)[all_targets_arr]
    try:
        macro_roc_auc = float(roc_auc_score(targets_oh, all_probs_arr, multi_class="ovr", average="macro"))
    except Exception:
        macro_roc_auc = 0.0

    # 2. Grouped Clinical Diagnostics
    pred_names = [TARGET_CLASSES[p] for p in all_preds_arr]
    true_names = [TARGET_CLASSES[t] for t in all_targets_arr]

    grouped_pneu_true = ["Pneumonia" if "Pneumonia" in n else n for n in true_names]
    grouped_pneu_pred = ["Pneumonia" if "Pneumonia" in n else n for n in pred_names]
    pneu_grouped_acc = accuracy_score(grouped_pneu_true, grouped_pneu_pred)

    grouped_all_true = [
        "Opacities" if n in ["Atelectasis", "Pulmonary Edema"] else n for n in grouped_pneu_true
    ]
    grouped_all_pred = [
        "Opacities" if n in ["Atelectasis", "Pulmonary Edema"] else n for n in grouped_pneu_pred
    ]
    clinical_grouped_acc = accuracy_score(grouped_all_true, grouped_all_pred)

    # 3. Per-Class Detailed Breakdown (Sensitivity, Specificity, AUC)
    cm = confusion_matrix(all_targets_arr, all_preds_arr)
    per_class_rows = []
    for i, c in enumerate(TARGET_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        try:
            auc_c = float(roc_auc_score((all_targets_arr == i).astype(int), all_probs_arr[:, i]))
        except Exception:
            auc_c = 0.0

        per_class_rows.append({
            "Disease": c,
            "Sensitivity (Recall)": f"{sens*100:.2f}%",
            "Specificity": f"{spec*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "F1-Score": f"{f1_c:.4f}",
            "ROC-AUC": f"{auc_c:.4f}",
            "Support (Cases)": int(cm[i, :].sum()),
            "raw_sens": float(sens),
            "raw_spec": float(spec),
            "raw_prec": float(prec),
            "raw_f1": float(f1_c),
            "raw_auc": float(auc_c),
        })

    avg_specificity = float(np.mean([r["raw_spec"] for r in per_class_rows]))

    # Print Summary Report
    print("\n" + "=" * 80)
    print(f"             FINAL EVALUATION RESULTS ({args.split.upper()} SET - {total_samples:,} RADIOGRAPHS)")
    print("=" * 80)
    print(f"Top-1 Multi-Class Accuracy      : {top1_acc * 100:.2f}%")
    print(f"Top-2 Clinical Diagnostic Acc   : {top2_acc * 100:.2f}% (Triage / Emergency Standard)")
    print(f"Macro ROC-AUC (OVR)             : {macro_roc_auc:.4f} ({macro_roc_auc*100:.2f}%)")
    print(f"Macro Precision                 : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Macro Recall (Sensitivity)      : {recall:.4f} ({recall*100:.2f}%)")
    print(f"Macro F1-Score                  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"Average Specificity             : {avg_specificity:.4f} ({avg_specificity*100:.2f}%)")
    print("-" * 80)
    print(f"Pneumonia Unified Accuracy      : {pneu_grouped_acc * 100:.2f}% (Bacterial + Viral combined)")
    print(f"Clinical Grouped Accuracy       : {clinical_grouped_acc * 100:.2f}% (Pneumonias & Opacities unified)")
    print(f"Mean Predictive Confidence      : {np.mean(all_confidences):.4f}")
    print(f"Mean Epistemic Uncertainty      : {np.mean(all_uncertainties):.4f} (Normalized Shannon Entropy)")
    print(f"Total Inference Time            : {duration:.2f}s ({total_samples / duration:.1f} radiographs/sec)")
    print("=" * 80)

    print("\nPer-Class Clinical Diagnostic Breakdown:")
    header = f"{'Disease':<22} | {'Sensitivity':<12} | {'Specificity':<12} | {'Precision':<12} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'Support':<7}"
    print(header)
    print("-" * len(header))
    for r in per_class_rows:
        print(f"{r['Disease']:<22} | {r['Sensitivity (Recall)']:<12} | {r['Specificity']:<12} | {r['Precision']:<12} | {r['F1-Score']:<10} | {r['ROC-AUC']:<9} | {r['Support (Cases)']:<7}")

    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # Save Output Files
    print("\nSaving evaluation artifacts...")
    plot_confusion_matrix(cm, TARGET_CLASSES, output_dir / "confusion_matrix.png")
    plot_roc_curves(all_targets_arr, all_probs_arr, TARGET_CLASSES, output_dir / "roc_curves.png")
    is_correct = [t == p for t, p in zip(all_targets_arr, all_preds_arr)]
    plot_uncertainty_analysis(is_correct, all_uncertainties, all_confidences, output_dir / "uncertainty_distribution.png")

    # CSV: Per-class metrics
    df_per_class = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("raw_")} for r in per_class_rows])
    df_per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # CSV: Standard classification report
    report_dict = classification_report(all_targets_arr, all_preds_arr, target_names=TARGET_CLASSES, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "classification_report.csv")

    # CSV: Predictions log
    df_preds = pd.DataFrame({
        "sample_id": range(total_samples),
        "true_label_idx": all_targets_arr,
        "true_diagnosis": true_names,
        "predicted_label_idx": all_preds_arr,
        "predicted_diagnosis": pred_names,
        "top1_correct": is_correct,
        "confidence": all_confidences,
        "mc_uncertainty_entropy": all_uncertainties,
    })
    for i, c in enumerate(TARGET_CLASSES):
        df_preds[f"prob_{c}"] = all_probs_arr[:, i]
    df_preds.to_csv(output_dir / "test_predictions.csv", index=False)

    # JSON: Metrics summary
    metrics_summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "total_test_samples": total_samples,
        "checkpoint_evaluated": str(ckpt_path.resolve()),
        "checkpoint_epoch": int(saved_epoch) if saved_epoch != "N/A" else None,
        "inference_mode": mode_str,
        "headline_metrics": {
            "top1_accuracy_percent": round(top1_acc * 100, 2),
            "top2_accuracy_percent": round(top2_acc * 100, 2),
            "macro_roc_auc": round(macro_roc_auc, 4),
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
            "average_specificity": round(avg_specificity, 4),
            "pneumonia_grouped_accuracy_percent": round(pneu_grouped_acc * 100, 2),
            "clinical_grouped_accuracy_percent": round(clinical_grouped_acc * 100, 2),
            "mean_predictive_confidence": round(float(np.mean(all_confidences)), 4),
            "mean_uncertainty_entropy": round(float(np.mean(all_uncertainties)), 4),
            "inference_duration_seconds": round(duration, 2),
            "throughput_samples_per_sec": round(total_samples / duration, 1),
        },
        "per_class_metrics": {
            r["Disease"]: {
                "sensitivity": r["Sensitivity (Recall)"],
                "specificity": r["Specificity"],
                "precision": r["Precision"],
                "f1_score": r["F1-Score"],
                "roc_auc": r["ROC-AUC"],
                "support": r["Support (Cases)"],
            }
            for r in per_class_rows
        },
        "confusion_matrix_raw": cm.tolist(),
        "classes": TARGET_CLASSES,
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # Text summary file
    summary_text_path = output_dir / "evaluation_summary.txt"
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"RESPIRA - MODEL EVALUATION SUMMARY ({args.split.upper()} SET - {total_samples:,} RADIOGRAPHS)\n")
        f.write(f"Evaluated Checkpoint: {ckpt_path.name} (Trained Epoch {saved_epoch})\n")
        f.write(f"Evaluation Mode     : {mode_str}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Top-1 Accuracy                 : {top1_acc * 100:.2f}%\n")
        f.write(f"Top-2 Clinical Diagnostic Acc   : {top2_acc * 100:.2f}%\n")
        f.write(f"Macro ROC-AUC (OVR)             : {macro_roc_auc:.4f} ({macro_roc_auc*100:.2f}%)\n")
        f.write(f"Macro Precision                : {precision:.4f}\n")
        f.write(f"Macro Recall (Sensitivity)     : {recall:.4f}\n")
        f.write(f"Macro F1-Score                 : {f1:.4f}\n")
        f.write(f"Average Specificity            : {avg_specificity:.4f}\n")
        f.write(f"Pneumonia Unified Accuracy     : {pneu_grouped_acc * 100:.2f}%\n")
        f.write(f"Clinical Grouped Accuracy      : {clinical_grouped_acc * 100:.2f}%\n")
        f.write(f"Inference Throughput           : {total_samples / duration:.1f} samples/sec\n\n")
        f.write("-" * 80 + "\n")
        f.write("PER-CLASS CLINICAL PERFORMANCE BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in per_class_rows:
            f.write(f"{r['Disease']:<22} | {r['Sensitivity (Recall)']:<12} | {r['Specificity']:<12} | {r['Precision']:<12} | {r['F1-Score']:<10} | {r['ROC-AUC']:<9} | {r['Support (Cases)']:<7}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("CONFUSION MATRIX (RAW COUNTS)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'True \\ Pred':<22} | " + " | ".join(f"{c[:7]:>7}" for c in TARGET_CLASSES) + "\n")
        f.write("-" * 80 + "\n")
        for i, c in enumerate(TARGET_CLASSES):
            row_str = " | ".join(f"{cm[i, j]:>7d}" for j in range(NUM_CLASSES))
            f.write(f"{c:<22} | {row_str}\n")
        f.write("=" * 80 + "\n")

    print(f"[*] Complete evaluation summary written to: {summary_text_path.resolve()}")
    print(f"[*] All evaluation files saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
