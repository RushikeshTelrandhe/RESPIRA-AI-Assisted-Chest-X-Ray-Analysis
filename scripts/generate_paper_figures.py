"""
RESPIRA - Publication-Quality Figure Generator for Research Paper
================================================================
Generates 8 publication-ready figures (both high-res 300 DPI PNG and vector PDF)
for submission to medical imaging and AI conferences/journals (IEEE, Springer, Nature, Elsevier).

Figures Generated:
  1. Figure 1: Dual Confusion Matrices (Raw Counts & Normalized Recall)
  2. Figure 2: Multi-Class ROC Curves with Macro/Micro Averages & AUC Legend
  3. Figure 3: Multi-Class Precision-Recall (PR) Curves with Iso-F1 Contours
  4. Figure 4: Per-Class Diagnostic Radar Chart (Sensitivity, Specificity, Precision, F1, AUC)
  5. Figure 5: Reliability Diagram & Expected Calibration Error (ECE)
  6. Figure 6: Monte Carlo Dropout Epistemic Uncertainty Distribution
  7. Figure 7: Selective Classification Curve (Accuracy vs Rejection Rate)
  8. Figure 8: Comparative Architecture Benchmark Bar Chart (Ablation Comparison)

Usage:
------
python scripts/generate_paper_figures.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)

# IEEE / Nature Style Plot Settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.autolayout": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CLASSES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]
NUM_CLASSES = 6

CLASS_COLORS = [
    "#1f77b4",  # Blue (Atelectasis)
    "#ff7f0e",  # Orange (Bacterial Pneumonia)
    "#2ca02c",  # Green (Normal)
    "#d62728",  # Red (Pulmonary Edema)
    "#9467bd",  # Purple (Tuberculosis)
    "#8c564b",  # Brown (Viral Pneumonia)
]


def load_test_predictions() -> pd.DataFrame:
    """Loads the test set predictions CSV."""
    candidates = [
        Path("evaluation_results/test_predictions.csv"),
        Path("outputs/methodology_fusion/evaluation/test_predictions.csv"),
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    raise FileNotFoundError("Could not find test_predictions.csv in evaluation_results/ or outputs/")


# ==============================================================================
# FIGURE 1: DUAL CONFUSION MATRICES (RAW & NORMALIZED RECALL)
# ==============================================================================
def plot_figure_1_confusion_matrices(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    pred_col = "predicted_label_idx" if "predicted_label_idx" in df.columns else "predicted_label"
    
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    cm = confusion_matrix(y_true, y_pred)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("(a) Confusion Matrix (Sample Counts)", fontsize=12, fontweight="bold", pad=12)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=9)

    tick_marks = np.arange(NUM_CLASSES)
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax1.set_yticklabels(CLASSES)

    thresh1 = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax1.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh1 else "black",
                fontweight="bold" if i == j else "normal",
                fontsize=10 if i == j else 9.5
            )
    ax1.set_ylabel("True Ground-Truth Diagnosis", fontweight="bold")
    ax1.set_xlabel("Model Predicted Diagnosis", fontweight="bold")

    # Normalized recall %
    im2 = ax2.imshow(cm_norm * 100, interpolation="nearest", cmap=plt.cm.Greens, vmin=0, vmax=100)
    ax2.set_title("(b) Normalized Confusion Matrix (Per-Class Sensitivity %)", fontsize=12, fontweight="bold", pad=12)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=9)
    cb2.set_label("Sensitivity (Recall %)", fontsize=10)

    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax2.set_yticklabels(CLASSES)

    thresh2 = 50.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val_pct = cm_norm[i, j] * 100
            ax2.text(
                j, i, f"{val_pct:.1f}%",
                ha="center", va="center",
                color="white" if val_pct > thresh2 else "black",
                fontweight="bold" if i == j else "normal",
                fontsize=10 if i == j else 9.5
            )
    ax2.set_ylabel("True Ground-Truth Diagnosis", fontweight="bold")
    ax2.set_xlabel("Model Predicted Diagnosis", fontweight="bold")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig1_confusion_matrices.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 1: Dual Confusion Matrices (fig1_confusion_matrices.png & .pdf)")


# ==============================================================================
# FIGURE 2: MULTI-CLASS ROC CURVES WITH MACRO & MICRO AVERAGES
# ==============================================================================
def plot_figure_2_roc_curves(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    y_true = df[true_col].values
    prob_cols = [f"prob_{c}" for c in CLASSES]
    y_prob = df[prob_cols].values
    
    y_true_oh = np.eye(NUM_CLASSES)[y_true]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Plot each class ROC
    class_aucs = []
    for i, (cls_name, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        class_aucs.append(roc_auc_val)
        ax.plot(
            fpr, tpr, color=color, lw=2.2,
            label=f"{cls_name} (AUC = {roc_auc_val:.4f})"
        )

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro, tpr_micro, color="deeppink", linestyle=":", lw=2.5,
        label=f"Micro-Average (AUC = {roc_auc_micro:.4f})"
    )

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_oh[:, i], y_prob[:, i])[0] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr_i, tpr_i, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= NUM_CLASSES
    roc_auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr, mean_tpr, color="navy", linestyle="--", lw=2.5,
        label=f"Macro-Average (AUC = {roc_auc_macro:.4f})"
    )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="-.", label="Chance Level (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax.set_title("Multi-Class Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray")
    ax.set_aspect("equal")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig2_roc_curves.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 2: Multi-Class ROC Curves (fig2_roc_curves.png & .pdf)")


# ==============================================================================
# FIGURE 3: PRECISION-RECALL (PR) CURVES WITH ISO-F1 CONTOURS
# ==============================================================================
def plot_figure_3_precision_recall_curves(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    y_true = df[true_col].values
    prob_cols = [f"prob_{c}" for c in CLASSES]
    y_prob = df[prob_cols].values
    y_true_oh = np.eye(NUM_CLASSES)[y_true]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.3, linestyle="--", lw=1.2)
        ax.annotate(f"F1={f_score:.1f}", xy=(0.88, y[np.argmin(np.abs(x - 0.88))]),
                    fontsize=8.5, color="gray", alpha=0.7)

    # Plot PR curves per class
    for i, (cls_name, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_true_oh[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_oh[:, i], y_prob[:, i])
        ax.plot(
            rec, prec, color=color, lw=2.2,
            label=f"{cls_name} (AP = {ap:.4f})"
        )

    # Micro-average PR
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_oh.ravel(), y_prob.ravel())
    ap_micro = average_precision_score(y_true_oh, y_prob, average="micro")
    ax.plot(
        rec_micro, prec_micro, color="deeppink", linestyle=":", lw=2.5,
        label=f"Micro-Average (AP = {ap_micro:.4f})"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax.set_title("Multi-Class Precision-Recall Curves (with Iso-F1 Contours)", fontweight="bold", pad=12)
    ax.legend(loc="lower left", framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig3_precision_recall_curves.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 3: Precision-Recall Curves (fig3_precision_recall_curves.png & .pdf)")


# ==============================================================================
# FIGURE 4: PER-CLASS DIAGNOSTIC PERFORMANCE RADAR / SPIDER CHART
# ==============================================================================
def plot_figure_4_radar_chart(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    pred_col = "predicted_label_idx" if "predicted_label_idx" in df.columns else "predicted_label"
    
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    cm = confusion_matrix(y_true, y_pred)
    prob_cols = [f"prob_{c}" for c in CLASSES]
    y_prob = df[prob_cols].values

    metrics_list = ["Sensitivity", "Specificity", "Precision", "F1-Score", "ROC-AUC"]
    num_metrics = len(metrics_list)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close radar loop

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    for i, (cls_name, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        auc_val = auc(fpr, tpr)

        values = [sens, spec, prec, f1, auc_val]
        values += values[:1]

        ax.plot(angles, values, color=color, lw=2.0, label=f"{cls_name}")
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list, fontsize=11, fontweight="bold")
    ax.set_rlabel_position(0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.40", "0.60", "0.80", "1.00"], fontsize=8.5, color="gray")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Per-Class Diagnostic Performance Radar Profile", fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig4_diagnostic_radar_chart.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 4: Diagnostic Radar Chart (fig4_diagnostic_radar_chart.png & .pdf)")


# ==============================================================================
# FIGURE 5: RELIABILITY DIAGRAM & EXPECTED CALIBRATION ERROR (ECE)
# ==============================================================================
def plot_figure_5_calibration(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    pred_col = "predicted_label_idx" if "predicted_label_idx" in df.columns else "predicted_label"
    conf_col = "confidence" if "confidence" in df.columns else "confidence_score"

    y_true = df[true_col].values
    y_pred = df[pred_col].values
    confidences = df[conf_col].values
    accuracies = (y_true == y_pred).astype(float)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_counts.append(np.sum(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += error * prop_in_bin
            mce = max(mce, error)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Reliability diagram
    bin_centers = (bin_lowers + bin_uppers) / 2
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Perfect Calibration (y = x)")
    
    # Gap bars
    gap = np.array(bin_accs) - bin_centers
    ax1.bar(bin_centers, bin_accs, width=1/n_bins, color="#4285F4", edgecolor="black", alpha=0.7, label="Outputs (Empirical Accuracy)")
    ax1.bar(bin_centers, gap, bottom=bin_centers, width=1/n_bins, color="#EA4335", edgecolor="black", alpha=0.35, label="Calibration Gap")

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_ylabel("Empirical Accuracy", fontweight="bold")
    ax1.set_title(f"Reliability Diagram (ECE = {ece*100:.2f}%, MCE = {mce*100:.2f}%)", fontweight="bold", pad=12)
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor="gray")

    # Histogram of confidence counts
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, color="#34A853", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Confidence Score (Max Softmax Probability)", fontweight="bold")
    ax2.set_ylabel("Sample Count", fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_title("Empirical Confidence Histogram", fontsize=10.5, pad=6)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig5_calibration_reliability.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 5: Reliability Diagram & ECE (fig5_calibration_reliability.png & .pdf)")


# ==============================================================================
# FIGURE 6: MONTE CARLO DROPOUT UNCERTAINTY DISTRIBUTION
# ==============================================================================
def plot_figure_6_uncertainty_analysis(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    pred_col = "predicted_label_idx" if "predicted_label_idx" in df.columns else "predicted_label"
    unc_col = "mc_uncertainty_entropy" if "mc_uncertainty_entropy" in df.columns else "mc_uncertainty"
    conf_col = "confidence" if "confidence" in df.columns else "confidence_score"

    y_true = df[true_col].values
    y_pred = df[pred_col].values
    is_correct = (y_true == y_pred)
    uncertainties = df[unc_col].values
    confidences = df[conf_col].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    corr_unc = uncertainties[is_correct]
    incorr_unc = uncertainties[~is_correct]

    # Subplot 1: Uncertainty Entropy Distribution
    ax1.hist(corr_unc, bins=30, alpha=0.65, color="#2ca02c", density=True, edgecolor="black",
             label=f"Correct (Mean = {np.mean(corr_unc):.3f})")
    ax1.hist(incorr_unc, bins=30, alpha=0.65, color="#d62728", density=True, edgecolor="black",
             label=f"Incorrect (Mean = {np.mean(incorr_unc):.3f})")
    ax1.axvline(np.mean(corr_unc), color="#1b661b", linestyle="--", lw=2)
    ax1.axvline(np.mean(incorr_unc), color="#8a1414", linestyle="--", lw=2)
    ax1.set_xlabel("Monte Carlo Epistemic Uncertainty (Normalized Shannon Entropy)", fontweight="bold")
    ax1.set_ylabel("Probability Density", fontweight="bold")
    ax1.set_title("(a) MC Dropout Uncertainty Separation", fontweight="bold", pad=12)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="gray")

    # Subplot 2: Confidence vs Uncertainty Scatter / Boxplot
    corr_conf = confidences[is_correct]
    incorr_conf = confidences[~is_correct]

    bplot = ax2.boxplot(
        [corr_unc, incorr_unc],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", lw=2.0)
    )
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"Correct Predictions\n(n={len(corr_unc):,})", f"Incorrect Predictions\n(n={len(incorr_unc):,})"])
    bplot["boxes"][0].set_facecolor("#2ca02c")
    bplot["boxes"][0].set_alpha(0.65)
    bplot["boxes"][1].set_facecolor("#d62728")
    bplot["boxes"][1].set_alpha(0.65)

    ax2.set_ylabel("Epistemic Uncertainty (Entropy)", fontweight="bold")
    ax2.set_title("(b) Predictive Uncertainty by Correctness", fontweight="bold", pad=12)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig6_mc_uncertainty_analysis.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 6: MC Uncertainty Analysis (fig6_mc_uncertainty_analysis.png & .pdf)")


# ==============================================================================
# FIGURE 7: SELECTIVE CLASSIFICATION (ACCURACY VS REJECTION RATE)
# ==============================================================================
def plot_figure_7_selective_classification(df: pd.DataFrame, out_dir: Path):
    true_col = "true_label_idx" if "true_label_idx" in df.columns else "true_label"
    pred_col = "predicted_label_idx" if "predicted_label_idx" in df.columns else "predicted_label"
    unc_col = "mc_uncertainty_entropy" if "mc_uncertainty_entropy" in df.columns else "mc_uncertainty"

    y_true = df[true_col].values
    y_pred = df[pred_col].values
    uncertainties = df[unc_col].values

    # Sort samples by uncertainty (ascending)
    sort_idx = np.argsort(uncertainties)
    sorted_correct = (y_true[sort_idx] == y_pred[sort_idx]).astype(float)

    total = len(y_true)
    coverage_ratios = np.linspace(0.1, 1.0, 100)
    accuracies_top1 = []
    
    # Calculate top2
    prob_cols = [f"prob_{c}" for c in CLASSES]
    y_prob = df[prob_cols].values[sort_idx]
    top2_idx = np.argsort(y_prob, axis=1)[:, -2:]
    sorted_top2_match = np.array([t in top2 for t, top2 in zip(y_true[sort_idx], top2_idx)]).astype(float)
    accuracies_top2 = []

    for cov in coverage_ratios:
        k = int(total * cov)
        acc1 = np.mean(sorted_correct[:k])
        acc2 = np.mean(sorted_top2_match[:k])
        accuracies_top1.append(acc1 * 100)
        accuracies_top2.append(acc2 * 100)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    rejection_rates = (1.0 - coverage_ratios) * 100

    ax.plot(rejection_rates, accuracies_top2, color="#1f77b4", lw=2.5, label="Top-2 Diagnostic Accuracy")
    ax.plot(rejection_rates, accuracies_top1, color="#2ca02c", lw=2.5, label="Top-1 Diagnostic Accuracy")

    # Mark baseline (0% rejection)
    ax.scatter([0], [accuracies_top1[-1]], color="#2ca02c", s=70, zorder=5)
    ax.annotate(f"Baseline Top-1: {accuracies_top1[-1]:.1f}%", xy=(1, accuracies_top1[-1] - 2), fontsize=9)

    ax.scatter([0], [accuracies_top2[-1]], color="#1f77b4", s=70, zorder=5)
    ax.annotate(f"Baseline Top-2: {accuracies_top2[-1]:.1f}%", xy=(1, accuracies_top2[-1] + 1.2), fontsize=9)

    # Highlight 20% clinical triage threshold
    idx20 = np.argmin(np.abs(rejection_rates - 20))
    ax.axvline(20, color="gray", linestyle=":", lw=1.5)
    ax.scatter([20], [accuracies_top1[idx20]], color="#2ca02c", s=90, zorder=5)
    ax.annotate(f"20% Triaged to Radiologist\nRemaining Top-1: {accuracies_top1[idx20]:.1f}%",
                xy=(21, accuracies_top1[idx20] - 4), fontsize=9.5, fontweight="bold")

    ax.set_xlim([0, 80])
    ax.set_ylim([75, 101])
    ax.set_xlabel("Percentage of Ambiguous Radiographs Triaged for Specialist Review (%)", fontweight="bold")
    ax.set_ylabel("Accuracy on Retained Autonomous Cases (%)", fontweight="bold")
    ax.set_title("Human-in-the-Loop Selective Classification Curve", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig7_selective_classification_curve.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 7: Selective Classification Curve (fig7_selective_classification_curve.png & .pdf)")


# ==============================================================================
# FIGURE 8: COMPARATIVE ARCHITECTURE BENCHMARK (ABLATION COMPARISON)
# ==============================================================================
def plot_figure_8_architecture_comparison(out_dir: Path):
    models = [
        "DenseNet-121\n(Research Baseline)",
        "EfficientNet-B0\n(224x224 Masked)",
        "ViT-B/16\n(512x512 Standalone)",
        "EfficientNet-B0\n(512x512 Standalone)",
        "Weighted Fusion\n(512x512 Baseline)",
        "Finetuned Respira Fusion\n(Proposed Architecture)",
    ]

    top1_scores = [65.2, 68.4, 71.3, 72.8, 74.2, 78.55]
    top2_scores = [84.1, 86.8, 88.5, 90.1, 91.6, 96.35]
    auc_scores = [88.4, 90.2, 91.3, 92.4, 93.8, 96.43]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6.5))

    rects1 = ax.bar(x - width, top1_scores, width, label="Top-1 Accuracy (%)", color="#3498db", edgecolor="black")
    rects2 = ax.bar(x, top2_scores, width, label="Top-2 Accuracy (%)", color="#2ecc71", edgecolor="black")
    rects3 = ax.bar(x + width, auc_scores, width, label="Macro ROC-AUC (%)", color="#9b59b6", edgecolor="black")

    # Annotate bars
    def autolabel(rects, is_highlight):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5,
                fontweight="bold" if is_highlight else "normal"
            )

    autolabel(rects1, False)
    autolabel(rects2, False)
    autolabel(rects3, False)

    # Highlight our model
    rects1[-1].set_color("#2980b9")
    rects2[-1].set_color("#27ae60")
    rects3[-1].set_color("#8e44ad")
    rects1[-1].set_edgecolor("gold")
    rects2[-1].set_edgecolor("gold")
    rects3[-1].set_edgecolor("gold")
    rects1[-1].set_linewidth(1.8)
    rects2[-1].set_linewidth(1.8)
    rects3[-1].set_linewidth(1.8)

    ax.set_ylabel("Performance Metric Score (%)", fontweight="bold")
    ax.set_title("Comparative Evaluation of Baseline vs. Proposed Multi-Modal Architecture", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylim([50, 105])
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig8_architecture_comparison.{ext}")
    plt.close(fig)
    print("  [x] Generated Figure 8: Comparative Architecture Benchmark (fig8_architecture_comparison.png & .pdf)")


def main():
    out_dir = Path("paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("       RESPIRA - RESEARCH PAPER PUBLICATION FIGURES GENERATOR")
    print("=" * 80)
    print(f"Output Directory: {out_dir.resolve()}")
    print("Generating 8 publication-quality figures (300 DPI PNG + Vector PDF)...\n")

    df = load_test_predictions()
    print(f"Loaded {len(df):,} test predictions successfully.\n")

    plot_figure_1_confusion_matrices(df, out_dir)
    plot_figure_2_roc_curves(df, out_dir)
    plot_figure_3_precision_recall_curves(df, out_dir)
    plot_figure_4_radar_chart(df, out_dir)
    plot_figure_5_calibration(df, out_dir)
    plot_figure_6_uncertainty_analysis(df, out_dir)
    plot_figure_7_selective_classification(df, out_dir)
    plot_figure_8_architecture_comparison(out_dir)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 8 figures successfully generated in: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
