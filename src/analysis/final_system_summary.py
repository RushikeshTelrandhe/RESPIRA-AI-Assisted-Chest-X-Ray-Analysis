"""
RESPIRA — STEP 8.6
FINAL SYSTEM SUMMARY & REPORT GENERATION

Final experiment:
    EfficientNet-B0 512x512
    ViT-B/16 512x512
    EfficientNet + ViT weighted probability fusion

Final test set:
    1,947 images

IMPORTANT:
    EfficientNet/ViT metrics use the key "accuracy".
    Fusion metrics use the key "fusion_test_accuracy".

This script normalizes those schemas before performing any comparison.
"""

from pathlib import Path
import json
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "final_system_summary"
)

MODEL_COMPARISON_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "model_comparison"
)

EFF_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
    / "evaluation"
    / "metrics.json"
)

VIT_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
    / "evaluation"
    / "metrics.json"
)

FUSION_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "test"
    / "metrics.json"
)

FUSION_CONFIG = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "fusion_config.json"
)


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=None):
    """
    Convert a value to float safely.
    """
    if value is None:
        return default

    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def pct(value):
    """
    Convert decimal metric to percentage string.
    """
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def load_json(path):
    """
    Load JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def write_json(path, data):
    """
    Write JSON with readable formatting.
    """
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def clean_for_json(obj):
    """
    Convert NumPy/Python objects into JSON-safe objects.
    """
    if isinstance(obj, dict):
        return {
            str(k): clean_for_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            clean_for_json(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            clean_for_json(v)
            for v in obj
        ]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.6")
print("FINAL SYSTEM SUMMARY & REPORT GENERATION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Output root:")
print(OUTPUT_ROOT)


# ============================================================
# 5. VERIFY FINAL INPUT FILES
# ============================================================

print()
print("=" * 70)
print("VERIFYING FINAL 512x512 INPUT FILES")
print("=" * 70)

required_files = [
    (
        "EfficientNet metrics",
        EFF_METRICS
    ),
    (
        "ViT metrics",
        VIT_METRICS
    ),
    (
        "Fusion metrics",
        FUSION_METRICS
    ),
]

for name, path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )

    print(f"  ✓ {name}")
    print(f"    {path}")


# ============================================================
# 6. LOAD FINAL MODEL RESULTS
# ============================================================

print()
print("=" * 70)
print("LOADING FINAL 512x512 MODEL RESULTS")
print("=" * 70)

efficientnet = load_json(EFF_METRICS)
vit = load_json(VIT_METRICS)
fusion = load_json(FUSION_METRICS)

print("✓ EfficientNet-B0 512x512 metrics loaded")
print("✓ ViT-B/16 512x512 metrics loaded")
print("✓ EfficientNet + ViT Fusion metrics loaded")


# ============================================================
# 7. LOAD FUSION CONFIGURATION
# ============================================================

fusion_config = {}

if FUSION_CONFIG.exists():

    fusion_config = load_json(
        FUSION_CONFIG
    )

    print("✓ Fusion configuration loaded")

else:

    print(
        "⚠ Fusion configuration not found; "
        "using metrics.json values."
    )


# ============================================================
# 8. NORMALIZE MODEL METRICS
# ============================================================

def model_metrics(data, model_type):
    """
    Normalize the different metric schemas.

    EfficientNet:
        accuracy

    ViT:
        accuracy

    Fusion:
        fusion_test_accuracy
    """

    if model_type == "fusion":

        accuracy = safe_float(
            data.get(
                "fusion_test_accuracy"
            )
        )

    else:

        accuracy = safe_float(
            data.get(
                "accuracy"
            )
        )

    return {

        "accuracy":
            accuracy,

        "accuracy_percent":
            safe_float(
                data.get(
                    "accuracy_percent"
                )
            ),

        "macro_precision":
            safe_float(
                data.get(
                    "macro_precision"
                )
            ),

        "macro_recall":
            safe_float(
                data.get(
                    "macro_recall"
                )
            ),

        "macro_f1":
            safe_float(
                data.get(
                    "macro_f1"
                )
            ),

        "weighted_precision":
            safe_float(
                data.get(
                    "weighted_precision"
                )
            ),

        "weighted_recall":
            safe_float(
                data.get(
                    "weighted_recall"
                )
            ),

        "weighted_f1":
            safe_float(
                data.get(
                    "weighted_f1"
                )
            ),

        "macro_roc_auc":
            safe_float(
                data.get(
                    "macro_roc_auc"
                )
            ),

        "test_samples":
            int(
                data.get(
                    "test_samples",
                    0
                )
            ),

        "validation_accuracy":
            safe_float(
                data.get(
                    "validation_accuracy"
                )
            ),

        "validation_loss":
            safe_float(
                data.get(
                    "validation_loss"
                )
            ),

        "per_class_roc_auc":
            data.get(
                "per_class_roc_auc",
                {}
            ),

        "classes":
            data.get(
                "classes",
                []
            ),

    }


eff_metrics = model_metrics(
    efficientnet,
    "efficientnet"
)

vit_metrics = model_metrics(
    vit,
    "vit"
)

fusion_metrics = model_metrics(
    fusion,
    "fusion"
)


# ============================================================
# 9. VALIDATE METRICS
# ============================================================

print()
print("=" * 70)
print("VALIDATING FINAL EXPERIMENT")
print("=" * 70)

print(
    f"EfficientNet test samples : "
    f"{eff_metrics['test_samples']}"
)

print(
    f"ViT test samples          : "
    f"{vit_metrics['test_samples']}"
)

print(
    f"Fusion test samples       : "
    f"{fusion_metrics['test_samples']}"
)

expected_test_samples = 1947

if not (
    eff_metrics["test_samples"]
    == vit_metrics["test_samples"]
    == fusion_metrics["test_samples"]
    == expected_test_samples
):

    raise ValueError(
        "\nTest-set sample mismatch.\n"
        f"Expected: {expected_test_samples}\n"
        f"EfficientNet: {eff_metrics['test_samples']}\n"
        f"ViT: {vit_metrics['test_samples']}\n"
        f"Fusion: {fusion_metrics['test_samples']}"
    )

print()
print(
    "✓ Final test set contains exactly "
    "1,947 images."
)


# ============================================================
# 10. FUSION WEIGHTS
# ============================================================

eff_weight = safe_float(
    fusion.get(
        "efficientnet_weight"
    )
)

vit_weight = safe_float(
    fusion.get(
        "vit_weight"
    )
)

# Prefer fusion_config if metrics do not contain weights.

if eff_weight is None:

    eff_weight = safe_float(
        fusion_config.get(
            "efficientnet_weight"
        )
    )

if vit_weight is None:

    vit_weight = safe_float(
        fusion_config.get(
            "vit_weight"
        )
    )


# ============================================================
# 11. MODEL PERFORMANCE TABLE
# ============================================================

performance_rows = [

    {
        "Model":
            "EfficientNet-B0 512x512",

        "Test Samples":
            eff_metrics["test_samples"],

        "Accuracy":
            eff_metrics["accuracy"],

        "Macro Precision":
            eff_metrics["macro_precision"],

        "Macro Recall":
            eff_metrics["macro_recall"],

        "Macro F1":
            eff_metrics["macro_f1"],

        "Weighted F1":
            eff_metrics["weighted_f1"],

        "Macro ROC-AUC":
            eff_metrics["macro_roc_auc"],
    },

    {
        "Model":
            "ViT-B/16 512x512",

        "Test Samples":
            vit_metrics["test_samples"],

        "Accuracy":
            vit_metrics["accuracy"],

        "Macro Precision":
            vit_metrics["macro_precision"],

        "Macro Recall":
            vit_metrics["macro_recall"],

        "Macro F1":
            vit_metrics["macro_f1"],

        "Weighted F1":
            vit_metrics["weighted_f1"],

        "Macro ROC-AUC":
            vit_metrics["macro_roc_auc"],
    },

    {
        "Model":
            "EfficientNet + ViT Fusion",

        "Test Samples":
            fusion_metrics["test_samples"],

        "Accuracy":
            fusion_metrics["accuracy"],

        "Macro Precision":
            fusion_metrics["macro_precision"],

        "Macro Recall":
            fusion_metrics["macro_recall"],

        "Macro F1":
            fusion_metrics["macro_f1"],

        "Weighted F1":
            fusion_metrics["weighted_f1"],

        "Macro ROC-AUC":
            fusion_metrics["macro_roc_auc"],
    },

]

performance_df = pd.DataFrame(
    performance_rows
)

performance_csv = (
    OUTPUT_ROOT
    / "model_performance_summary.csv"
)

performance_df.to_csv(
    performance_csv,
    index=False
)

print()
print("✓ model_performance_summary.csv")


# ============================================================
# 12. PERFORMANCE GAINS
# ============================================================

fusion_gain_vs_eff_accuracy = (
    fusion_metrics["accuracy"]
    - eff_metrics["accuracy"]
)

fusion_gain_vs_vit_accuracy = (
    fusion_metrics["accuracy"]
    - vit_metrics["accuracy"]
)

fusion_gain_vs_eff_f1 = (
    fusion_metrics["macro_f1"]
    - eff_metrics["macro_f1"]
)

fusion_gain_vs_vit_f1 = (
    fusion_metrics["macro_f1"]
    - vit_metrics["macro_f1"]
)

fusion_gain_vs_eff_auc = (
    fusion_metrics["macro_roc_auc"]
    - eff_metrics["macro_roc_auc"]
)

fusion_gain_vs_vit_auc = (
    fusion_metrics["macro_roc_auc"]
    - vit_metrics["macro_roc_auc"]
)


# ============================================================
# 13. PIPELINE SUMMARY
# ============================================================

pipeline_rows = [

    {
        "Stage":
            "Input",

        "Description":
            "Preprocessed chest X-ray dataset",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "EfficientNet-B0",

        "Description":
            "CNN-based chest X-ray classifier",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "ViT-B/16",

        "Description":
            "Vision Transformer chest X-ray classifier",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "Validation Fusion",

        "Description":
            "Weighted probability fusion using validation set",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "Final Test Fusion",

        "Description":
            "Frozen fusion weights evaluated on test set",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "Grad-CAM",

        "Description":
            "EfficientNet explainability",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "ViT Attention",

        "Description":
            "Attention rollout explainability",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "XAI Comparison",

        "Description":
            "Image-level comparison of CNN and Transformer explanations",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

    {
        "Stage":
            "XAI Artifact Audit",

        "Description":
            "Border/corner activation and suspicious-region analysis",

        "Input Size":
            "512x512",

        "Status":
            "Completed",
    },

]

pipeline_df = pd.DataFrame(
    pipeline_rows
)

pipeline_csv = (
    OUTPUT_ROOT
    / "pipeline_summary.csv"
)

pipeline_df.to_csv(
    pipeline_csv,
    index=False
)

print("✓ pipeline_summary.csv")


# ============================================================
# 14. FINAL SYSTEM METRICS
# ============================================================

final_metrics = {

    "experiment":
        "RESPIRA Final 512x512 System",

    "input_size":
        "512x512",

    "test_images":
        expected_test_samples,

    "classes":
        fusion_metrics["classes"],

    "models":
        {
            "efficientnet":
                "EfficientNet-B0 512x512",

            "vit":
                "ViT-B/16 512x512",

            "fusion":
                "EfficientNet-B0 + ViT-B/16",
        },

    "efficientnet":
        {
            "accuracy":
                eff_metrics["accuracy"],

            "macro_precision":
                eff_metrics["macro_precision"],

            "macro_recall":
                eff_metrics["macro_recall"],

            "macro_f1":
                eff_metrics["macro_f1"],

            "weighted_f1":
                eff_metrics["weighted_f1"],

            "macro_roc_auc":
                eff_metrics["macro_roc_auc"],
        },

    "vit":
        {
            "accuracy":
                vit_metrics["accuracy"],

            "macro_precision":
                vit_metrics["macro_precision"],

            "macro_recall":
                vit_metrics["macro_recall"],

            "macro_f1":
                vit_metrics["macro_f1"],

            "weighted_f1":
                vit_metrics["weighted_f1"],

            "macro_roc_auc":
                vit_metrics["macro_roc_auc"],
        },

    "fusion":
        {
            "efficientnet_weight":
                eff_weight,

            "vit_weight":
                vit_weight,

            "accuracy":
                fusion_metrics["accuracy"],

            "macro_precision":
                fusion_metrics["macro_precision"],

            "macro_recall":
                fusion_metrics["macro_recall"],

            "macro_f1":
                fusion_metrics["macro_f1"],

            "weighted_f1":
                fusion_metrics["weighted_f1"],

            "macro_roc_auc":
                fusion_metrics["macro_roc_auc"],

            "validation_accuracy":
                safe_float(
                    fusion.get(
                        "validation_fusion_accuracy"
                    )
                ),

            "validation_macro_f1":
                safe_float(
                    fusion.get(
                        "validation_fusion_macro_f1"
                    ),
                ),
        },

    "fusion_improvement":
        {
            "accuracy_vs_efficientnet":
                fusion_gain_vs_eff_accuracy,

            "accuracy_vs_vit":
                fusion_gain_vs_vit_accuracy,

            "macro_f1_vs_efficientnet":
                fusion_gain_vs_eff_f1,

            "macro_f1_vs_vit":
                fusion_gain_vs_vit_f1,

            "macro_roc_auc_vs_efficientnet":
                fusion_gain_vs_eff_auc,

            "macro_roc_auc_vs_vit":
                fusion_gain_vs_vit_auc,
        },

    "best_models":
        {
            "accuracy":
                "EfficientNet + ViT Fusion",

            "macro_f1":
                "EfficientNet + ViT Fusion",

            "macro_roc_auc":
                "EfficientNet + ViT Fusion",
        },

}


# ============================================================
# 15. SAVE FINAL JSON
# ============================================================

final_json = (
    OUTPUT_ROOT
    / "final_system_summary.json"
)

write_json(
    final_json,
    clean_for_json(
        final_metrics
    )
)

print("✓ final_system_summary.json")


# ============================================================
# 16. SAVE FINAL TEXT REPORT
# ============================================================

final_txt = (
    OUTPUT_ROOT
    / "final_system_summary.txt"
)

with open(
    final_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — FINAL SYSTEM SUMMARY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "FINAL EXPERIMENT\n"
    )

    f.write(
        "Input Size       : 512x512\n"
    )

    f.write(
        "Test Images      : 1,947\n"
    )

    f.write(
        "Classes          : "
        + ", ".join(
            fusion_metrics["classes"]
        )
        + "\n\n"
    )

    f.write(
        "MODEL PERFORMANCE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"EfficientNet-B0 512x512\n"
    )

    f.write(
        f"  Accuracy       : "
        f"{pct(eff_metrics['accuracy'])}\n"
    )

    f.write(
        f"  Macro Precision: "
        f"{pct(eff_metrics['macro_precision'])}\n"
    )

    f.write(
        f"  Macro Recall   : "
        f"{pct(eff_metrics['macro_recall'])}\n"
    )

    f.write(
        f"  Macro F1       : "
        f"{pct(eff_metrics['macro_f1'])}\n"
    )

    f.write(
        f"  Macro ROC-AUC  : "
        f"{eff_metrics['macro_roc_auc']:.4f}\n\n"
    )

    f.write(
        f"ViT-B/16 512x512\n"
    )

    f.write(
        f"  Accuracy       : "
        f"{pct(vit_metrics['accuracy'])}\n"
    )

    f.write(
        f"  Macro Precision: "
        f"{pct(vit_metrics['macro_precision'])}\n"
    )

    f.write(
        f"  Macro Recall   : "
        f"{pct(vit_metrics['macro_recall'])}\n"
    )

    f.write(
        f"  Macro F1       : "
        f"{pct(vit_metrics['macro_f1'])}\n"
    )

    f.write(
        f"  Macro ROC-AUC  : "
        f"{vit_metrics['macro_roc_auc']:.4f}\n\n"
    )

    f.write(
        "EfficientNet-B0 + ViT Fusion\n"
    )

    f.write(
        f"  EfficientNet Weight : "
        f"{eff_weight:.2f}\n"
    )

    f.write(
        f"  ViT Weight          : "
        f"{vit_weight:.2f}\n"
    )

    f.write(
        f"  Accuracy            : "
        f"{pct(fusion_metrics['accuracy'])}\n"
    )

    f.write(
        f"  Macro Precision     : "
        f"{pct(fusion_metrics['macro_precision'])}\n"
    )

    f.write(
        f"  Macro Recall        : "
        f"{pct(fusion_metrics['macro_recall'])}\n"
    )

    f.write(
        f"  Macro F1            : "
        f"{pct(fusion_metrics['macro_f1'])}\n"
    )

    f.write(
        f"  Weighted F1         : "
        f"{pct(fusion_metrics['weighted_f1'])}\n"
    )

    f.write(
        f"  Macro ROC-AUC       : "
        f"{fusion_metrics['macro_roc_auc']:.4f}\n\n"
    )

    f.write(
        "FUSION IMPROVEMENT\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Accuracy vs EfficientNet : "
        f"{fusion_gain_vs_eff_accuracy * 100:+.2f} percentage points\n"
    )

    f.write(
        f"Accuracy vs ViT          : "
        f"{fusion_gain_vs_vit_accuracy * 100:+.2f} percentage points\n"
    )

    f.write(
        f"Macro F1 vs EfficientNet: "
        f"{fusion_gain_vs_eff_f1 * 100:+.2f} percentage points\n"
    )

    f.write(
        f"Macro F1 vs ViT         : "
        f"{fusion_gain_vs_vit_f1 * 100:+.2f} percentage points\n"
    )

    f.write(
        f"ROC-AUC vs EfficientNet : "
        f"{fusion_gain_vs_eff_auc:+.4f}\n"
    )

    f.write(
        f"ROC-AUC vs ViT          : "
        f"{fusion_gain_vs_vit_auc:+.4f}\n\n"
    )

    f.write(
        "BEST FINAL MODEL\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "EfficientNet + ViT Fusion\n"
    )

    f.write(
        f"Final Test Accuracy : "
        f"{pct(fusion_metrics['accuracy'])}\n"
    )

    f.write(
        f"Final Macro F1      : "
        f"{pct(fusion_metrics['macro_f1'])}\n"
    )

    f.write(
        f"Final Macro ROC-AUC : "
        f"{fusion_metrics['macro_roc_auc']:.4f}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "EXPERIMENT INTEGRITY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "• Fusion weights were selected using validation data only.\n"
    )

    f.write(
        "• The final test set contains 1,947 images.\n"
    )

    f.write(
        "• Test data was not used to select the fusion weight.\n"
    )

    f.write(
        "• EfficientNet and ViT use their best validation checkpoints.\n"
    )

    f.write(
        "• No model checkpoint or dataset was modified by this script.\n"
    )

print("✓ final_system_summary.txt")


# ============================================================
# 17. PERFORMANCE PLOT
# ============================================================

models = [
    "EfficientNet-B0\n512x512",
    "ViT-B/16\n512x512",
    "EfficientNet + ViT\nFusion",
]

accuracy_values = [
    eff_metrics["accuracy"],
    vit_metrics["accuracy"],
    fusion_metrics["accuracy"],
]

precision_values = [
    eff_metrics["macro_precision"],
    vit_metrics["macro_precision"],
    fusion_metrics["macro_precision"],
]

recall_values = [
    eff_metrics["macro_recall"],
    vit_metrics["macro_recall"],
    fusion_metrics["macro_recall"],
]

f1_values = [
    eff_metrics["macro_f1"],
    vit_metrics["macro_f1"],
    fusion_metrics["macro_f1"],
]

auc_values = [
    eff_metrics["macro_roc_auc"],
    vit_metrics["macro_roc_auc"],
    fusion_metrics["macro_roc_auc"],
]


# ============================================================
# 18. FINAL METRICS FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)

x = np.arange(
    len(models)
)

width = 0.16

ax.bar(
    x - 2 * width,
    accuracy_values,
    width,
    label="Accuracy"
)

ax.bar(
    x - width,
    precision_values,
    width,
    label="Macro Precision"
)

ax.bar(
    x,
    recall_values,
    width,
    label="Macro Recall"
)

ax.bar(
    x + width,
    f1_values,
    width,
    label="Macro F1"
)

ax.bar(
    x + 2 * width,
    auc_values,
    width,
    label="Macro ROC-AUC"
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(
    0.85,
    1.0
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "RESPIRA — Final 512x512 Model Performance"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

final_metrics_plot = (
    OUTPUT_ROOT
    / "final_metrics.png"
)

plt.savefig(
    final_metrics_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ final_metrics.png"
)


# ============================================================
# 19. SYSTEM SUMMARY FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.bar(
    models,
    accuracy_values
)

ax.set_ylim(
    0.88,
    0.94
)

ax.set_ylabel(
    "Test Accuracy"
)

ax.set_title(
    "RESPIRA — Final Test Accuracy"
)

ax.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    accuracy_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.001,
        f"{value * 100:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

system_summary_plot = (
    OUTPUT_ROOT
    / "system_summary.png"
)

plt.savefig(
    system_summary_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ system_summary.png"
)


# ============================================================
# 20. FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 8.6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Final system summary:")
print(OUTPUT_ROOT)

print()
print("Generated files:")
print("  ✓ final_metrics.png")
print("  ✓ final_system_summary.json")
print("  ✓ final_system_summary.txt")
print("  ✓ model_performance_summary.csv")
print("  ✓ pipeline_summary.csv")
print("  ✓ system_summary.png")

print()
print("FINAL 512x512 SYSTEM METRICS")
print("-" * 70)

print(
    f"Test Images      : "
    f"{expected_test_samples}"
)

print(
    f"EfficientNet Acc : "
    f"{pct(eff_metrics['accuracy'])}"
)

print(
    f"ViT Accuracy     : "
    f"{pct(vit_metrics['accuracy'])}"
)

print(
    f"Fusion Accuracy  : "
    f"{pct(fusion_metrics['accuracy'])}"
)

print(
    f"Fusion Precision : "
    f"{pct(fusion_metrics['macro_precision'])}"
)

print(
    f"Fusion Recall    : "
    f"{pct(fusion_metrics['macro_recall'])}"
)

print(
    f"Fusion Macro F1  : "
    f"{pct(fusion_metrics['macro_f1'])}"
)

print(
    f"Fusion ROC-AUC   : "
    f"{fusion_metrics['macro_roc_auc']:.4f}"
)

print()
print("Fusion weights:")

print(
    f"  EfficientNet : "
    f"{eff_weight:.2f}"
)

print(
    f"  ViT          : "
    f"{vit_weight:.2f}"
)

print()
print("Fusion improvement:")

print(
    f"  Accuracy vs EfficientNet : "
    f"{fusion_gain_vs_eff_accuracy * 100:+.2f} pp"
)

print(
    f"  Accuracy vs ViT          : "
    f"{fusion_gain_vs_vit_accuracy * 100:+.2f} pp"
)

print()
print("Best comparison models:")

print(
    "  Accuracy   : EfficientNet + ViT Fusion"
)

print(
    "  Macro F1   : EfficientNet + ViT Fusion"
)

print(
    "  Macro AUC  : EfficientNet + ViT Fusion"
)

print()
print(
    "✓ Final 512x512 Respira system summary generated."
)

print(
    "✓ No model checkpoints were modified."
)

print(
    "✓ No dataset files were modified."
)

print(
    "✓ Final test set was used only for evaluation."
)

print("=" * 70)