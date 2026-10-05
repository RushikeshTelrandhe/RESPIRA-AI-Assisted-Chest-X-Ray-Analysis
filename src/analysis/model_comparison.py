# ============================================================
# RESPIRA — STEP 8.5
# 512x512 MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS
# ============================================================

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "model_comparison"
)

EFFICIENTNET_METRICS = (
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

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.5")
print("512x512 MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Output root  : {OUTPUT_ROOT}")


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

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


print()
print("=" * 70)
print("LOADING 512x512 MODEL METRICS")
print("=" * 70)

efficientnet = load_json(
    EFFICIENTNET_METRICS
)

print("✓ EfficientNet-B0 512x512 metrics loaded")
print(f"  Source: {EFFICIENTNET_METRICS}")

vit = load_json(
    VIT_METRICS
)

print("✓ ViT-B/16 512x512 metrics loaded")
print(f"  Source: {VIT_METRICS}")

fusion = load_json(
    FUSION_METRICS
)

print("✓ EfficientNet + ViT Fusion metrics loaded")
print(f"  Source: {FUSION_METRICS}")


# ============================================================
# LOAD FUSION CONFIG
# ============================================================

fusion_config = {}

if FUSION_CONFIG.exists():

    fusion_config = load_json(
        FUSION_CONFIG
    )

    print()
    print("✓ Fusion configuration loaded")


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "Model":
            "EfficientNet-B0 512x512",

        "Accuracy":
            efficientnet["accuracy"],

        "Precision Macro":
            efficientnet["macro_precision"],

        "Recall Macro":
            efficientnet["macro_recall"],

        "F1 Macro":
            efficientnet["macro_f1"],

        "Precision Weighted":
            efficientnet["weighted_precision"],

        "Recall Weighted":
            efficientnet["weighted_recall"],

        "F1 Weighted":
            efficientnet["weighted_f1"],

        "ROC-AUC Macro":
            efficientnet["macro_roc_auc"],

        "Test Samples":
            efficientnet["test_samples"]
    },

    {
        "Model":
            "ViT-B/16 512x512",

        "Accuracy":
            vit["accuracy"],

        "Precision Macro":
            vit["macro_precision"],

        "Recall Macro":
            vit["macro_recall"],

        "F1 Macro":
            vit["macro_f1"],

        "Precision Weighted":
            vit["weighted_precision"],

        "Recall Weighted":
            vit["weighted_recall"],

        "F1 Weighted":
            vit["weighted_f1"],

        "ROC-AUC Macro":
            vit["macro_roc_auc"],

        "Test Samples":
            vit["test_samples"]
    },

    {
        "Model":
            "EfficientNet + ViT Fusion",

        "Accuracy":
            fusion["fusion_test_accuracy"],

        "Precision Macro":
            fusion["macro_precision"],

        "Recall Macro":
            fusion["macro_recall"],

        "F1 Macro":
            fusion["macro_f1"],

        "Precision Weighted":
            fusion["weighted_precision"],

        "Recall Weighted":
            fusion["weighted_recall"],

        "F1 Weighted":
            fusion["weighted_f1"],

        "ROC-AUC Macro":
            fusion["macro_roc_auc"],

        "Test Samples":
            fusion["test_samples"]
    }
])


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("512x512 MODEL COMPARISON")
print("=" * 70)
print()

print(
    comparison[
        [
            "Model",
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "ROC-AUC Macro"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_ROOT
    / "model_comparison.csv"
)

comparison.to_csv(
    csv_path,
    index=False
)

print()
print(f"✓ Saved: {csv_path}")


# ============================================================
# BEST MODELS
# ============================================================

best_accuracy_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

best_f1_model = comparison.loc[
    comparison["F1 Macro"].idxmax(),
    "Model"
]

best_auc_model = comparison.loc[
    comparison["ROC-AUC Macro"].idxmax(),
    "Model"
]


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

def improvement(
    baseline,
    proposed
):

    return {
        "absolute":
            proposed - baseline,

        "percentage_points":
            (proposed - baseline) * 100
    }


improvements = {

    "fusion_vs_efficientnet": {

        "accuracy":
            improvement(
                efficientnet["accuracy"],
                fusion["fusion_test_accuracy"]
            ),

        "macro_precision":
            improvement(
                efficientnet["macro_precision"],
                fusion["macro_precision"]
            ),

        "macro_recall":
            improvement(
                efficientnet["macro_recall"],
                fusion["macro_recall"]
            ),

        "macro_f1":
            improvement(
                efficientnet["macro_f1"],
                fusion["macro_f1"]
            ),

        "macro_roc_auc":
            improvement(
                efficientnet["macro_roc_auc"],
                fusion["macro_roc_auc"]
            )
    },

    "fusion_vs_vit": {

        "accuracy":
            improvement(
                vit["accuracy"],
                fusion["fusion_test_accuracy"]
            ),

        "macro_precision":
            improvement(
                vit["macro_precision"],
                fusion["macro_precision"]
            ),

        "macro_recall":
            improvement(
                vit["macro_recall"],
                fusion["macro_recall"]
            ),

        "macro_f1":
            improvement(
                vit["macro_f1"],
                fusion["macro_f1"]
            ),

        "macro_roc_auc":
            improvement(
                vit["macro_roc_auc"],
                fusion["macro_roc_auc"]
            )
    }
}


# ============================================================
# FUSION WEIGHTS
# ============================================================

fusion_weights = {
    "efficientnet_weight":
        fusion["efficientnet_weight"],

    "vit_weight":
        fusion["vit_weight"]
}


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "STEP 8.5",

    "experiment":
        "512x512 final model comparison",

    "test_samples":
        fusion["test_samples"],

    "models":
        comparison.to_dict(
            orient="records"
        ),

    "best_accuracy_model":
        best_accuracy_model,

    "best_macro_f1_model":
        best_f1_model,

    "best_macro_roc_auc_model":
        best_auc_model,

    "fusion_weights":
        fusion_weights,

    "validation_fusion_accuracy":
        fusion["validation_fusion_accuracy"],

    "validation_fusion_macro_f1":
        fusion["validation_fusion_macro_f1"],

    "improvements":
        improvements
}


summary_path = (
    OUTPUT_ROOT
    / "model_comparison_summary.json"
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

print(
    f"✓ Saved: {summary_path}"
)


# ============================================================
# TEXT SUMMARY
# ============================================================

text_path = (
    OUTPUT_ROOT
    / "model_comparison_summary.txt"
)

with open(
    text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — STEP 8.5\n"
    )

    f.write(
        "512x512 MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Test samples: 1947\n\n"
    )

    f.write(
        comparison.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Best Accuracy Model: "
        f"{best_accuracy_model}\n"
    )

    f.write(
        f"Best Macro F1 Model: "
        f"{best_f1_model}\n"
    )

    f.write(
        f"Best Macro ROC-AUC Model: "
        f"{best_auc_model}\n"
    )

    f.write(
        "\nFusion weights:\n"
    )

    f.write(
        f"EfficientNet: "
        f"{fusion['efficientnet_weight']}\n"
    )

    f.write(
        f"ViT: "
        f"{fusion['vit_weight']}\n"
    )

    f.write(
        "\nValidation fusion performance:\n"
    )

    f.write(
        f"Accuracy: "
        f"{fusion['validation_fusion_accuracy']:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{fusion['validation_fusion_macro_f1']:.6f}\n"
    )

    f.write(
        "\nFinal test fusion performance:\n"
    )

    f.write(
        f"Accuracy: "
        f"{fusion['fusion_test_accuracy']:.6f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{fusion['macro_precision']:.6f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{fusion['macro_recall']:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{fusion['macro_f1']:.6f}\n"
    )

    f.write(
        f"Macro ROC-AUC: "
        f"{fusion['macro_roc_auc']:.6f}\n"
    )


print(
    f"✓ Saved: {text_path}"
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def create_metric_plot(
    column,
    filename,
    title
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        comparison["Model"],
        comparison[column]
    )

    plt.ylabel(
        column
    )

    plt.title(
        title
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
        / filename
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"✓ {filename}"
    )


# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

print()
print("=" * 70)
print("GENERATING PERFORMANCE PLOTS")
print("=" * 70)

create_metric_plot(
    "Accuracy",
    "accuracy_comparison.png",
    "512x512 Model Accuracy Comparison"
)

create_metric_plot(
    "Precision Macro",
    "precision_comparison.png",
    "512x512 Macro Precision Comparison"
)

create_metric_plot(
    "Recall Macro",
    "recall_comparison.png",
    "512x512 Macro Recall Comparison"
)

create_metric_plot(
    "F1 Macro",
    "f1_comparison.png",
    "512x512 Macro F1 Score Comparison"
)

create_metric_plot(
    "ROC-AUC Macro",
    "roc_auc_comparison.png",
    "512x512 Macro ROC-AUC Comparison"
)


# ============================================================
# OVERALL PLOT
# ============================================================

metrics = [
    "Accuracy",
    "Precision Macro",
    "Recall Macro",
    "F1 Macro",
    "ROC-AUC Macro"
]

plt.figure(
    figsize=(13, 7)
)

x = range(
    len(comparison)
)

width = 0.15

for i, metric in enumerate(metrics):

    offset = (
        i - 2
    ) * width

    plt.bar(
        [
            value + offset
            for value in x
        ],
        comparison[metric],
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    comparison["Model"],
    rotation=15,
    ha="right"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Overall 512x512 Model Performance Comparison"
)

plt.legend()

plt.tight_layout()

overall_path = (
    OUTPUT_ROOT
    / "overall_model_comparison.png"
)

plt.savefig(
    overall_path,
    dpi=200
)

plt.close()

print(
    f"✓ {overall_path.name}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 8.5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Model comparison outputs:")
print(OUTPUT_ROOT)

print()
print("Best Accuracy Model:")
print(
    f"  {best_accuracy_model}"
)

print()
print("Best Macro F1 Model:")
print(
    f"  {best_f1_model}"
)

print()
print("Best Macro ROC-AUC Model:")
print(
    f"  {best_auc_model}"
)

print()
print("Fusion weights:")
print(
    f"  EfficientNet : "
    f"{fusion['efficientnet_weight']:.2f}"
)

print(
    f"  ViT          : "
    f"{fusion['vit_weight']:.2f}"
)

print()
print("Final Fusion Test Accuracy:")
print(
    f"  {fusion['fusion_test_accuracy'] * 100:.2f}%"
)

print()
print("✓ STEP 8.5 COMPLETED.")

print()
print("Next stage:")
print(
    "STEP 8.6 — FINAL SYSTEM SUMMARY & REPORT GENERATION"
)