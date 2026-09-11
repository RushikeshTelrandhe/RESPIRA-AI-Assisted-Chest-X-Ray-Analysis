# ============================================================
# Respira - STEP 8.5
# MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS
# ============================================================
#
# Compares:
#   1. DenseNet121
#   2. EfficientNet-B0
#   3. EfficientNet + ViT Fusion
#
# Uses previously generated metrics.json files.
#
# Output:
#   outputs/final_analysis/model_comparison
#
# ============================================================

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "model_comparison"
)

DENSENET_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
    / "evaluation"
    / "metrics.json"
)

EFFICIENTNET_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0"
    / "evaluation"
    / "metrics.json"
)

FUSION_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "classification"
    / "metrics"
    / "metrics.json"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("RESPIRA — STEP 8.5")
print("MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Output root  : {OUTPUT_ROOT}")


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics(path, model_name):

    if not path.exists():

        print()
        print(
            f"⚠ Metrics not found for {model_name}:"
        )

        print(path)

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"✓ {model_name} metrics loaded"
    )

    return data


print()
print("=" * 70)
print("LOADING MODEL METRICS")
print("=" * 70)

densenet = load_metrics(
    DENSENET_METRICS,
    "DenseNet121"
)

efficientnet = load_metrics(
    EFFICIENTNET_METRICS,
    "EfficientNet-B0"
)

fusion = load_metrics(
    FUSION_METRICS,
    "EfficientNet + ViT Fusion"
)


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

models = []


def add_model(
    model_name,
    data
):

    if data is None:
        return

    models.append({

        "Model":
            model_name,

        "Accuracy":
            data.get(
                "accuracy",
                None
            ),

        "Precision Macro":
            data.get(
                "precision_macro",
                None
            ),

        "Recall Macro":
            data.get(
                "recall_macro",
                None
            ),

        "F1 Macro":
            data.get(
                "f1_macro",
                None
            ),

        "Precision Weighted":
            data.get(
                "precision_weighted",
                None
            ),

        "Recall Weighted":
            data.get(
                "recall_weighted",
                None
            ),

        "F1 Weighted":
            data.get(
                "f1_weighted",
                None
            ),

        "ROC-AUC Macro":
            data.get(
                "roc_auc_macro",
                None
            ),

        "Test Images":
            data.get(
                "test_images",
                2072
            )
    })


add_model(
    "DenseNet121",
    densenet
)

add_model(
    "EfficientNet-B0",
    efficientnet
)

add_model(
    "EfficientNet + ViT Fusion",
    fusion
)


comparison = pd.DataFrame(
    models
)


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()

if not comparison.empty:

    display_columns = [
        "Model",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "ROC-AUC Macro"
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
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
print(
    f"✓ Saved: {csv_path}"
)


# ============================================================
# DETERMINE BEST MODELS
# ============================================================

best_accuracy = None
best_f1 = None
best_auc = None


if not comparison.empty:

    accuracy_series = comparison[
        "Accuracy"
    ].dropna()

    f1_series = comparison[
        "F1 Macro"
    ].dropna()

    auc_series = comparison[
        "ROC-AUC Macro"
    ].dropna()

    if not accuracy_series.empty:

        best_accuracy = comparison.loc[
            accuracy_series.idxmax(),
            "Model"
        ]

    if not f1_series.empty:

        best_f1 = comparison.loc[
            f1_series.idxmax(),
            "Model"
        ]

    if not auc_series.empty:

        best_auc = comparison.loc[
            auc_series.idxmax(),
            "Model"
        ]


# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

improvements = {}


def calculate_improvement(
    baseline,
    proposed,
    metric
):

    if (
        baseline is None
        or proposed is None
        or metric not in baseline
        or metric not in proposed
    ):

        return None

    baseline_value = baseline.get(
        metric
    )

    proposed_value = proposed.get(
        metric
    )

    if (
        baseline_value is None
        or proposed_value is None
    ):

        return None

    absolute = (
        proposed_value
        - baseline_value
    )

    relative = (
        absolute
        / baseline_value
        * 100
    )

    return {

        "baseline":
            baseline_value,

        "fusion":
            proposed_value,

        "absolute_improvement":
            absolute,

        "relative_improvement_percent":
            relative
    }


if (
    densenet is not None
    and fusion is not None
):

    for metric in [

        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro"

    ]:

        result = calculate_improvement(
            densenet,
            fusion,
            metric
        )

        if result is not None:

            improvements[
                f"Fusion_vs_DenseNet_{metric}"
            ] = result


if (
    efficientnet is not None
    and fusion is not None
):

    for metric in [

        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro"

    ]:

        result = calculate_improvement(
            efficientnet,
            fusion,
            metric
        )

        if result is not None:

            improvements[
                f"Fusion_vs_EfficientNet_{metric}"
            ] = result


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "STEP 8.5",

    "analysis":
        "Model Comparison & Final Performance Analysis",

    "models_compared":
        comparison[
            "Model"
        ].tolist()
        if not comparison.empty
        else [],

    "best_accuracy_model":
        best_accuracy,

    "best_macro_f1_model":
        best_f1,

    "best_macro_roc_auc_model":
        best_auc,

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
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print()
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
) as file:

    file.write(
        "RESPIRA — STEP 8.5\n"
    )

    file.write(
        "MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        "MODELS COMPARED\n"
    )

    for model in comparison[
        "Model"
    ].tolist():

        file.write(
            f"  - {model}\n"
        )

    file.write(
        "\nPERFORMANCE\n\n"
    )

    if not comparison.empty:

        file.write(
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
                index=False
            )
        )

        file.write(
            "\n\n"
        )

    file.write(
        f"Best Accuracy Model : "
        f"{best_accuracy}\n"
    )

    file.write(
        f"Best Macro F1 Model : "
        f"{best_f1}\n"
    )

    file.write(
        f"Best Macro ROC-AUC  : "
        f"{best_auc}\n"
    )

    file.write(
        "\n"
    )

    file.write(
        "FUSION IMPROVEMENTS\n"
    )

    for name, result in improvements.items():

        file.write(
            f"\n{name}\n"
        )

        file.write(
            f"  Baseline       : "
            f"{result['baseline']:.4f}\n"
        )

        file.write(
            f"  Fusion         : "
            f"{result['fusion']:.4f}\n"
        )

        file.write(
            f"  Absolute gain  : "
            f"{result['absolute_improvement']:.4f}\n"
        )

        file.write(
            f"  Relative gain  : "
            f"{result['relative_improvement_percent']:.2f}%\n"
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

    if comparison.empty:
        return

    values = comparison[
        column
    ]

    if values.isna().all():
        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        comparison["Model"],
        values
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
# INDIVIDUAL METRIC PLOTS
# ============================================================

print()
print("=" * 70)
print("GENERATING PERFORMANCE PLOTS")
print("=" * 70)

create_metric_plot(
    "Accuracy",
    "accuracy_comparison.png",
    "Model Accuracy Comparison"
)

create_metric_plot(
    "Precision Macro",
    "precision_comparison.png",
    "Macro Precision Comparison"
)

create_metric_plot(
    "Recall Macro",
    "recall_comparison.png",
    "Macro Recall Comparison"
)

create_metric_plot(
    "F1 Macro",
    "f1_comparison.png",
    "Macro F1 Score Comparison"
)

create_metric_plot(
    "ROC-AUC Macro",
    "roc_auc_comparison.png",
    "Macro ROC-AUC Comparison"
)


# ============================================================
# OVERALL COMPARISON PLOT
# ============================================================

if not comparison.empty:

    metrics = [
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "ROC-AUC Macro"
    ]

    x = range(
        len(comparison)
    )

    width = 0.15

    plt.figure(
        figsize=(13, 7)
    )

    for i, metric in enumerate(
        metrics
    ):

        offset = (
            i
            - len(metrics) / 2
        ) * width

        plt.bar(
            [
                value + offset
                for value in x
            ],
            comparison[
                metric
            ],
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
        "Overall Model Performance Comparison"
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
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("STEP 8.5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Model comparison outputs:"
)

print(
    OUTPUT_ROOT
)

print()
print(
    "Generated:"
)

for path in sorted(
    OUTPUT_ROOT.iterdir()
):

    if path.is_file():

        print(
            f"  ✓ {path.name}"
        )

print()
print(
    "Best Accuracy Model:"
)

print(
    f"  {best_accuracy}"
)

print()
print(
    "Best Macro F1 Model:"
)

print(
    f"  {best_f1}"
)

print()
print(
    "Best Macro ROC-AUC Model:"
)

print(
    f"  {best_auc}"
)

print()
print(
    "✓ STEP 8.5 COMPLETED."
)

print()
print(
    "Next stage:"
)

print(
    "STEP 8.6 — FINAL SYSTEM SUMMARY & REPORT GENERATION"
)