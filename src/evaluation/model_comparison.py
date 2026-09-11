# ============================================================
# Respira - STEP 8.1
# Automated Model Comparison & Graphical Analysis
# ============================================================
#
# Purpose:
#   Compare all major Respira models using their existing
#   evaluation results.
#
# Models:
#   1. DenseNet121
#   2. EfficientNet-B0
#   3. ViT-B/16
#   4. Final Fusion
#
# Outputs:
#   outputs/final_analysis/model_comparison/
#
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

VIT_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
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

FINAL_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
    / "final_summary.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "model_comparison"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.1")
print("AUTOMATED MODEL COMPARISON & GRAPHICAL ANALYSIS")
print("=" * 70)

print()
print(f"Project root:")
print(PROJECT_ROOT)

print()
print(f"Output directory:")
print(OUTPUT_ROOT)


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


# ============================================================
# CHECK INPUT FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

input_files = {

    "DenseNet121":
        DENSENET_METRICS,

    "EfficientNet-B0":
        EFFICIENTNET_METRICS,

    "ViT-B/16":
        VIT_METRICS,

    "Final Fusion":
        FUSION_METRICS,

}


loaded_data = {}


for model_name, path in input_files.items():

    if path.exists():

        print(f"✓ {model_name}")
        print(f"  {path}")

        loaded_data[model_name] = load_json(path)

    else:

        print(f"✗ {model_name}")
        print(f"  Missing:")
        print(f"  {path}")


if len(loaded_data) < 2:

    raise RuntimeError(
        "\nNot enough model evaluation files "
        "were found for comparison."
    )


# ============================================================
# LOAD FINAL SUMMARY
# ============================================================

final_summary = None

if FINAL_SUMMARY.exists():

    final_summary = load_json(
        FINAL_SUMMARY
    )

    print()
    print("✓ Final prediction summary loaded")


# ============================================================
# EXTRACT METRICS
# ============================================================

print()
print("=" * 70)
print("EXTRACTING MODEL METRICS")
print("=" * 70)


metric_fields = [

    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_macro",

]


comparison_rows = []


for model_name, data in loaded_data.items():

    row = {

        "Model": model_name,

    }

    for metric in metric_fields:

        value = data.get(
            metric,
            np.nan
        )

        if value is None:

            value = np.nan

        row[metric] = value

    row["Test Images"] = data.get(
        "test_images",
        np.nan
    )

    comparison_rows.append(row)


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# CONVERT TO PERCENTAGE
# ============================================================

percentage_metrics = [

    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_macro",

]


# ============================================================
# SAVE RAW COMPARISON
# ============================================================

csv_path = (
    OUTPUT_ROOT
    / "model_comparison.csv"
)

comparison_df.to_csv(
    csv_path,
    index=False
)

print()
print("✓ Comparison table saved:")
print(csv_path)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print()
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

display_df = comparison_df.copy()

for column in percentage_metrics:

    if column in display_df.columns:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)


print()
print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# DETERMINE BEST MODELS
# ============================================================

best_models = {}


for metric in percentage_metrics:

    valid = comparison_df[
        comparison_df[metric].notna()
    ]

    if len(valid) == 0:

        continue

    best_row = valid.loc[
        valid[metric].idxmax()
    ]

    best_models[metric] = {

        "model":
            best_row["Model"],

        "value":
            float(best_row[metric]),

    }


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = {

    "stage":
        "STEP 8.1",

    "purpose":
        "Automated model comparison and graphical analysis",

    "models_compared":
        list(loaded_data.keys()),

    "best_models":
        best_models,

    "comparison": [],

}


for _, row in comparison_df.iterrows():

    model_result = {

        "model":
            row["Model"],

        "test_images":
            None
            if pd.isna(row["Test Images"])
            else int(row["Test Images"]),

    }

    for metric in percentage_metrics:

        value = row[metric]

        model_result[metric] = (

            None
            if pd.isna(value)
            else float(value)

        )

    summary["comparison"].append(
        model_result
    )


json_path = (
    OUTPUT_ROOT
    / "model_comparison.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print("✓ Comparison JSON saved:")
print(json_path)


# ============================================================
# GRAPH 1
# OVERALL ACCURACY
# ============================================================

print()
print("=" * 70)
print("GENERATING GRAPHS")
print("=" * 70)


models = comparison_df["Model"].tolist()

accuracy_values = (
    comparison_df["accuracy"] * 100
)


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    models,
    accuracy_values
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Respira Model Accuracy Comparison"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

accuracy_plot = (
    OUTPUT_ROOT
    / "accuracy_comparison.png"
)

plt.savefig(
    accuracy_plot,
    dpi=300
)

plt.close()

print(
    f"✓ Accuracy graph:\n  {accuracy_plot}"
)


# ============================================================
# GRAPH 2
# MACRO METRICS
# ============================================================

macro_metrics = {

    "Precision":
        "precision_macro",

    "Recall":
        "recall_macro",

    "F1 Score":
        "f1_macro",

    "ROC-AUC":
        "roc_auc_macro",

}


for title, column in macro_metrics.items():

    values = (
        comparison_df[column] * 100
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        models,
        values
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        f"Respira {title} Comparison"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    filename = (
        title.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_comparison.png"
    )

    output_path = (
        OUTPUT_ROOT
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"✓ {title} graph:\n  {output_path}"
    )


# ============================================================
# GRAPH 3
# ALL AVAILABLE METRICS
# ============================================================

available_columns = [

    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro",

]


plt.figure(
    figsize=(12, 7)
)

x = np.arange(
    len(models)
)

width = 0.15

for i, column in enumerate(
    available_columns
):

    values = (
        comparison_df[column]
        .fillna(0)
        * 100
    )

    plt.bar(
        x + i * width,
        values,
        width,
        label=column.replace(
            "_",
            " "
        ).title()
    )


plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Respira — Overall Model Performance"
)

plt.xticks(
    x + width * 2,
    models,
    rotation=15
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

all_metrics_plot = (
    OUTPUT_ROOT
    / "overall_model_comparison.png"
)

plt.savefig(
    all_metrics_plot,
    dpi=300
)

plt.close()

print()
print(
    f"✓ Overall comparison graph:\n"
    f"  {all_metrics_plot}"
)


# ============================================================
# FINAL ANALYSIS
# ============================================================

print()
print("=" * 70)
print("FINAL STEP 8.1 ANALYSIS")
print("=" * 70)


if "accuracy" in best_models:

    best_accuracy = best_models[
        "accuracy"
    ]

    print()
    print(
        f"Best Accuracy Model:"
    )

    print(
        f"  {best_accuracy['model']}"
    )

    print(
        f"  Accuracy: "
        f"{best_accuracy['value'] * 100:.2f}%"
    )


if "f1_macro" in best_models:

    best_f1 = best_models[
        "f1_macro"
    ]

    print()
    print(
        f"Best Macro F1 Model:"
    )

    print(
        f"  {best_f1['model']}"
    )

    print(
        f"  Macro F1: "
        f"{best_f1['value'] * 100:.2f}%"
    )


if "roc_auc_macro" in best_models:

    best_auc = best_models[
        "roc_auc_macro"
    ]

    print()
    print(
        f"Best Macro ROC-AUC Model:"
    )

    print(
        f"  {best_auc['model']}"
    )

    print(
        f"  ROC-AUC: "
        f"{best_auc['value'] * 100:.2f}%"
    )


# ============================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================

text_summary = (
    OUTPUT_ROOT
    / "comparison_summary.txt"
)


with open(
    text_summary,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — STEP 8.1\n"
    )

    f.write(
        "MODEL COMPARISON SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    for metric, result in best_models.items():

        f.write(
            f"{metric}:\n"
        )

        f.write(
            f"  Best model: "
            f"{result['model']}\n"
        )

        f.write(
            f"  Score: "
            f"{result['value'] * 100:.2f}%\n\n"
        )


print()
print(
    f"✓ Text summary saved:\n"
    f"  {text_summary}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 8.1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Model comparison outputs:")
print(
    OUTPUT_ROOT
)

print()
print("Generated files:")

for path in sorted(
    OUTPUT_ROOT.iterdir()
):

    if path.is_file():

        print(
            f"  ✓ {path.name}"
        )

print()
print(
    "Next stage:"
)

print(
    "STEP 8.2 — PER-CLASS PERFORMANCE ANALYSIS"
)

print("=" * 70)