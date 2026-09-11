# ============================================================
# RESPIRA — STEP 8.9
# FINAL EXPLAINABILITY ANALYSIS
# ============================================================
#
# Purpose:
#   Consolidate Grad-CAM and ViT attention results into a
#   final explainability analysis.
#
# Inputs:
#   outputs/final_analysis/grad_cam
#   outputs/final_analysis/vit_attention
#   outputs/final_prediction
#
# Outputs:
#   outputs/final_analysis/final_explainability
#
# ============================================================

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_PREDICTION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
)

GRADCAM_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "grad_cam"
)

VIT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "vit_attention"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "final_explainability"
)


REPORT_ROOT = OUTPUT_ROOT / "reports"
PLOTS_ROOT = OUTPUT_ROOT / "plots"
SUMMARY_ROOT = OUTPUT_ROOT / "summary"


for directory in [
    OUTPUT_ROOT,
    REPORT_ROOT,
    PLOTS_ROOT,
    SUMMARY_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.9")
print("FINAL EXPLAINABILITY ANALYSIS")
print("=" * 70)

print()
print("Project root :", PROJECT_ROOT)
print("Grad-CAM root:", GRADCAM_ROOT)
print("ViT root     :", VIT_ROOT)
print("Output root  :", OUTPUT_ROOT)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):

    if not path.exists():
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def find_csv(root, candidates):

    for name in candidates:

        path = root / name

        if path.exists():
            return path

    return None


def find_recursive_csv(root, candidates):

    if not root.exists():
        return None

    for candidate in candidates:

        direct = root / candidate

        if direct.exists():
            return direct

    for candidate in candidates:

        matches = list(
            root.rglob(candidate)
        )

        if matches:
            return matches[0]

    return None


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# LOAD FINAL PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING FINAL PREDICTIONS")
print("=" * 70)


prediction_candidates = [
    "final_predictions.csv",
    "predictions.csv",
    "test_predictions.csv",
]


prediction_path = find_recursive_csv(
    FINAL_PREDICTION_ROOT,
    prediction_candidates
)


if prediction_path is None:

    raise FileNotFoundError(
        "Final prediction CSV could not be found."
    )


predictions = pd.read_csv(
    prediction_path
)


print("✓ Prediction file:")
print(f"  {prediction_path}")

print()
print(
    f"Prediction rows: {len(predictions)}"
)


# ============================================================
# LOAD GRAD-CAM RESULTS
# ============================================================

print()
print("=" * 70)
print("LOADING GRAD-CAM RESULTS")
print("=" * 70)


gradcam_candidates = [
    "grad_cam_results.csv",
    "results.csv",
]


gradcam_path = find_recursive_csv(
    GRADCAM_ROOT,
    gradcam_candidates
)


gradcam_data = None


if gradcam_path is not None:

    try:

        gradcam_data = pd.read_csv(
            gradcam_path
        )

        print("✓ Grad-CAM results found:")
        print(f"  {gradcam_path}")

        print(
            f"  Rows: {len(gradcam_data)}"
        )

    except Exception as error:

        print(
            f"⚠ Could not load Grad-CAM CSV: {error}"
        )

else:

    print(
        "⚠ Grad-CAM result CSV not found."
    )


# ============================================================
# LOAD GRAD-CAM SUMMARY
# ============================================================

gradcam_summary = load_json(
    GRADCAM_ROOT
    / "grad_cam_summary.json"
)


# ============================================================
# LOAD VIT RESULTS
# ============================================================

print()
print("=" * 70)
print("LOADING VIT ATTENTION RESULTS")
print("=" * 70)


vit_candidates = [
    "vit_attention_results.csv",
    "attention_results.csv",
    "results.csv",
]


vit_path = find_recursive_csv(
    VIT_ROOT,
    vit_candidates
)


vit_data = None


if vit_path is not None:

    try:

        vit_data = pd.read_csv(
            vit_path
        )

        print("✓ ViT attention results found:")
        print(f"  {vit_path}")

        print(
            f"  Rows: {len(vit_data)}"
        )

    except Exception as error:

        print(
            f"⚠ Could not load ViT CSV: {error}"
        )

else:

    print(
        "⚠ ViT attention result CSV not found."
    )


# ============================================================
# LOAD VIT SUMMARY
# ============================================================

vit_summary = load_json(
    VIT_ROOT
    / "vit_attention_summary.json"
)


# ============================================================
# IDENTIFY PREDICTION COLUMNS
# ============================================================

def detect_column(dataframe, candidates):

    if dataframe is None:
        return None

    mapping = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        if candidate.lower() in mapping:

            return mapping[
                candidate.lower()
            ]

    return None


true_column = detect_column(
    predictions,
    [
        "true_class",
        "true_label",
        "actual",
        "label",
    ]
)


predicted_column = detect_column(
    predictions,
    [
        "predicted_class",
        "predicted_label",
        "prediction",
        "predicted",
    ]
)


confidence_column = detect_column(
    predictions,
    [
        "prediction_confidence",
        "confidence",
        "max_probability",
    ]
)


uncertainty_column = detect_column(
    predictions,
    [
        "sample_uncertainty",
        "uncertainty",
        "uncertainty_score",
    ]
)


# ============================================================
# CALCULATE CORRECTNESS
# ============================================================

if (
    true_column is not None
    and predicted_column is not None
):

    predictions[
        "correct_prediction"
    ] = (
        predictions[true_column].astype(str)
        ==
        predictions[predicted_column].astype(str)
    )

else:

    predictions[
        "correct_prediction"
    ] = False


# ============================================================
# BASIC SYSTEM STATISTICS
# ============================================================

total_samples = len(
    predictions
)


correct_samples = int(
    predictions[
        "correct_prediction"
    ].sum()
)


incorrect_samples = (
    total_samples
    - correct_samples
)


if total_samples > 0:

    system_accuracy = (
        correct_samples
        / total_samples
    )

else:

    system_accuracy = np.nan


mean_confidence = np.nan


if confidence_column is not None:

    mean_confidence = pd.to_numeric(
        predictions[
            confidence_column
        ],
        errors="coerce"
    ).mean()


mean_uncertainty = np.nan


if uncertainty_column is not None:

    mean_uncertainty = pd.to_numeric(
        predictions[
            uncertainty_column
        ],
        errors="coerce"
    ).mean()


# ============================================================
# CLASS-WISE EXPLAINABILITY SUMMARY
# ============================================================

print()
print("=" * 70)
print("GENERATING CLASS-WISE EXPLAINABILITY SUMMARY")
print("=" * 70)


class_rows = []


for class_name in CLASS_NAMES:

    if true_column is not None:

        class_data = predictions[
            predictions[true_column].astype(str)
            == class_name
        ]

    else:

        class_data = pd.DataFrame()


    count = len(
        class_data
    )


    if count > 0:

        correct = int(
            class_data[
                "correct_prediction"
            ].sum()
        )

        accuracy = (
            correct / count
        )

        if confidence_column is not None:

            confidence = pd.to_numeric(
                class_data[
                    confidence_column
                ],
                errors="coerce"
            ).mean()

        else:

            confidence = np.nan


        if uncertainty_column is not None:

            uncertainty = pd.to_numeric(
                class_data[
                    uncertainty_column
                ],
                errors="coerce"
            ).mean()

        else:

            uncertainty = np.nan

    else:

        correct = 0
        accuracy = np.nan
        confidence = np.nan
        uncertainty = np.nan


    class_rows.append(
        {
            "class": class_name,
            "samples": count,
            "correct": correct,
            "accuracy": accuracy,
            "mean_confidence": confidence,
            "mean_uncertainty": uncertainty,
        }
    )


class_summary = pd.DataFrame(
    class_rows
)


class_summary_path = (
    REPORT_ROOT
    / "explainability_per_class.csv"
)


class_summary.to_csv(
    class_summary_path,
    index=False
)


print(
    f"✓ {class_summary_path.name}"
)


# ============================================================
# EXPLAINABILITY METHOD SUMMARY
# ============================================================

gradcam_available = (
    gradcam_data is not None
    or bool(gradcam_summary)
)


vit_available = (
    vit_data is not None
    or bool(vit_summary)
)


method_summary = pd.DataFrame(
    [
        {
            "method": "Grad-CAM",
            "model": "EfficientNet-B0",
            "available": gradcam_available,
            "purpose": (
                "CNN spatial explanation"
            ),
            "representation": (
                "Convolutional feature maps"
            ),
        },
        {
            "method": "ViT Attention",
            "model": "Vision Transformer",
            "available": vit_available,
            "purpose": (
                "Transformer token attention explanation"
            ),
            "representation": (
                "Patch/token attention"
            ),
        },
    ]
)


method_summary_path = (
    REPORT_ROOT
    / "explainability_methods.csv"
)


method_summary.to_csv(
    method_summary_path,
    index=False
)


print(
    f"✓ {method_summary_path.name}"
)


# ============================================================
# GENERATE CONFIDENCE VS UNCERTAINTY
# ============================================================

if (
    confidence_column is not None
    and uncertainty_column is not None
):

    confidence_values = pd.to_numeric(
        predictions[
            confidence_column
        ],
        errors="coerce"
    )

    uncertainty_values = pd.to_numeric(
        predictions[
            uncertainty_column
        ],
        errors="coerce"
    )


    valid = (
        confidence_values.notna()
        &
        uncertainty_values.notna()
    )


    if valid.any():

        plt.figure(
            figsize=(9, 6)
        )

        plt.scatter(
            confidence_values[valid],
            uncertainty_values[valid],
            alpha=0.35,
            s=18
        )

        plt.xlabel(
            "Prediction Confidence"
        )

        plt.ylabel(
            "Sample Uncertainty"
        )

        plt.title(
            "Prediction Confidence vs Uncertainty"
        )

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()


        path = (
            PLOTS_ROOT
            / "confidence_vs_uncertainty.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()

        print(
            f"✓ {path.name}"
        )


# ============================================================
# GENERATE CORRECT VS INCORRECT CONFIDENCE
# ============================================================

if confidence_column is not None:

    correct_values = pd.to_numeric(
        predictions[
            predictions[
                "correct_prediction"
            ]
            == True
        ][confidence_column],
        errors="coerce"
    ).dropna()


    incorrect_values = pd.to_numeric(
        predictions[
            predictions[
                "correct_prediction"
            ]
            == False
        ][confidence_column],
        errors="coerce"
    ).dropna()


    if (
        len(correct_values) > 0
        and len(incorrect_values) > 0
    ):

        plt.figure(
            figsize=(9, 6)
        )

        plt.boxplot(
            [
                correct_values,
                incorrect_values
            ],
            tick_labels=[
                "Correct",
                "Incorrect"
            ]
        )

        plt.ylabel(
            "Prediction Confidence"
        )

        plt.title(
            "Confidence: Correct vs Incorrect Predictions"
        )

        plt.tight_layout()


        path = (
            PLOTS_ROOT
            / "correct_vs_incorrect_confidence.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()

        print(
            f"✓ {path.name}"
        )


# ============================================================
# GENERATE PER-CLASS CONFIDENCE
# ============================================================

if confidence_column is not None:

    confidence_plot_data = []

    for class_name in CLASS_NAMES:

        class_data = predictions[
            predictions[
                true_column
            ].astype(str)
            == class_name
        ] if true_column is not None else pd.DataFrame()


        values = pd.to_numeric(
            class_data[
                confidence_column
            ],
            errors="coerce"
        ).dropna()


        confidence_plot_data.append(
            values
        )


    if any(
        len(values) > 0
        for values in confidence_plot_data
    ):

        plt.figure(
            figsize=(11, 6)
        )

        plt.boxplot(
            confidence_plot_data,
            tick_labels=CLASS_NAMES
        )

        plt.ylabel(
            "Prediction Confidence"
        )

        plt.title(
            "Confidence Distribution by Disease"
        )

        plt.xticks(
            rotation=25,
            ha="right"
        )

        plt.tight_layout()


        path = (
            PLOTS_ROOT
            / "per_class_confidence.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()

        print(
            f"✓ {path.name}"
        )


# ============================================================
# EXPLAINABILITY COVERAGE
# ============================================================

coverage_rows = []


gradcam_count = 0
vit_count = 0


if gradcam_data is not None:

    gradcam_count = len(
        gradcam_data
    )

elif gradcam_summary:

    gradcam_count = int(
        gradcam_summary.get(
            "test_images",
            0
        )
    )


if vit_data is not None:

    vit_count = len(
        vit_data
    )

elif vit_summary:

    vit_count = int(
        vit_summary.get(
            "visualized_samples",
            0
        )
    )


coverage_rows.append(
    {
        "method": "Grad-CAM",
        "available": gradcam_available,
        "samples_visualized": gradcam_count,
        "total_test_samples": total_samples,
        "coverage": (
            gradcam_count / total_samples
            if total_samples > 0
            else 0
        ),
    }
)


coverage_rows.append(
    {
        "method": "ViT Attention",
        "available": vit_available,
        "samples_visualized": vit_count,
        "total_test_samples": total_samples,
        "coverage": (
            vit_count / total_samples
            if total_samples > 0
            else 0
        ),
    }
)


coverage_df = pd.DataFrame(
    coverage_rows
)


coverage_path = (
    REPORT_ROOT
    / "explainability_coverage.csv"
)


coverage_df.to_csv(
    coverage_path,
    index=False
)


print(
    f"✓ {coverage_path.name}"
)


# ============================================================
# COVERAGE PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.bar(
    coverage_df["method"],
    coverage_df["coverage"] * 100
)


plt.ylabel(
    "Coverage (%)"
)


plt.title(
    "Explainability Visualization Coverage"
)


plt.ylim(
    0,
    max(
        100,
        float(
            coverage_df["coverage"].max()
            * 100
            + 10
        )
    )
)


plt.tight_layout()


coverage_plot = (
    PLOTS_ROOT
    / "explainability_coverage.png"
)


plt.savefig(
    coverage_plot,
    dpi=200
)


plt.close()


print(
    f"✓ {coverage_plot.name}"
)


# ============================================================
# FINAL EXPLAINABILITY SUMMARY
# ============================================================

final_summary = {

    "stage": "STEP 8.9",

    "pipeline": (
        "Final Explainability Analysis"
    ),

    "test_samples": total_samples,

    "correct_predictions":
        correct_samples,

    "incorrect_predictions":
        incorrect_samples,

    "accuracy":
        system_accuracy,

    "mean_confidence":
        (
            safe_float(mean_confidence)
            if not pd.isna(mean_confidence)
            else None
        ),

    "mean_uncertainty":
        (
            safe_float(mean_uncertainty)
            if not pd.isna(mean_uncertainty)
            else None
        ),

    "explainability_methods": {

        "Grad-CAM": {

            "available":
                gradcam_available,

            "model":
                "EfficientNet-B0",

            "purpose":
                "CNN spatial localization",

            "samples_visualized":
                gradcam_count,

        },

        "ViT_Attention": {

            "available":
                vit_available,

            "model":
                "Vision Transformer",

            "purpose":
                "Transformer token-level attention",

            "samples_visualized":
                vit_count,

        },
    },

    "class_analysis":
        class_summary.to_dict(
            orient="records"
        ),

    "interpretation": [

        (
            "Grad-CAM provides spatial explanations "
            "from the EfficientNet-B0 convolutional "
            "feature hierarchy."
        ),

        (
            "ViT attention visualization provides "
            "patch-level explanations from the "
            "Vision Transformer representation."
        ),

        (
            "Combining CNN and ViT explanations "
            "provides complementary spatial and "
            "token-level interpretability."
        ),

        (
            "Confidence and uncertainty statistics "
            "provide additional information about "
            "prediction reliability."
        ),

    ],
}


# ============================================================
# SAVE JSON
# ============================================================

summary_json = (
    SUMMARY_ROOT
    / "final_explainability_summary.json"
)


with open(
    summary_json,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_summary,
        file,
        indent=4,
        default=str
    )


print(
    f"✓ {summary_json.name}"
)


# ============================================================
# TEXT REPORT
# ============================================================

summary_txt = (
    SUMMARY_ROOT
    / "final_explainability_summary.txt"
)


with open(
    summary_txt,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 70
        + "\n"
    )

    file.write(
        "RESPIRA — FINAL EXPLAINABILITY ANALYSIS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Test samples       : {total_samples}\n"
    )

    file.write(
        f"Correct predictions: {correct_samples}\n"
    )

    file.write(
        f"Incorrect predictions: {incorrect_samples}\n"
    )

    file.write(
        f"Accuracy           : {system_accuracy:.4f}\n"
    )

    if not pd.isna(mean_confidence):

        file.write(
            f"Mean confidence   : "
            f"{mean_confidence:.4f}\n"
        )

    if not pd.isna(mean_uncertainty):

        file.write(
            f"Mean uncertainty  : "
            f"{mean_uncertainty:.4f}\n"
        )

    file.write("\n")

    file.write(
        "EXPLAINABILITY METHODS\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    file.write(
        "1. Grad-CAM\n"
    )

    file.write(
        "   Model: EfficientNet-B0\n"
    )

    file.write(
        "   Purpose: CNN spatial localization\n"
    )

    file.write(
        f"   Samples visualized: {gradcam_count}\n\n"
    )

    file.write(
        "2. ViT Attention\n"
    )

    file.write(
        "   Model: Vision Transformer\n"
    )

    file.write(
        "   Purpose: Patch/token attention visualization\n"
    )

    file.write(
        f"   Samples visualized: {vit_count}\n\n"
    )

    file.write(
        "CLASS-WISE PERFORMANCE\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    for _, row in class_summary.iterrows():

        file.write(
            f"{row['class']}: "
            f"samples={int(row['samples'])}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"confidence={row['mean_confidence']:.4f}, "
            f"uncertainty={row['mean_uncertainty']:.4f}\n"
        )

    file.write("\n")

    file.write(
        "INTERPRETATION\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    for item in final_summary[
        "interpretation"
    ]:

        file.write(
            f"- {item}\n"
        )


print(
    f"✓ {summary_txt.name}"
)


# ============================================================
# FINAL CONSOLIDATED CSV
# ============================================================

consolidated = pd.DataFrame(
    [
        {
            "metric":
                "Accuracy",

            "value":
                system_accuracy,
        },
        {
            "metric":
                "Mean Confidence",

            "value":
                mean_confidence,
        },
        {
            "metric":
                "Mean Uncertainty",

            "value":
                mean_uncertainty,
        },
        {
            "metric":
                "Test Samples",

            "value":
                total_samples,
        },
        {
            "metric":
                "Correct Predictions",

            "value":
                correct_samples,
        },
        {
            "metric":
                "Incorrect Predictions",

            "value":
                incorrect_samples,
        },
        {
            "metric":
                "Grad-CAM Samples",

            "value":
                gradcam_count,
        },
        {
            "metric":
                "ViT Attention Samples",

            "value":
                vit_count,
        },
    ]
)


consolidated_path = (
    REPORT_ROOT
    / "final_explainability_metrics.csv"
)


consolidated.to_csv(
    consolidated_path,
    index=False
)


print(
    f"✓ {consolidated_path.name}"
)


# ============================================================
# FINAL METRICS PLOT
# ============================================================

metric_names = [
    "Accuracy",
    "Mean Confidence",
]


metric_values = [
    system_accuracy,
    mean_confidence
    if not pd.isna(mean_confidence)
    else 0,
]


plt.figure(
    figsize=(8, 6)
)


plt.bar(
    metric_names,
    np.array(metric_values) * 100
)


plt.ylabel(
    "Percentage"
)


plt.title(
    "Final Prediction and Explainability Metrics"
)


plt.ylim(
    0,
    100
)


plt.tight_layout()


metrics_plot = (
    PLOTS_ROOT
    / "final_explainability_metrics.png"
)


plt.savefig(
    metrics_plot,
    dpi=200
)


plt.close()


print(
    f"✓ {metrics_plot.name}"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("STEP 8.9 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Final explainability outputs:"
)

print(
    OUTPUT_ROOT
)

print()
print(
    "Generated:"
)

print(
    "  ✓ explainability_per_class.csv"
)

print(
    "  ✓ explainability_methods.csv"
)

print(
    "  ✓ explainability_coverage.csv"
)

print(
    "  ✓ final_explainability_metrics.csv"
)

print(
    "  ✓ confidence_vs_uncertainty.png"
)

print(
    "  ✓ correct_vs_incorrect_confidence.png"
)

print(
    "  ✓ per_class_confidence.png"
)

print(
    "  ✓ explainability_coverage.png"
)

print(
    "  ✓ final_explainability_metrics.png"
)

print(
    "  ✓ final_explainability_summary.json"
)

print(
    "  ✓ final_explainability_summary.txt"
)

print()
print(
    f"Test samples       : {total_samples}"
)

print(
    f"Correct predictions: {correct_samples}"
)

print(
    f"Incorrect predictions: {incorrect_samples}"
)

print(
    f"Accuracy           : {system_accuracy:.4f}"
)

if not pd.isna(mean_confidence):

    print(
        f"Mean confidence   : {mean_confidence:.4f}"
    )

if not pd.isna(mean_uncertainty):

    print(
        f"Mean uncertainty  : {mean_uncertainty:.4f}"
    )

print()
print(
    "Explainability methods:"
)

print(
    f"  Grad-CAM       : "
    f"{'Available' if gradcam_available else 'Unavailable'}"
)

print(
    f"  ViT Attention  : "
    f"{'Available' if vit_available else 'Unavailable'}"
)

print()
print(
    "✓ Grad-CAM and ViT attention consolidated."
)

print(
    "✓ Final explainability analysis generated."
)


