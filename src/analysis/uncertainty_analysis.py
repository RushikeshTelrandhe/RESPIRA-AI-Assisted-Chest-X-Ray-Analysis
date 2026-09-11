# ============================================================
# Respira - STEP 8.3
# UNCERTAINTY & CONFIDENCE ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze confidence and uncertainty of the final Respira
#   prediction pipeline.
#
# Input:
#   outputs/final_prediction/
#
# Output:
#   outputs/final_analysis/uncertainty_analysis/
#
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "uncertainty_analysis"
)

PREDICTION_ROOT = (
    FINAL_ROOT
    / "predictions"
)

UNCERTAINTY_ROOT = (
    FINAL_ROOT
    / "uncertainty"
)

REPORT_ROOT = (
    FINAL_ROOT
    / "reports"
)

SUMMARY_PATH = (
    FINAL_ROOT
    / "final_summary.json"
)

# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

PREDICTIONS_PATH = (
    PREDICTION_ROOT
    / "final_predictions.csv"
)

UNCERTAINTY_PATH = (
    UNCERTAINTY_ROOT
    / "uncertainty_scores.csv"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_ROOT
    / "classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    REPORT_ROOT
    / "confusion_matrix.csv"
)

# ============================================================
# CLASSES
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
# DIRECTORY SETUP
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.3")
print("UNCERTAINTY & CONFIDENCE ANALYSIS")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Input root   : {FINAL_ROOT}")
print(f"Output root  : {OUTPUT_ROOT}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_file(root, names):
    """
    Recursively search for the first matching file.
    """

    for name in names:

        direct = root / name

        if direct.exists():
            return direct

        matches = list(root.rglob(name))

        if matches:
            return matches[0]

    return None


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


# ============================================================
# LOAD FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("LOADING FINAL SUMMARY")
print("=" * 70)

if not SUMMARY_PATH.exists():

    raise FileNotFoundError(
        f"\nFinal summary not found:\n"
        f"{SUMMARY_PATH}"
    )

final_summary = load_json(
    SUMMARY_PATH
)

print("✓ Final summary loaded")


# ============================================================
# DISPLAY EXISTING SUMMARY
# ============================================================

print()
print("Existing final-pipeline statistics:")

print(
    f"  Test images        : "
    f"{final_summary.get('test_images', 'N/A')}"
)

print(
    f"  Accuracy            : "
    f"{final_summary.get('accuracy', 0):.4f}"
)

print(
    f"  Mean confidence     : "
    f"{final_summary.get('mean_confidence', 0):.4f}"
)

print(
    f"  Mean uncertainty    : "
    f"{final_summary.get('mean_uncertainty', 0):.4f}"
)

print(
    f"  Median uncertainty  : "
    f"{final_summary.get('median_uncertainty', 0):.4f}"
)


# ============================================================
# FIND PREDICTION FILE
# ============================================================

print()
print("=" * 70)
print("LOCATING PREDICTION DATA")
print("=" * 70)

prediction_candidates = [
    "final_predictions.csv",
    "test_predictions.csv",
    "predictions.csv",
]

prediction_path = find_file(
    PREDICTION_ROOT,
    prediction_candidates
)

if prediction_path is None:

    prediction_path = find_file(
        FINAL_ROOT,
        prediction_candidates
    )

if prediction_path is None:

    raise FileNotFoundError(
        f"\nPrediction CSV could not be found.\n"
        f"Expected location:\n"
        f"{PREDICTION_ROOT / 'final_predictions.csv'}"
    )

print(f"✓ Prediction CSV found:")
print(
    f"✓ Prediction file:\n"
    f"  {prediction_path}"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    prediction_path
)

print()
print(
    f"Prediction shape: "
    f"{predictions.shape}"
)

print(
    "Prediction columns:"
)

for column in predictions.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# FIND UNCERTAINTY DATA
# ============================================================

print()
print("=" * 70)
print("LOCATING UNCERTAINTY DATA")
print("=" * 70)

uncertainty_csv = find_file(
    UNCERTAINTY_ROOT,
    [
        "uncertainty.csv",
        "uncertainty_scores.csv",
        "test_uncertainty.csv",
    ]
)

if uncertainty_csv is not None:

    uncertainty_df = pd.read_csv(
        uncertainty_csv
    )

    print(
        f"✓ Uncertainty CSV loaded:\n"
        f"  {uncertainty_csv}"
    )

else:

    uncertainty_df = None

    print(
        "⚠ No separate uncertainty CSV found."
    )


# ============================================================
# MERGE UNCERTAINTY DATA IF AVAILABLE
# ============================================================

if uncertainty_df is not None:

    common_columns = list(
        set(predictions.columns)
        & set(uncertainty_df.columns)
    )

    if common_columns:

        try:

            predictions = predictions.merge(
                uncertainty_df,
                on=common_columns,
                how="left",
                suffixes=("", "_uncertainty")
            )

            print(
                "✓ Prediction and uncertainty "
                "data merged."
            )

        except Exception:

            print(
                "⚠ Could not merge uncertainty "
                "data. Continuing."
            )


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

def find_column(
    dataframe,
    candidates
):

    lower_map = {
        column.lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


uncertainty_column = find_column(
    predictions,
    [
        "sample_uncertainty",
        "uncertainty",
        "uncertainty_score",
        "mean_uncertainty",
        "entropy",
    ]
)

confidence_column = find_column(
    predictions,
    [
        "confidence",
        "max_probability",
        "prediction_confidence",
    ]
)

true_column = find_column(
    predictions,
    [
        "true_label",
        "true_class",
        "actual",
        "label",
        "target",
    ]
)

predicted_column = find_column(
    predictions,
    [
        "predicted_label",
        "predicted_class",
        "prediction",
        "predicted",
    ]
)


# ============================================================
# FALLBACK: USE FINAL SUMMARY
# ============================================================

if uncertainty_column is None:

    print()
    print(
        "⚠ Per-sample uncertainty column "
        "not found."
    )

    print(
        "Using summary-level uncertainty "
        "statistics where possible."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

if uncertainty_column is not None:

    predictions[
        uncertainty_column
    ] = pd.to_numeric(
        predictions[
            uncertainty_column
        ],
        errors="coerce"
    )

if confidence_column is not None:

    predictions[
        confidence_column
    ] = pd.to_numeric(
        predictions[
            confidence_column
        ],
        errors="coerce"
    )


# ============================================================
# CREATE CORRECTNESS COLUMN
# ============================================================

correct_column = None

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

    correct_column = "correct_prediction"

    print()
    print(
        "✓ Correctness calculated from "
        "true/predicted labels."
    )

elif "correct" in [
    c.lower()
    for c in predictions.columns
]:

    correct_column = [
        c
        for c in predictions.columns
        if c.lower() == "correct"
    ][0]

    predictions[
        "correct_prediction"
    ] = predictions[
        correct_column
    ].astype(bool)

    correct_column = "correct_prediction"


# ============================================================
# UNCERTAINTY CATEGORIES
# ============================================================

if uncertainty_column is not None:

    # Expected interpretation from Step 7.8:
    #
    # Low       : uncertainty < 0.20
    # Moderate  : 0.20 <= uncertainty < 0.50
    # High      : uncertainty >= 0.50

    def uncertainty_category(value):

        if pd.isna(value):

            return "Unknown"

        if value < 0.20:

            return "Low"

        elif value < 0.50:

            return "Moderate"

        else:

            return "High"

    predictions[
        "uncertainty_category"
    ] = predictions[
        uncertainty_column
    ].apply(
        uncertainty_category
    )

else:

    predictions[
        "uncertainty_category"
    ] = "Unavailable"


# ============================================================
# CONFIDENCE CATEGORIES
# ============================================================

if confidence_column is not None:

    def confidence_category(value):

        if pd.isna(value):

            return "Unknown"

        if value >= 0.80:

            return "High"

        elif value >= 0.50:

            return "Moderate"

        else:

            return "Low"

    predictions[
        "confidence_category"
    ] = predictions[
        confidence_column
    ].apply(
        confidence_category
    )

else:

    predictions[
        "confidence_category"
    ] = "Unavailable"


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print()
print("=" * 70)
print("CALCULATING UNCERTAINTY STATISTICS")
print("=" * 70)

analysis_summary = {
    "stage": "STEP 8.3",
    "analysis": "Uncertainty and Confidence Analysis",
    "test_images": int(
        final_summary.get(
            "test_images",
            len(predictions)
        )
    ),
    "accuracy": float(
        final_summary.get(
            "accuracy",
            0
        )
    ),
    "mean_confidence": float(
        final_summary.get(
            "mean_confidence",
            0
        )
    ),
    "mean_uncertainty": float(
        final_summary.get(
            "mean_uncertainty",
            0
        )
    ),
    "median_uncertainty": float(
        final_summary.get(
            "median_uncertainty",
            0
        )
    ),
}


# ============================================================
# PER-SAMPLE STATISTICS
# ============================================================

if uncertainty_column is not None:

    valid_uncertainty = predictions[
        uncertainty_column
    ].dropna()

    if len(valid_uncertainty) > 0:

        analysis_summary.update({

            "uncertainty_min": float(
                valid_uncertainty.min()
            ),

            "uncertainty_max": float(
                valid_uncertainty.max()
            ),

            "uncertainty_mean": float(
                valid_uncertainty.mean()
            ),

            "uncertainty_median": float(
                valid_uncertainty.median()
            ),

            "uncertainty_std": float(
                valid_uncertainty.std()
            ),
        })


if confidence_column is not None:

    valid_confidence = predictions[
        confidence_column
    ].dropna()

    if len(valid_confidence) > 0:

        analysis_summary.update({

            "confidence_min": float(
                valid_confidence.min()
            ),

            "confidence_max": float(
                valid_confidence.max()
            ),

            "confidence_mean": float(
                valid_confidence.mean()
            ),

            "confidence_median": float(
                valid_confidence.median()
            ),

            "confidence_std": float(
                valid_confidence.std()
            ),
        })


# ============================================================
# CATEGORY COUNTS
# ============================================================

category_counts = (
    predictions[
        "uncertainty_category"
    ]
    .value_counts()
    .to_dict()
)

analysis_summary[
    "uncertainty_categories"
] = {
    key: int(value)
    for key, value
    in category_counts.items()
}


# ============================================================
# CORRECT VS INCORRECT
# ============================================================

if (
    correct_column is not None
    and uncertainty_column is not None
):

    correct_df = predictions[
        predictions[
            "correct_prediction"
        ] == True
    ]

    incorrect_df = predictions[
        predictions[
            "correct_prediction"
        ] == False
    ]

    analysis_summary[
        "correct_prediction_count"
    ] = int(
        len(correct_df)
    )

    analysis_summary[
        "incorrect_prediction_count"
    ] = int(
        len(incorrect_df)
    )

    if len(correct_df) > 0:

        analysis_summary[
            "correct_mean_uncertainty"
        ] = float(
            correct_df[
                uncertainty_column
            ].mean()
        )

    if len(incorrect_df) > 0:

        analysis_summary[
            "incorrect_mean_uncertainty"
        ] = float(
            incorrect_df[
                uncertainty_column
            ].mean()
        )


if (
    correct_column is not None
    and confidence_column is not None
):

    correct_df = predictions[
        predictions[
            "correct_prediction"
        ] == True
    ]

    incorrect_df = predictions[
        predictions[
            "correct_prediction"
        ] == False
    ]

    if len(correct_df) > 0:

        analysis_summary[
            "correct_mean_confidence"
        ] = float(
            correct_df[
                confidence_column
            ].mean()
        )

    if len(incorrect_df) > 0:

        analysis_summary[
            "incorrect_mean_confidence"
        ] = float(
            incorrect_df[
                confidence_column
            ].mean()
        )


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

per_class_rows = []

if (
    true_column is not None
    and uncertainty_column is not None
):

    for class_name in CLASS_NAMES:

        class_df = predictions[
            predictions[
                true_column
            ].astype(str)
            ==
            str(class_name)
        ]

        if len(class_df) == 0:

            continue

        row = {

            "class": class_name,

            "samples": int(
                len(class_df)
            ),

            "mean_uncertainty": float(
                class_df[
                    uncertainty_column
                ].mean()
            ),

            "median_uncertainty": float(
                class_df[
                    uncertainty_column
                ].median()
            ),
        }

        if confidence_column is not None:

            row[
                "mean_confidence"
            ] = float(
                class_df[
                    confidence_column
                ].mean()
            )

        if correct_column is not None:

            row[
                "accuracy"
            ] = float(
                class_df[
                    "correct_prediction"
                ].mean()
            )

        per_class_rows.append(
            row
        )


per_class_df = pd.DataFrame(
    per_class_rows
)


# ============================================================
# SAVE PER-CLASS DATA
# ============================================================

if len(per_class_df) > 0:

    per_class_path = (
        OUTPUT_ROOT
        / "per_class_uncertainty.csv"
    )

    per_class_df.to_csv(
        per_class_path,
        index=False
    )

    print(
        f"✓ Saved:\n"
        f"  {per_class_path}"
    )


# ============================================================
# HIGH UNCERTAINTY SAMPLES
# ============================================================

if uncertainty_column is not None:

    high_uncertainty = predictions[
        predictions[
            uncertainty_column
        ] >= 0.50
    ].copy()

    high_path = (
        OUTPUT_ROOT
        / "high_uncertainty_samples.csv"
    )

    high_uncertainty.to_csv(
        high_path,
        index=False
    )

    print()
    print(
        f"High uncertainty samples: "
        f"{len(high_uncertainty)}"
    )

    print(
        f"✓ Saved:\n"
        f"  {high_path}"
    )


# ============================================================
# PLOT 1
# UNCERTAINTY DISTRIBUTION
# ============================================================

if uncertainty_column is not None:

    values = predictions[
        uncertainty_column
    ].dropna()

    if len(values) > 0:

        plt.figure(
            figsize=(9, 6)
        )

        plt.hist(
            values,
            bins=30
        )

        plt.xlabel(
            "Uncertainty"
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.title(
            "Uncertainty Distribution"
        )

        plt.tight_layout()

        path = (
            OUTPUT_ROOT
            / "uncertainty_distribution.png"
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
# PLOT 2
# CONFIDENCE DISTRIBUTION
# ============================================================

if confidence_column is not None:

    values = predictions[
        confidence_column
    ].dropna()

    if len(values) > 0:

        plt.figure(
            figsize=(9, 6)
        )

        plt.hist(
            values,
            bins=30
        )

        plt.xlabel(
            "Confidence"
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.title(
            "Prediction Confidence Distribution"
        )

        plt.tight_layout()

        path = (
            OUTPUT_ROOT
            / "confidence_distribution.png"
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
# PLOT 3
# CONFIDENCE VS UNCERTAINTY
# ============================================================

if (
    confidence_column is not None
    and uncertainty_column is not None
):

    plot_df = predictions[
        [
            confidence_column,
            uncertainty_column
        ]
    ].dropna()

    if len(plot_df) > 0:

        plt.figure(
            figsize=(9, 6)
        )

        plt.scatter(
            plot_df[
                confidence_column
            ],
            plot_df[
                uncertainty_column
            ],
            alpha=0.35,
            s=15
        )

        plt.xlabel(
            "Confidence"
        )

        plt.ylabel(
            "Uncertainty"
        )

        plt.title(
            "Confidence vs Uncertainty"
        )

        plt.tight_layout()

        path = (
            OUTPUT_ROOT
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


## ============================================================
# PLOT 4
# CORRECT VS INCORRECT UNCERTAINTY
# ============================================================

if (
    correct_column is not None
    and uncertainty_column is not None
):

    correct_values = predictions[
        predictions[
            "correct_prediction"
        ] == True
    ][
        uncertainty_column
    ].dropna()

    incorrect_values = predictions[
        predictions[
            "correct_prediction"
        ] == False
    ][
        uncertainty_column
    ].dropna()

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
        "Uncertainty"
    )

    plt.title(
        "Uncertainty: Correct vs Incorrect Predictions"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
        / "correct_vs_incorrect_uncertainty.png"
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
# PLOT 5
# CORRECT VS INCORRECT CONFIDENCE
# ============================================================

if (
    correct_column is not None
    and confidence_column is not None
):

    correct_values = predictions[
        predictions[
            "correct_prediction"
        ] == True
    ][
        confidence_column
    ].dropna()

    incorrect_values = predictions[
        predictions[
            "correct_prediction"
        ] == False
    ][
        confidence_column
    ].dropna()

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
        "Confidence"
    )

    plt.title(
        "Confidence: Correct vs Incorrect Predictions"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
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
# PLOT 6
# PER-CLASS UNCERTAINTY
# ============================================================

if len(per_class_df) > 0:

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        per_class_df["class"],
        per_class_df[
            "mean_uncertainty"
        ]
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "Mean Uncertainty"
    )

    plt.title(
        "Mean Uncertainty by Disease Class"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
        / "per_class_uncertainty.png"
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
# PLOT 7
# PER-CLASS CONFIDENCE
# ============================================================

if (
    len(per_class_df) > 0
    and "mean_confidence"
    in per_class_df.columns
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        per_class_df["class"],
        per_class_df[
            "mean_confidence"
        ]
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "Mean Confidence"
    )

    plt.title(
        "Mean Confidence by Disease Class"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
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
# PLOT 8
# UNCERTAINTY CATEGORY COUNTS
# ============================================================

categories = [
    "Low",
    "Moderate",
    "High"
]

counts = [
    category_counts.get(
        category,
        0
    )
    for category in categories
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    categories,
    counts
)

plt.xlabel(
    "Uncertainty Category"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Prediction Uncertainty Categories"
)

plt.tight_layout()

path = (
    OUTPUT_ROOT
    / "uncertainty_categories.png"
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
# SAVE ENRICHED PREDICTIONS
# ============================================================

enriched_path = (
    OUTPUT_ROOT
    / "uncertainty_analysis_predictions.csv"
)

predictions.to_csv(
    enriched_path,
    index=False
)

print()
print(
    f"✓ Enriched prediction analysis saved:\n"
    f"  {enriched_path}"
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary_path = (
    OUTPUT_ROOT
    / "uncertainty_summary.json"
)

save_json(
    analysis_summary,
    summary_path
)


# ============================================================
# TEXT SUMMARY
# ============================================================

text_lines = []

text_lines.append(
    "RESPIRA — STEP 8.3"
)

text_lines.append(
    "UNCERTAINTY & CONFIDENCE ANALYSIS"
)

text_lines.append(
    "=" * 60
)

text_lines.append("")

text_lines.append(
    f"Test images: "
    f"{analysis_summary.get('test_images', 'N/A')}"
)

text_lines.append(
    f"Accuracy: "
    f"{analysis_summary.get('accuracy', 0):.4f}"
)

text_lines.append(
    f"Mean confidence: "
    f"{analysis_summary.get('mean_confidence', 0):.4f}"
)

text_lines.append(
    f"Mean uncertainty: "
    f"{analysis_summary.get('mean_uncertainty', 0):.4f}"
)

text_lines.append(
    f"Median uncertainty: "
    f"{analysis_summary.get('median_uncertainty', 0):.4f}"
)

text_lines.append("")

text_lines.append(
    "Uncertainty categories:"
)

for category in categories:

    text_lines.append(
        f"  {category}: "
        f"{category_counts.get(category, 0)}"
    )


if (
    "correct_mean_uncertainty"
    in analysis_summary
):

    text_lines.append("")

    text_lines.append(
        "Correct vs incorrect uncertainty:"
    )

    text_lines.append(
        f"  Correct: "
        f"{analysis_summary['correct_mean_uncertainty']:.4f}"
    )

    text_lines.append(
        f"  Incorrect: "
        f"{analysis_summary['incorrect_mean_uncertainty']:.4f}"
    )


if (
    "correct_mean_confidence"
    in analysis_summary
):

    text_lines.append("")

    text_lines.append(
        "Correct vs incorrect confidence:"
    )

    text_lines.append(
        f"  Correct: "
        f"{analysis_summary['correct_mean_confidence']:.4f}"
    )

    text_lines.append(
        f"  Incorrect: "
        f"{analysis_summary['incorrect_mean_confidence']:.4f}"
    )


text_lines.append("")

text_lines.append(
    "Interpretation:"
)

text_lines.append(
    "The uncertainty analysis evaluates how confidently "
    "the final disease classification pipeline makes "
    "predictions. Lower uncertainty indicates a more "
    "stable prediction, while higher uncertainty identifies "
    "samples that may require additional review."
)

text_lines.append("")

text_lines.append(
    "Generated by Respira Step 8.3."
)

text_path = (
    OUTPUT_ROOT
    / "uncertainty_summary.txt"
)

with open(
    text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(text_lines)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 8.3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Uncertainty analysis outputs:"
)

print(
    OUTPUT_ROOT
)

print()
print(
    "Generated:"
)

for file in sorted(
    OUTPUT_ROOT.iterdir()
):

    if file.is_file():

        print(
            f"  ✓ {file.name}"
        )

print()
print(
    "Key results:"
)

print(
    f"  Mean confidence : "
    f"{analysis_summary.get('mean_confidence', 0):.4f}"
)

print(
    f"  Mean uncertainty: "
    f"{analysis_summary.get('mean_uncertainty', 0):.4f}"
)

print(
    f"  Low uncertainty : "
    f"{category_counts.get('Low', 0)}"
)

print(
    f"  Moderate        : "
    f"{category_counts.get('Moderate', 0)}"
)

print(
    f"  High            : "
    f"{category_counts.get('High', 0)}"
)

print()
print(
    "Next stage:"
)

print(
    "STEP 8.4 — ERROR & MISCLASSIFICATION ANALYSIS"
)