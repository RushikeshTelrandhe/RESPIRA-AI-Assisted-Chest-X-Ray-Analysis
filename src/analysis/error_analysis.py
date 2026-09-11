# ============================================================
# Respira — STEP 8.4
# ERROR & MISCLASSIFICATION ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze final model errors and misclassifications.
#
# Input:
#   outputs/final_prediction/predictions/final_predictions.csv
#
# Output:
#   outputs/final_analysis/error_analysis/
#
# Generated:
#   - error_summary.json
#   - error_summary.txt
#   - error_analysis_predictions.csv
#   - misclassified_samples.csv
#   - confusion_matrix.csv
#   - misclassification_pairs.csv
#   - per_class_errors.csv
#   - confusion_matrix.png
#   - error_rate_by_class.png
#   - misclassification_pairs.png
#   - correct_vs_incorrect.png
#
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
)

PREDICTION_ROOT = (
    FINAL_ROOT
    / "predictions"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "error_analysis"
)

OUTPUT_ROOT.mkdir(
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

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.4")
print("ERROR & MISCLASSIFICATION ANALYSIS")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Input root   : {FINAL_ROOT}")
print(f"Output root  : {OUTPUT_ROOT}")


# ============================================================
# FIND PREDICTION FILE
# ============================================================

print()
print("=" * 70)
print("LOCATING FINAL PREDICTIONS")
print("=" * 70)


prediction_candidates = [
    PREDICTION_ROOT / "final_predictions.csv",
    FINAL_ROOT / "final_predictions.csv",
]

prediction_path = None

for candidate in prediction_candidates:

    if candidate.exists():

        prediction_path = candidate
        break


if prediction_path is None:

    raise FileNotFoundError(
        "\nFinal prediction CSV could not be found.\n"
        "Expected:\n"
        f"{PREDICTION_ROOT / 'final_predictions.csv'}"
    )


print()
print("✓ Prediction file found:")
print(f"  {prediction_path}")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING PREDICTIONS")
print("=" * 70)


predictions = pd.read_csv(
    prediction_path
)

print()
print(
    f"Prediction shape: {predictions.shape}"
)

print()
print("Columns:")

for column in predictions.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# COLUMN DETECTION
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


true_index_column = find_column(
    predictions,
    [
        "true_label_index",
        "label_index",
        "true_index",
        "actual_index",
    ]
)

true_class_column = find_column(
    predictions,
    [
        "true_class",
        "true_label",
        "actual",
        "label",
    ]
)

predicted_index_column = find_column(
    predictions,
    [
        "predicted_label_index",
        "predicted_index",
    ]
)

predicted_class_column = find_column(
    predictions,
    [
        "predicted_class",
        "predicted_label",
        "prediction",
        "predicted",
    ]
)

confidence_column = find_column(
    predictions,
    [
        "prediction_confidence",
        "confidence",
        "max_probability",
    ]
)

uncertainty_column = find_column(
    predictions,
    [
        "sample_uncertainty",
        "uncertainty",
        "uncertainty_score",
    ]
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

if (
    true_index_column is None
    and true_class_column is None
):

    raise RuntimeError(
        "Could not identify true-label column."
    )


if (
    predicted_index_column is None
    and predicted_class_column is None
):

    raise RuntimeError(
        "Could not identify predicted-label column."
    )


print()
print("Detected columns:")

print(
    f"  True label       : "
    f"{true_class_column or true_index_column}"
)

print(
    f"  Predicted label  : "
    f"{predicted_class_column or predicted_index_column}"
)

print(
    f"  Confidence       : "
    f"{confidence_column}"
)

print(
    f"  Uncertainty      : "
    f"{uncertainty_column}"
)


# ============================================================
# NORMALIZE TRUE LABELS
# ============================================================

print()
print("=" * 70)
print("NORMALIZING LABELS")
print("=" * 70)


if true_index_column is not None:

    true_indices = pd.to_numeric(
        predictions[
            true_index_column
        ],
        errors="coerce"
    )

else:

    true_indices = predictions[
        true_class_column
    ].map(
        {
            name: index
            for index, name
            in enumerate(CLASS_NAMES)
        }
    )


# ============================================================
# NORMALIZE PREDICTED LABELS
# ============================================================

if predicted_index_column is not None:

    predicted_indices = pd.to_numeric(
        predictions[
            predicted_index_column
        ],
        errors="coerce"
    )

else:

    predicted_indices = predictions[
        predicted_class_column
    ].map(
        {
            name: index
            for index, name
            in enumerate(CLASS_NAMES)
        }
    )


if (
    true_indices.isna().any()
    or predicted_indices.isna().any()
):

    raise RuntimeError(
        "Some labels could not be converted "
        "to class indices."
    )


true_indices = true_indices.astype(int)
predicted_indices = predicted_indices.astype(int)


# ============================================================
# ADD STANDARDIZED LABEL COLUMNS
# ============================================================

predictions[
    "analysis_true_index"
] = true_indices

predictions[
    "analysis_predicted_index"
] = predicted_indices

predictions[
    "analysis_true_class"
] = [
    CLASS_NAMES[index]
    if 0 <= index < NUM_CLASSES
    else "Unknown"
    for index in true_indices
]

predictions[
    "analysis_predicted_class"
] = [
    CLASS_NAMES[index]
    if 0 <= index < NUM_CLASSES
    else "Unknown"
    for index in predicted_indices
]


# ============================================================
# CORRECTNESS
# ============================================================

predictions[
    "correct_prediction"
] = (
    predictions[
        "analysis_true_index"
    ]
    ==
    predictions[
        "analysis_predicted_index"
    ]
)


predictions[
    "error"
] = ~predictions[
    "correct_prediction"
]


# ============================================================
# ERROR TYPE
# ============================================================

predictions[
    "error_type"
] = np.where(
    predictions[
        "correct_prediction"
    ],
    "Correct",
    "Misclassified"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_samples = len(
    predictions
)

correct_count = int(
    predictions[
        "correct_prediction"
    ].sum()
)

incorrect_count = (
    total_samples
    - correct_count
)

accuracy = (
    correct_count
    / total_samples
    if total_samples > 0
    else 0.0
)

error_rate = (
    incorrect_count
    / total_samples
    if total_samples > 0
    else 0.0
)


print()
print("=" * 70)
print("OVERALL ERROR STATISTICS")
print("=" * 70)

print()
print(
    f"Total samples     : {total_samples}"
)

print(
    f"Correct           : {correct_count}"
)

print(
    f"Incorrect         : {incorrect_count}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Error rate        : {error_rate:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


confusion_matrix = np.zeros(
    (
        NUM_CLASSES,
        NUM_CLASSES
    ),
    dtype=int
)


for true_index, predicted_index in zip(
    true_indices,
    predicted_indices
):

    if (
        0 <= true_index < NUM_CLASSES
        and
        0 <= predicted_index < NUM_CLASSES
    ):

        confusion_matrix[
            true_index,
            predicted_index
        ] += 1


confusion_df = pd.DataFrame(
    confusion_matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

confusion_df.index.name = (
    "True Class"
)


confusion_path = (
    OUTPUT_ROOT
    / "confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_path
)

print()
print(
    f"✓ {confusion_path.name}"
)


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 70)


per_class_rows = []


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    true_mask = (
        true_indices
        == class_index
    )

    total_class = int(
        true_mask.sum()
    )

    correct_class = int(
        (
            true_mask
            &
            (
                predicted_indices
                == class_index
            )
        ).sum()
    )

    incorrect_class = (
        total_class
        - correct_class
    )

    class_accuracy = (
        correct_class
        / total_class
        if total_class > 0
        else 0.0
    )

    class_error_rate = (
        incorrect_class
        / total_class
        if total_class > 0
        else 0.0
    )

    predicted_as_class = int(
        (
            predicted_indices
            == class_index
        ).sum()
    )

    false_positive = (
        predicted_as_class
        - correct_class
    )

    false_negative = (
        incorrect_class
    )

    per_class_rows.append(
        {
            "class_index": class_index,
            "class": class_name,
            "total_samples": total_class,
            "correct_predictions": correct_class,
            "incorrect_predictions": incorrect_class,
            "accuracy": class_accuracy,
            "error_rate": class_error_rate,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)


per_class_path = (
    OUTPUT_ROOT
    / "per_class_errors.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False
)

print()

for row in per_class_rows:

    print(
        f"{row['class']:<22} "
        f"Total={row['total_samples']:<4} "
        f"Correct={row['correct_predictions']:<4} "
        f"Errors={row['incorrect_predictions']:<4} "
        f"Error Rate={row['error_rate']:.4f}"
    )

print()
print(
    f"✓ {per_class_path.name}"
)


# ============================================================
# MISCLASSIFICATION PAIRS
# ============================================================

print()
print("=" * 70)
print("TOP MISCLASSIFICATION PAIRS")
print("=" * 70)


pair_rows = []


for true_index in range(
    NUM_CLASSES
):

    for predicted_index in range(
        NUM_CLASSES
    ):

        if (
            true_index
            == predicted_index
        ):
            continue

        count = int(
            confusion_matrix[
                true_index,
                predicted_index
            ]
        )

        if count == 0:
            continue

        pair_rows.append(
            {
                "true_class": CLASS_NAMES[
                    true_index
                ],
                "predicted_class": CLASS_NAMES[
                    predicted_index
                ],
                "count": count,
            }
        )


pairs_df = pd.DataFrame(
    pair_rows
)


if not pairs_df.empty:

    pairs_df = pairs_df.sort_values(
        "count",
        ascending=False
    ).reset_index(
        drop=True
    )

    pairs_df[
        "percentage_of_errors"
    ] = (
        pairs_df["count"]
        / incorrect_count
        * 100
        if incorrect_count > 0
        else 0
    )

else:

    pairs_df = pd.DataFrame(
        columns=[
            "true_class",
            "predicted_class",
            "count",
            "percentage_of_errors",
        ]
    )


pairs_path = (
    OUTPUT_ROOT
    / "misclassification_pairs.csv"
)

pairs_df.to_csv(
    pairs_path,
    index=False
)


print()

if pairs_df.empty:

    print(
        "No misclassification pairs found."
    )

else:

    for _, row in pairs_df.head(10).iterrows():

        print(
            f"{row['true_class']} "
            f"→ "
            f"{row['predicted_class']} : "
            f"{int(row['count'])}"
        )

print()
print(
    f"✓ {pairs_path.name}"
)


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified = predictions[
    predictions[
        "correct_prediction"
    ] == False
].copy()


misclassified = misclassified.sort_values(
    by=(
        confidence_column
        if confidence_column is not None
        else "analysis_predicted_index"
    ),
    ascending=False
)


misclassified_path = (
    OUTPUT_ROOT
    / "misclassified_samples.csv"
)

misclassified.to_csv(
    misclassified_path,
    index=False
)

print()
print(
    f"✓ {misclassified_path.name}"
)


# ============================================================
# ERROR ANALYSIS DATASET
# ============================================================

analysis_path = (
    OUTPUT_ROOT
    / "error_analysis_predictions.csv"
)

predictions.to_csv(
    analysis_path,
    index=False
)

print(
    f"✓ {analysis_path.name}"
)


# ============================================================
# PLOT 1
# CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(11, 9)
)

plt.imshow(
    confusion_matrix
)

plt.colorbar(
    label="Number of Samples"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "Respira — Confusion Matrix"
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

path = (
    OUTPUT_ROOT
    / "confusion_matrix.png"
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
# ERROR RATE BY CLASS
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    per_class_df["class"],
    per_class_df["error_rate"]
)

plt.ylabel(
    "Error Rate"
)

plt.xlabel(
    "Disease Class"
)

plt.title(
    "Error Rate by Disease Class"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_ROOT
    / "error_rate_by_class.png"
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
# TOP MISCLASSIFICATION PAIRS
# ============================================================

if not pairs_df.empty:

    plot_pairs = pairs_df.head(
        min(10, len(pairs_df))
    ).copy()

    labels = [
        f"{row.true_class}\n→\n{row.predicted_class}"
        for _, row in plot_pairs.iterrows()
    ]

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        labels,
        plot_pairs["count"]
    )

    plt.ylabel(
        "Number of Errors"
    )

    plt.xlabel(
        "Misclassification Pair"
    )

    plt.title(
        "Top Misclassification Pairs"
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT
        / "misclassification_pairs.png"
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
# PLOT 4
# CORRECT VS INCORRECT
# ============================================================

correct_count_plot = correct_count
incorrect_count_plot = incorrect_count

plt.figure(
    figsize=(7, 6)
)

plt.bar(
    [
        "Correct",
        "Incorrect"
    ],
    [
        correct_count_plot,
        incorrect_count_plot
    ]
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Correct vs Incorrect Predictions"
)

plt.tight_layout()

path = (
    OUTPUT_ROOT
    / "correct_vs_incorrect.png"
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
# CONFIDENCE ERROR ANALYSIS
# ============================================================

confidence_error_summary = {}


if confidence_column is not None:

    confidence_values = pd.to_numeric(
        predictions[
            confidence_column
        ],
        errors="coerce"
    )

    correct_confidence = confidence_values[
        predictions[
            "correct_prediction"
        ]
    ].dropna()

    incorrect_confidence = confidence_values[
        ~predictions[
            "correct_prediction"
        ]
    ].dropna()

    if len(correct_confidence) > 0:

        confidence_error_summary[
            "correct_mean_confidence"
        ] = float(
            correct_confidence.mean()
        )

        confidence_error_summary[
            "correct_median_confidence"
        ] = float(
            correct_confidence.median()
        )

    if len(incorrect_confidence) > 0:

        confidence_error_summary[
            "incorrect_mean_confidence"
        ] = float(
            incorrect_confidence.mean()
        )

        confidence_error_summary[
            "incorrect_median_confidence"
        ] = float(
            incorrect_confidence.median()
        )


# ============================================================
# UNCERTAINTY ERROR ANALYSIS
# ============================================================

uncertainty_error_summary = {}


if uncertainty_column is not None:

    uncertainty_values = pd.to_numeric(
        predictions[
            uncertainty_column
        ],
        errors="coerce"
    )

    correct_uncertainty = uncertainty_values[
        predictions[
            "correct_prediction"
        ]
    ].dropna()

    incorrect_uncertainty = uncertainty_values[
        ~predictions[
            "correct_prediction"
        ]
    ].dropna()

    if len(correct_uncertainty) > 0:

        uncertainty_error_summary[
            "correct_mean_uncertainty"
        ] = float(
            correct_uncertainty.mean()
        )

        uncertainty_error_summary[
            "correct_median_uncertainty"
        ] = float(
            correct_uncertainty.median()
        )

    if len(incorrect_uncertainty) > 0:

        uncertainty_error_summary[
            "incorrect_mean_uncertainty"
        ] = float(
            incorrect_uncertainty.mean()
        )

        uncertainty_error_summary[
            "incorrect_median_uncertainty"
        ] = float(
            incorrect_uncertainty.median()
        )


# ============================================================
# TOP ERROR PAIR
# ============================================================

if not pairs_df.empty:

    top_pair = {
        "true_class": str(
            pairs_df.iloc[0][
                "true_class"
            ]
        ),
        "predicted_class": str(
            pairs_df.iloc[0][
                "predicted_class"
            ]
        ),
        "count": int(
            pairs_df.iloc[0][
                "count"
            ]
        ),
    }

else:

    top_pair = None


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage": "STEP 8.4",

    "analysis": (
        "Error and Misclassification Analysis"
    ),

    "prediction_file": str(
        prediction_path
    ),

    "test_images": int(
        total_samples
    ),

    "correct_predictions": int(
        correct_count
    ),

    "incorrect_predictions": int(
        incorrect_count
    ),

    "accuracy": float(
        accuracy
    ),

    "error_rate": float(
        error_rate
    ),

    "num_classes": NUM_CLASSES,

    "classes": CLASS_NAMES,

    "top_misclassification": (
        top_pair
    ),

    "confidence_analysis": (
        confidence_error_summary
    ),

    "uncertainty_analysis": (
        uncertainty_error_summary
    ),

    "output_directory": str(
        OUTPUT_ROOT
    ),
}


summary_path = (
    OUTPUT_ROOT
    / "error_summary.json"
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
    f"✓ {summary_path.name}"
)


# ============================================================
# SUMMARY TXT
# ============================================================

txt_path = (
    OUTPUT_ROOT
    / "error_summary.txt"
)


with open(
    txt_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RESPIRA — STEP 8.4\n"
    )

    file.write(
        "ERROR & MISCLASSIFICATION ANALYSIS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Test images       : {total_samples}\n"
    )

    file.write(
        f"Correct predictions: {correct_count}\n"
    )

    file.write(
        f"Incorrect predictions: {incorrect_count}\n"
    )

    file.write(
        f"Accuracy           : {accuracy:.4f}\n"
    )

    file.write(
        f"Error rate         : {error_rate:.4f}\n"
    )

    file.write(
        "\n"
        "Per-class errors\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    for row in per_class_rows:

        file.write(
            f"{row['class']}: "
            f"Total={row['total_samples']}, "
            f"Correct={row['correct_predictions']}, "
            f"Errors={row['incorrect_predictions']}, "
            f"Error Rate={row['error_rate']:.4f}\n"
        )

    file.write(
        "\nTop misclassification pairs\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    for _, row in pairs_df.head(10).iterrows():

        file.write(
            f"{row['true_class']} -> "
            f"{row['predicted_class']} : "
            f"{int(row['count'])}\n"
        )

    if confidence_error_summary:

        file.write(
            "\nConfidence analysis\n"
        )

        for key, value in (
            confidence_error_summary.items()
        ):

            file.write(
                f"{key}: {value:.6f}\n"
            )

    if uncertainty_error_summary:

        file.write(
            "\nUncertainty analysis\n"
        )

        for key, value in (
            uncertainty_error_summary.items()
        ):

            file.write(
                f"{key}: {value:.6f}\n"
            )


print(
    f"✓ {txt_path.name}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 8.4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Error analysis outputs:"
)

print(
    OUTPUT_ROOT
)

print()
print(
    "Generated:"
)

generated_files = [
    "error_summary.json",
    "error_summary.txt",
    "error_analysis_predictions.csv",
    "misclassified_samples.csv",
    "confusion_matrix.csv",
    "misclassification_pairs.csv",
    "per_class_errors.csv",
    "confusion_matrix.png",
    "error_rate_by_class.png",
    "misclassification_pairs.png",
    "correct_vs_incorrect.png",
]

for filename in generated_files:

    path = (
        OUTPUT_ROOT
        / filename
    )

    if path.exists():

        print(
            f"  ✓ {filename}"
        )

print()
print(
    "Next stage:"
)

print(
    "STEP 8.5 — MODEL COMPARISON & FINAL PERFORMANCE ANALYSIS"
)

print()