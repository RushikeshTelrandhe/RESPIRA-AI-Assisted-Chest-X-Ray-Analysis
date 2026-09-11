# ============================================================
# Respira - STEP 8.2
# Per-Class Performance Analysis
# ============================================================
#
# Purpose:
#   Compare disease-level performance across all models.
#
# Classes:
#   0 - Atelectasis
#   1 - Bacterial Pneumonia
#   2 - Normal
#   3 - Pulmonary Edema
#   4 - Tuberculosis
#   5 - Viral Pneumonia
#
# Models:
#   DenseNet121
#   EfficientNet-B0
#   ViT-B/16
#   Final Fusion
#
# Outputs:
#   outputs/final_analysis/class_analysis/
#
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# INPUT FILES
# ============================================================

MODEL_REPORTS = {

    "DenseNet121":
        PROJECT_ROOT
        / "outputs"
        / "densenet"
        / "evaluation"
        / "classification_report.csv",

    "EfficientNet-B0":
        PROJECT_ROOT
        / "outputs"
        / "efficientnet_b0"
        / "evaluation"
        / "classification_report.csv",

    "ViT-B/16":
        PROJECT_ROOT
        / "outputs"
        / "vit"
        / "evaluation"
        / "classification_report.csv",

    "Final Fusion":
        PROJECT_ROOT
        / "outputs"
        / "fusion"
        / "classification"
        / "test"
        / "classification_report.csv",
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "class_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.2")
print("PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Output directory:")
print(OUTPUT_ROOT)


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
# LOAD CLASSIFICATION REPORT
# ============================================================

def load_report(
    model_name,
    path
):

    if not path.exists():

        print()
        print(
            f"✗ {model_name} report not found:"
        )

        print(path)

        return None

    print()
    print(
        f"✓ Loading {model_name}"
    )

    print(
        f"  {path}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows: {len(df)}"
    )

    return df


# ============================================================
# LOAD ALL REPORTS
# ============================================================

print()
print("=" * 70)
print("LOADING CLASSIFICATION REPORTS")
print("=" * 70)


reports = {}


for model_name, path in MODEL_REPORTS.items():

    report = load_report(
        model_name,
        path
    )

    if report is not None:

        reports[model_name] = report


if len(reports) == 0:

    raise RuntimeError(
        "\nNo classification reports were found."
    )


# ============================================================
# SHOW REPORT COLUMNS
# ============================================================

print()
print("=" * 70)
print("REPORT STRUCTURE")
print("=" * 70)


for model_name, df in reports.items():

    print()
    print(
        f"{model_name}:"
    )

    print(
        "  Columns:",
        df.columns.tolist()
    )


# ============================================================
# FIND CLASS ROW
# ============================================================

def find_class_row(
    df,
    class_name
):

    # --------------------------------------------------------
    # Check common possible label columns
    # --------------------------------------------------------

    possible_label_columns = [

        "class",
        "Class",
        "label",
        "Label",
        "Unnamed: 0",

    ]

    label_column = None

    for column in possible_label_columns:

        if column in df.columns:

            label_column = column

            break

    # --------------------------------------------------------
    # Search using label column
    # --------------------------------------------------------

    if label_column is not None:

        values = (
            df[label_column]
            .astype(str)
            .str.strip()
        )

        matches = df[
            values == class_name
        ]

        if len(matches) > 0:

            return matches.iloc[0]

    # --------------------------------------------------------
    # Search all columns
    # --------------------------------------------------------

    for _, row in df.iterrows():

        row_values = [
            str(value).strip()
            for value in row.values
        ]

        if class_name in row_values:

            return row

    return None


# ============================================================
# EXTRACT METRICS
# ============================================================

print()
print("=" * 70)
print("EXTRACTING PER-CLASS METRICS")
print("=" * 70)


rows = []


for model_name, df in reports.items():

    for class_name in CLASS_NAMES:

        row = find_class_row(
            df,
            class_name
        )

        if row is None:

            print()
            print(
                f"⚠ {class_name} not found "
                f"in {model_name}"
            )

            continue

        # ----------------------------------------------------
        # Metric extraction
        # ----------------------------------------------------

        def get_value(
            column
        ):

            if column not in row.index:

                return np.nan

            value = row[column]

            try:

                return float(value)

            except:

                return np.nan

        rows.append({

            "Model":
                model_name,

            "Class":
                class_name,

            "Precision":
                get_value(
                    "precision"
                ),

            "Recall":
                get_value(
                    "recall"
                ),

            "F1 Score":
                get_value(
                    "f1-score"
                ),

            "Support":
                get_value(
                    "support"
                ),

        })


per_class_df = pd.DataFrame(
    rows
)


if len(per_class_df) == 0:

    raise RuntimeError(
        "\nNo per-class metrics could be extracted."
    )


# ============================================================
# SAVE RAW TABLE
# ============================================================

csv_path = (
    OUTPUT_ROOT
    / "per_class_comparison.csv"
)

per_class_df.to_csv(
    csv_path,
    index=False
)

print()
print(
    "✓ Per-class comparison saved:"
)

print(
    csv_path
)


# ============================================================
# PRINT TABLE
# ============================================================

print()
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

display_df = per_class_df.copy()

for column in [

    "Precision",
    "Recall",
    "F1 Score",

]:

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
# FIND BEST MODEL PER CLASS
# ============================================================

print()
print("=" * 70)
print("BEST MODEL PER DISEASE")
print("=" * 70)


best_by_class = {}


for class_name in CLASS_NAMES:

    class_df = per_class_df[
        per_class_df["Class"]
        == class_name
    ]

    if len(class_df) == 0:

        continue

    valid_df = class_df[
        class_df["F1 Score"].notna()
    ]

    if len(valid_df) == 0:

        continue

    best_row = valid_df.loc[
        valid_df["F1 Score"].idxmax()
    ]

    best_by_class[class_name] = {

        "best_model":
            best_row["Model"],

        "precision":
            float(
                best_row["Precision"]
            ),

        "recall":
            float(
                best_row["Recall"]
            ),

        "f1_score":
            float(
                best_row["F1 Score"]
            ),

    }

    print()
    print(
        f"{class_name}:"
    )

    print(
        f"  Best model : "
        f"{best_row['Model']}"
    )

    print(
        f"  F1 Score   : "
        f"{best_row['F1 Score'] * 100:.2f}%"
    )


# ============================================================
# GRAPH 1
# F1 SCORE BY DISEASE
# ============================================================

print()
print("=" * 70)
print("GENERATING PER-CLASS GRAPHS")
print("=" * 70)


for class_name in CLASS_NAMES:

    class_df = per_class_df[
        per_class_df["Class"]
        == class_name
    ]

    if len(class_df) == 0:

        continue

    plt.figure(
        figsize=(10, 6)
    )

    values = (
        class_df["F1 Score"] * 100
    )

    plt.bar(
        class_df["Model"],
        values
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "F1 Score (%)"
    )

    plt.title(
        f"F1 Score Comparison — {class_name}"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    safe_name = (
        class_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    output_path = (
        OUTPUT_ROOT
        / f"f1_{safe_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"✓ {output_path.name}"
    )


# ============================================================
# GRAPH 2
# OVERALL PER-CLASS F1
# ============================================================

plt.figure(
    figsize=(13, 7)
)

x = np.arange(
    len(CLASS_NAMES)
)

available_models = (
    per_class_df["Model"]
    .unique()
    .tolist()
)

width = (
    0.8
    / max(
        len(available_models),
        1
    )
)


for i, model_name in enumerate(
    available_models
):

    values = []

    for class_name in CLASS_NAMES:

        matching = per_class_df[
            (
                per_class_df["Model"]
                == model_name
            )
            &
            (
                per_class_df["Class"]
                == class_name
            )
        ]

        if len(matching) == 0:

            values.append(0)

        else:

            values.append(
                matching.iloc[0]["F1 Score"]
                * 100
            )

    plt.bar(
        x + i * width,
        values,
        width,
        label=model_name
    )


plt.xlabel(
    "Disease Class"
)

plt.ylabel(
    "F1 Score (%)"
)

plt.title(
    "Respira — Per-Class F1 Score Comparison"
)

plt.xticks(
    x + width * (
        len(available_models) - 1
    ) / 2,
    CLASS_NAMES,
    rotation=20
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

overall_path = (
    OUTPUT_ROOT
    / "per_class_performance.png"
)

plt.savefig(
    overall_path,
    dpi=300
)

plt.close()

print()
print(
    f"✓ Overall per-class graph:"
)

print(
    overall_path
)


# ============================================================
# GRAPH 3
# PRECISION / RECALL / F1 FOR FINAL FUSION
# ============================================================

if "Final Fusion" in reports:

    fusion_df = per_class_df[
        per_class_df["Model"]
        == "Final Fusion"
    ]

    if len(fusion_df) > 0:

        plt.figure(
            figsize=(13, 7)
        )

        x = np.arange(
            len(fusion_df)
        )

        width = 0.25

        plt.bar(
            x - width,
            fusion_df["Precision"] * 100,
            width,
            label="Precision"
        )

        plt.bar(
            x,
            fusion_df["Recall"] * 100,
            width,
            label="Recall"
        )

        plt.bar(
            x + width,
            fusion_df["F1 Score"] * 100,
            width,
            label="F1 Score"
        )

        plt.xlabel(
            "Disease Class"
        )

        plt.ylabel(
            "Score (%)"
        )

        plt.title(
            "Final Fusion — Per-Class Performance"
        )

        plt.xticks(
            x,
            fusion_df["Class"],
            rotation=20
        )

        plt.ylim(
            0,
            100
        )

        plt.legend()

        plt.tight_layout()

        fusion_plot = (
            OUTPUT_ROOT
            / "final_fusion_per_class.png"
        )

        plt.savefig(
            fusion_plot,
            dpi=300
        )

        plt.close()

        print()
        print(
            "✓ Final Fusion class graph:"
        )

        print(
            fusion_plot
        )


# ============================================================
# CLASS AVERAGE SUMMARY
# ============================================================

print()
print("=" * 70)
print("CLASS AVERAGE PERFORMANCE")
print("=" * 70)


class_average = (
    per_class_df
    .groupby("Model")[
        [
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]
    .mean()
    .reset_index()
)


print()

average_display = class_average.copy()

for column in [

    "Precision",
    "Recall",
    "F1 Score",

]:

    average_display[column] = (
        average_display[column] * 100
    ).round(2)


print(
    average_display.to_string(
        index=False
    )
)


class_average_path = (
    OUTPUT_ROOT
    / "class_average_comparison.csv"
)

class_average.to_csv(
    class_average_path,
    index=False
)

print()
print(
    f"✓ Class-average table saved:"
)

print(
    class_average_path
)


# ============================================================
# BEST MODEL OVERALL BY CLASS F1
# ============================================================

overall_class_f1 = (
    class_average
    .sort_values(
        "F1 Score",
        ascending=False
    )
)

if len(overall_class_f1) > 0:

    best_overall_model = (
        overall_class_f1.iloc[0]
    )

else:

    best_overall_model = None


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {

    "stage":
        "STEP 8.2",

    "purpose":
        "Per-class disease performance analysis",

    "classes":
        CLASS_NAMES,

    "models":
        list(reports.keys()),

    "best_model_per_class":
        best_by_class,

    "class_average_performance":
        [],

}


for _, row in class_average.iterrows():

    summary[
        "class_average_performance"
    ].append({

        "model":
            row["Model"],

        "precision":
            float(
                row["Precision"]
            ),

        "recall":
            float(
                row["Recall"]
            ),

        "f1_score":
            float(
                row["F1 Score"]
            ),

    })


if best_overall_model is not None:

    summary[
        "best_average_f1_model"
    ] = {

        "model":
            best_overall_model["Model"],

        "average_f1":
            float(
                best_overall_model["F1 Score"]
            ),

    }


json_path = (
    OUTPUT_ROOT
    / "per_class_summary.json"
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
print(
    "✓ JSON summary saved:"
)

print(
    json_path
)


# ============================================================
# TEXT SUMMARY
# ============================================================

text_path = (
    OUTPUT_ROOT
    / "per_class_summary.txt"
)

with open(
    text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — STEP 8.2\n"
    )

    f.write(
        "PER-CLASS PERFORMANCE ANALYSIS\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    for class_name, result in (
        best_by_class.items()
    ):

        f.write(
            f"{class_name}\n"
        )

        f.write(
            f"  Best model: "
            f"{result['best_model']}\n"
        )

        f.write(
            f"  Precision: "
            f"{result['precision'] * 100:.2f}%\n"
        )

        f.write(
            f"  Recall: "
            f"{result['recall'] * 100:.2f}%\n"
        )

        f.write(
            f"  F1 Score: "
            f"{result['f1_score'] * 100:.2f}%\n\n"
        )


print()
print(
    "✓ Text summary saved:"
)

print(
    text_path
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 8.2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Outputs:")
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
print("Next stage:")
print(
    "STEP 8.3 — UNCERTAINTY & CONFIDENCE ANALYSIS"
)

print("=" * 70)