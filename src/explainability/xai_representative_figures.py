# ============================================================
# Respira — STEP 8.10
# Representative XAI Figures
#
# EfficientNet-B0 Grad-CAM
#              vs
# ViT-B/16 Attention Rollout
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
)

COMPARISON_CSV = (
    COMPARISON_ROOT
    / "reports"
    / "xai_comparison_results.csv"
)

OUTPUT_ROOT = (
    COMPARISON_ROOT
    / "representative_figures"
)

for directory in [
    OUTPUT_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

FIGURE_DPI = 250


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.10")
print("REPRESENTATIVE XAI FIGURES")
print("=" * 70)

print()
print("Comparison file:")
print(COMPARISON_CSV)

print()
print("Output:")
print(OUTPUT_ROOT)


# ============================================================
# 4. CHECK INPUT
# ============================================================

if not COMPARISON_CSV.exists():

    raise FileNotFoundError(
        f"\nXAI comparison results not found:\n"
        f"{COMPARISON_CSV}"
    )


# ============================================================
# 5. LOAD COMPARISON
# ============================================================

df = pd.read_csv(
    COMPARISON_CSV
)

print()
print(
    f"Comparison records: {len(df)}"
)

if len(df) == 0:

    raise RuntimeError(
        "XAI comparison file is empty."
    )


# ============================================================
# 6. IMAGE LOADING
# ============================================================

def load_image(path):

    return np.asarray(
        Image.open(
            path
        ).convert("RGB")
    )


def load_heatmap(path):

    image = Image.open(
        path
    ).convert("L")

    array = np.asarray(
        image
    ).astype(np.float32)

    array -= array.min()

    maximum = array.max()

    if maximum > 0:
        array /= maximum

    return array


# ============================================================
# 7. REPRESENTATIVE CASE SELECTION
# ============================================================

representative_cases = {}


# ------------------------------------------------------------
# CASE 1
# Both models correct + highest XAI agreement
# ------------------------------------------------------------

subset = df[
    df["both_correct"] == True
].copy()

if len(subset) > 0:

    row = subset.sort_values(
        "pearson_similarity",
        ascending=False
    ).iloc[0]

    representative_cases[
        "01_both_correct_high_agreement"
    ] = row


# ------------------------------------------------------------
# CASE 2
# Both models correct + lowest XAI agreement
# ------------------------------------------------------------

subset = df[
    df["both_correct"] == True
].copy()

if len(subset) > 0:

    row = subset.sort_values(
        "pearson_similarity",
        ascending=True
    ).iloc[0]

    representative_cases[
        "02_both_correct_low_agreement"
    ] = row


# ------------------------------------------------------------
# CASE 3
# EfficientNet correct / ViT wrong
# ------------------------------------------------------------

subset = df[
    df["eff_correct_vit_wrong"] == True
].copy()

if len(subset) > 0:

    row = subset.sort_values(
        "pearson_similarity",
        ascending=True
    ).iloc[0]

    representative_cases[
        "03_efficientnet_correct_vit_wrong"
    ] = row


# ------------------------------------------------------------
# CASE 4
# ViT correct / EfficientNet wrong
# ------------------------------------------------------------

subset = df[
    df["eff_wrong_vit_correct"] == True
].copy()

if len(subset) > 0:

    row = subset.sort_values(
        "pearson_similarity",
        ascending=True
    ).iloc[0]

    representative_cases[
        "04_vit_correct_efficientnet_wrong"
    ] = row


# ------------------------------------------------------------
# CASE 5
# Both models wrong
# ------------------------------------------------------------

subset = df[
    df["both_wrong"] == True
].copy()

if len(subset) > 0:

    row = subset.sort_values(
        "pearson_similarity",
        ascending=True
    ).iloc[0]

    representative_cases[
        "05_both_wrong"
    ] = row


# ============================================================
# 8. PRINT SELECTED CASES
# ============================================================

print()
print("=" * 70)
print("SELECTED REPRESENTATIVE CASES")
print("=" * 70)

for case_name, row in representative_cases.items():

    print()
    print(case_name)

    print(
        f"  True class       : "
        f"{row['true_class']}"
    )

    print(
        f"  EfficientNet     : "
        f"{row['eff_predicted_class']} "
        f"({row['eff_confidence']:.4f})"
    )

    print(
        f"  ViT              : "
        f"{row['vit_predicted_class']} "
        f"({row['vit_confidence']:.4f})"
    )

    print(
        f"  Pearson          : "
        f"{row['pearson_similarity']:.4f}"
    )

    print(
        f"  Cosine           : "
        f"{row['cosine_similarity']:.4f}"
    )

    print(
        f"  Attention IoU    : "
        f"{row['attention_iou']:.4f}"
    )


# ============================================================
# 9. GENERATE FIGURE
# ============================================================

def create_comparison_figure(
    row,
    case_name,
    output_path
):

    original = load_image(
        row["image_path"]
    )

    eff_heatmap = load_heatmap(
        row["eff_heatmap_path"]
    )

    vit_heatmap = load_heatmap(
        row["vit_heatmap_path"]
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        "Original X-ray",
        fontsize=12
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------

    axes[1].imshow(
        original
    )

    axes[1].imshow(
        eff_heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        "EfficientNet-B0 — Grad-CAM",
        fontsize=12
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # ViT
    # --------------------------------------------------------

    axes[2].imshow(
        original
    )

    axes[2].imshow(
        vit_heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "ViT-B/16 — Attention Rollout",
        fontsize=12
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"{case_name.replace('_', ' ')}\n"
            f"True: {row['true_class']} | "
            f"EfficientNet: "
            f"{row['eff_predicted_class']} "
            f"({row['eff_confidence']:.3f}) | "
            f"ViT: "
            f"{row['vit_predicted_class']} "
            f"({row['vit_confidence']:.3f})"
        ),
        fontsize=13
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.02,
        (
            f"Pearson = "
            f"{row['pearson_similarity']:.4f}    |    "
            f"Cosine = "
            f"{row['cosine_similarity']:.4f}    |    "
            f"Top-20% IoU = "
            f"{row['attention_iou']:.4f}"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.90
        ]
    )

    plt.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. GENERATE ALL REPRESENTATIVE FIGURES
# ============================================================

print()
print("=" * 70)
print("GENERATING REPRESENTATIVE FIGURES")
print("=" * 70)

figure_records = []

for case_name, row in representative_cases.items():

    output_path = (
        OUTPUT_ROOT
        / f"{case_name}.png"
    )

    create_comparison_figure(
        row,
        case_name,
        output_path
    )

    figure_records.append({

        "case":
            case_name,

        "image_path":
            row["image_path"],

        "true_class":
            row["true_class"],

        "efficientnet_prediction":
            row["eff_predicted_class"],

        "efficientnet_confidence":
            row["eff_confidence"],

        "vit_prediction":
            row["vit_predicted_class"],

        "vit_confidence":
            row["vit_confidence"],

        "pearson_similarity":
            row["pearson_similarity"],

        "cosine_similarity":
            row["cosine_similarity"],

        "attention_iou":
            row["attention_iou"],

        "figure_path":
            str(output_path),
    })

    print(
        f"✓ {case_name}"
    )


# ============================================================
# 11. SAVE REPRESENTATIVE CASE TABLE
# ============================================================

figure_df = pd.DataFrame(
    figure_records
)

figure_csv = (
    OUTPUT_ROOT
    / "representative_cases.csv"
)

figure_df.to_csv(
    figure_csv,
    index=False
)


# ============================================================
# 12. SAVE JSON
# ============================================================

json_records = []

for record in figure_records:

    json_records.append(
        record
    )

summary = {

    "stage":
        "STEP 8.10",

    "analysis":
        "Representative XAI Figures",

    "models": [
        "EfficientNet-B0 Grad-CAM",
        "ViT-B/16 Attention Rollout"
    ],

    "cases_generated":
        len(figure_records),

    "cases":
        json_records
}

summary_json = (
    OUTPUT_ROOT
    / "representative_xai_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 13. FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 8.10 COMPLETED")
print("=" * 70)

print()

print(
    f"Figures generated : "
    f"{len(figure_records)}"
)

print()

print(
    f"Output directory  : "
    f"{OUTPUT_ROOT}"
)

print()

print(
    f"Representative CSV: "
    f"{figure_csv}"
)

print(
    f"Summary JSON       : "
    f"{summary_json}"
)

print()

print(
    "✓ Representative XAI figures generated."
)

print(
    "✓ EfficientNet and ViT explanations "
    "paired on the same X-rays."
)

print(
    "✓ Figures are ready for visual inspection."
)

print()

print(
    "Next stage:"
)

print(
    "STEP 8.11 — XAI ARTIFACT / REGION AUDIT"
)