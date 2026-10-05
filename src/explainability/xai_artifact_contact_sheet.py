# ============================================================
# RESPIRA — STEP 8.13
# XAI ARTIFACT / REGION AUDIT CONTACT SHEETS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "artifact_audit"
)

VIS_ROOT = (
    AUDIT_ROOT
    / "visualizations"
)

COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "reports"
    / "xai_comparison_results.csv"
)

OUTPUT_ROOT = (
    AUDIT_ROOT
    / "contact_sheets"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

TOP_N = 12
DPI = 200


# ============================================================
# LOAD ORIGINAL XAI COMPARISON
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.13")
print("XAI ARTIFACT / REGION AUDIT CONTACT SHEETS")
print("=" * 70)

print()
print("Audit files:")
print(VIS_ROOT)

print()
print("Original XAI comparison:")
print(COMPARISON_CSV)


if not COMPARISON_CSV.exists():

    raise FileNotFoundError(
        f"\nOriginal XAI comparison file not found:\n"
        f"{COMPARISON_CSV}"
    )


# ============================================================
# LOAD DATA
# ============================================================

comparison_df = pd.read_csv(
    COMPARISON_CSV
)

border_csv = (
    VIS_ROOT
    / "top_50_border_cases.csv"
)

corner_csv = (
    VIS_ROOT
    / "top_50_corner_cases.csv"
)


if not border_csv.exists():

    raise FileNotFoundError(
        f"\nBorder cases file not found:\n"
        f"{border_csv}"
    )


if not corner_csv.exists():

    raise FileNotFoundError(
        f"\nCorner cases file not found:\n"
        f"{corner_csv}"
    )


border_df = pd.read_csv(
    border_csv
)

corner_df = pd.read_csv(
    corner_csv
)


print()
print(
    f"Original XAI records : {len(comparison_df)}"
)

print(
    f"Border audit records : {len(border_df)}"
)

print(
    f"Corner audit records : {len(corner_df)}"
)


# ============================================================
# MERGE HEATMAP PATHS
# ============================================================

required_paths = [
    "image_path",
    "eff_heatmap_path",
    "vit_heatmap_path",
    "eff_overlay_path",
    "vit_overlay_path",
]


missing = [
    column
    for column in required_paths
    if column not in comparison_df.columns
]

if missing:

    raise ValueError(
        "\nOriginal XAI CSV is missing required columns:\n"
        + "\n".join(missing)
    )


path_df = comparison_df[
    required_paths
].copy()


# ------------------------------------------------------------
# Avoid duplicate columns
# ------------------------------------------------------------

border_df = border_df.drop(
    columns=[
        "eff_heatmap_path",
        "vit_heatmap_path",
        "eff_overlay_path",
        "vit_overlay_path",
    ],
    errors="ignore"
)

corner_df = corner_df.drop(
    columns=[
        "eff_heatmap_path",
        "vit_heatmap_path",
        "eff_overlay_path",
        "vit_overlay_path",
    ],
    errors="ignore"
)


# ============================================================
# MERGE
# ============================================================

border_df = border_df.merge(
    path_df,
    on="image_path",
    how="left",
    validate="one_to_one"
)

corner_df = corner_df.merge(
    path_df,
    on="image_path",
    how="left",
    validate="one_to_one"
)


print()
print(
    "Heatmap paths successfully restored."
)


# ============================================================
# VALIDATE PATHS
# ============================================================

for name, data in [
    ("border", border_df),
    ("corner", corner_df),
]:

    missing_eff = (
        data["eff_heatmap_path"]
        .isna()
        .sum()
    )

    missing_vit = (
        data["vit_heatmap_path"]
        .isna()
        .sum()
    )

    print()
    print(
        f"{name.capitalize()} cases:"
    )

    print(
        f"  Missing EfficientNet paths: "
        f"{missing_eff}"
    )

    print(
        f"  Missing ViT paths: "
        f"{missing_vit}"
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_gray(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    return np.asarray(
        Image.open(path)
        .convert("L")
    ).astype(np.float32)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(arr):

    arr = arr - arr.min()

    maximum = arr.max()

    if maximum > 0:

        arr = arr / maximum

    return arr


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    data,
    title,
    output_path,
):

    data = data.head(
        TOP_N
    )

    fig, axes = plt.subplots(
        TOP_N,
        3,
        figsize=(
            12,
            TOP_N * 3.3
        )
    )

    for i, (_, row) in enumerate(
        data.iterrows()
    ):

        # ----------------------------------------------------
        # LOAD IMAGES
        # ----------------------------------------------------

        original = load_gray(
            row["image_path"]
        )

        eff = normalize(
            load_gray(
                row["eff_heatmap_path"]
            )
        )

        vit = normalize(
            load_gray(
                row["vit_heatmap_path"]
            )
        )

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        axes[i, 0].imshow(
            original,
            cmap="gray"
        )

        axes[i, 0].set_title(
            (
                f"Original\n"
                f"True: {row['true_class']}"
            ),
            fontsize=9
        )

        axes[i, 0].axis("off")

        # ----------------------------------------------------
        # EFFICIENTNET
        # ----------------------------------------------------

        axes[i, 1].imshow(
            original,
            cmap="gray"
        )

        axes[i, 1].imshow(
            eff,
            cmap="jet",
            alpha=0.45
        )

        axes[i, 1].set_title(
            (
                f"EfficientNet-B0\n"
                f"Pred: {row['eff_predicted_class']}\n"
                f"Border: "
                f"{row['eff_border_activation']:.3f}\n"
                f"Corner: "
                f"{row['eff_corner_activation']:.3f}"
            ),
            fontsize=9
        )

        axes[i, 1].axis("off")

        # ----------------------------------------------------
        # VIT
        # ----------------------------------------------------

        axes[i, 2].imshow(
            original,
            cmap="gray"
        )

        axes[i, 2].imshow(
            vit,
            cmap="jet",
            alpha=0.45
        )

        axes[i, 2].set_title(
            (
                f"ViT-B/16\n"
                f"Pred: {row['vit_predicted_class']}\n"
                f"Border: "
                f"{row['vit_border_activation']:.3f}\n"
                f"Corner: "
                f"{row['vit_corner_activation']:.3f}"
            ),
            fontsize=9
        )

        axes[i, 2].axis("off")

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.98
        ]
    )

    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# BORDER CASES
# ============================================================

print()
print(
    "Generating border-activation contact sheet..."
)

create_contact_sheet(
    border_df,
    "Top Border-Activation XAI Cases",
    OUTPUT_ROOT
    / "top_border_cases.png"
)


# ============================================================
# CORNER CASES
# ============================================================

print(
    "Generating corner-activation contact sheet..."
)

create_contact_sheet(
    corner_df,
    "Top Corner-Activation XAI Cases",
    OUTPUT_ROOT
    / "top_corner_cases.png"
)


# ============================================================
# SAVE MERGED DATA
# ============================================================

border_df.to_csv(
    OUTPUT_ROOT
    / "top_50_border_cases_with_paths.csv",
    index=False
)

corner_df.to_csv(
    OUTPUT_ROOT
    / "top_50_corner_cases_with_paths.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 8.13 COMPLETED")
print("=" * 70)

print()

print(
    "Generated:"
)

print(
    OUTPUT_ROOT
    / "top_border_cases.png"
)

print(
    OUTPUT_ROOT
    / "top_corner_cases.png"
)

print()

print(
    "Merged inspection CSVs:"
)

print(
    OUTPUT_ROOT
    / "top_50_border_cases_with_paths.csv"
)

print(
    OUTPUT_ROOT
    / "top_50_corner_cases_with_paths.csv"
)

print()

print(
    "✓ Original XAI heatmap paths restored."
)

print(
    "✓ Top border cases visualized."
)

print(
    "✓ Top corner cases visualized."
)

print()
print(
    "Next: visually inspect both contact sheets."
)