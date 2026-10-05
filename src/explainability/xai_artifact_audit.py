# ============================================================
# RESPIRA — STEP 8.11
# XAI ARTIFACT / REGION AUDIT
#
# Quantitative audit of Grad-CAM and ViT Attention Rollout
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "reports"
    / "xai_comparison_results.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "artifact_audit"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Central region:
# 60% x 60% of image
CENTER_RATIO = 0.60

# Border width:
# 10% of image dimensions
BORDER_RATIO = 0.10

# Corner region:
# 20% x 20% corners
CORNER_RATIO = 0.20

EPS = 1e-8


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.11")
print("XAI ARTIFACT / REGION AUDIT")
print("=" * 70)

print()
print("Comparison CSV:")
print(COMPARISON_CSV)

print()
print("Output:")
print(OUTPUT_ROOT)


# ============================================================
# 4. VALIDATE INPUT
# ============================================================

if not COMPARISON_CSV.exists():

    raise FileNotFoundError(
        f"\nXAI comparison file not found:\n"
        f"{COMPARISON_CSV}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(
    COMPARISON_CSV
)

print()
print(
    f"Images to audit: {len(df)}"
)


# ============================================================
# 6. HEATMAP LOADER
# ============================================================

def load_heatmap(path):

    image = Image.open(
        path
    ).convert("L")

    heatmap = np.asarray(
        image
    ).astype(np.float32)

    heatmap -= heatmap.min()

    maximum = heatmap.max()

    if maximum > EPS:
        heatmap /= maximum

    return heatmap


# ============================================================
# 7. REGION MASKS
# ============================================================

def create_masks(height, width):

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center_h = int(
        height * CENTER_RATIO
    )

    center_w = int(
        width * CENTER_RATIO
    )

    y0 = (
        height - center_h
    ) // 2

    x0 = (
        width - center_w
    ) // 2

    center_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    center_mask[
        y0:y0 + center_h,
        x0:x0 + center_w
    ] = True

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    border_h = int(
        height * BORDER_RATIO
    )

    border_w = int(
        width * BORDER_RATIO
    )

    border_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    border_mask[
        :border_h,
        :
    ] = True

    border_mask[
        -border_h:,
        :
    ] = True

    border_mask[
        :,
        :border_w
    ] = True

    border_mask[
        :,
        -border_w:
    ] = True

    # --------------------------------------------------------
    # Corners
    # --------------------------------------------------------

    corner_h = int(
        height * CORNER_RATIO
    )

    corner_w = int(
        width * CORNER_RATIO
    )

    corner_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    corner_mask[
        :corner_h,
        :corner_w
    ] = True

    corner_mask[
        :corner_h,
        -corner_w:
    ] = True

    corner_mask[
        -corner_h:,
        :corner_w
    ] = True

    corner_mask[
        -corner_h:,
        -corner_w:
    ] = True

    # --------------------------------------------------------
    # Peripheral region
    # --------------------------------------------------------

    peripheral_mask = ~center_mask

    return (
        center_mask,
        border_mask,
        corner_mask,
        peripheral_mask
    )


# ============================================================
# 8. REGION STATISTICS
# ============================================================

def region_statistics(
    heatmap
):

    h, w = heatmap.shape

    (
        center_mask,
        border_mask,
        corner_mask,
        peripheral_mask
    ) = create_masks(
        h,
        w
    )

    total = (
        heatmap.sum()
        + EPS
    )

    center_activation = (
        heatmap[center_mask].sum()
        / total
    )

    border_activation = (
        heatmap[border_mask].sum()
        / total
    )

    corner_activation = (
        heatmap[corner_mask].sum()
        / total
    )

    peripheral_activation = (
        heatmap[peripheral_mask].sum()
        / total
    )

    # --------------------------------------------------------
    # Activation entropy
    # --------------------------------------------------------

    values = (
        heatmap.flatten()
        + EPS
    )

    probabilities = (
        values
        / values.sum()
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    normalized_entropy = (
        entropy
        / np.log(
            len(probabilities)
        )
    )

    # --------------------------------------------------------
    # Maximum activation location
    # --------------------------------------------------------

    max_index = np.argmax(
        heatmap
    )

    max_y, max_x = np.unravel_index(
        max_index,
        heatmap.shape
    )

    max_y_ratio = (
        max_y
        / h
    )

    max_x_ratio = (
        max_x
        / w
    )

    return {

        "center_activation":
            center_activation,

        "border_activation":
            border_activation,

        "corner_activation":
            corner_activation,

        "peripheral_activation":
            peripheral_activation,

        "normalized_entropy":
            normalized_entropy,

        "max_x_ratio":
            max_x_ratio,

        "max_y_ratio":
            max_y_ratio,
    }


# ============================================================
# 9. AUDIT ALL IMAGES
# ============================================================

records = []

print()
print(
    "Running region audit..."
)

for index, row in df.iterrows():

    try:

        eff_heatmap = load_heatmap(
            row["eff_heatmap_path"]
        )

        vit_heatmap = load_heatmap(
            row["vit_heatmap_path"]
        )

        eff_stats = region_statistics(
            eff_heatmap
        )

        vit_stats = region_statistics(
            vit_heatmap
        )

        record = {

            "image_path":
                row["image_path"],

            "true_class":
                row["true_class"],

            "eff_predicted_class":
                row["eff_predicted_class"],

            "vit_predicted_class":
                row["vit_predicted_class"],

            "eff_correct":
                row["eff_correct"],

            "vit_correct":
                row["vit_correct"],

            "eff_confidence":
                row["eff_confidence"],

            "vit_confidence":
                row["vit_confidence"],

            # EfficientNet
            "eff_center_activation":
                eff_stats[
                    "center_activation"
                ],

            "eff_border_activation":
                eff_stats[
                    "border_activation"
                ],

            "eff_corner_activation":
                eff_stats[
                    "corner_activation"
                ],

            "eff_peripheral_activation":
                eff_stats[
                    "peripheral_activation"
                ],

            "eff_entropy":
                eff_stats[
                    "normalized_entropy"
                ],

            "eff_max_x":
                eff_stats[
                    "max_x_ratio"
                ],

            "eff_max_y":
                eff_stats[
                    "max_y_ratio"
                ],

            # ViT
            "vit_center_activation":
                vit_stats[
                    "center_activation"
                ],

            "vit_border_activation":
                vit_stats[
                    "border_activation"
                ],

            "vit_corner_activation":
                vit_stats[
                    "corner_activation"
                ],

            "vit_peripheral_activation":
                vit_stats[
                    "peripheral_activation"
                ],

            "vit_entropy":
                vit_stats[
                    "normalized_entropy"
                ],

            "vit_max_x":
                vit_stats[
                    "max_x_ratio"
                ],

            "vit_max_y":
                vit_stats[
                    "max_y_ratio"
                ],
        }

        records.append(
            record
        )

    except Exception as e:

        print(
            f"WARNING: failed image "
            f"{index}: {e}"
        )


# ============================================================
# 10. CREATE DATAFRAME
# ============================================================

audit_df = pd.DataFrame(
    records
)

print()
print(
    f"Successfully audited: "
    f"{len(audit_df)}"
)


# ============================================================
# 11. SAVE COMPLETE RESULTS
# ============================================================

results_csv = (
    OUTPUT_ROOT
    / "xai_region_audit.csv"
)

audit_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# 12. SUMMARY STATISTICS
# ============================================================

summary = {

    "images_audited":
        int(len(audit_df)),

    "efficientnet": {

        "mean_center_activation":
            float(
                audit_df[
                    "eff_center_activation"
                ].mean()
            ),

        "mean_border_activation":
            float(
                audit_df[
                    "eff_border_activation"
                ].mean()
            ),

        "mean_corner_activation":
            float(
                audit_df[
                    "eff_corner_activation"
                ].mean()
            ),

        "mean_peripheral_activation":
            float(
                audit_df[
                    "eff_peripheral_activation"
                ].mean()
            ),

        "mean_entropy":
            float(
                audit_df[
                    "eff_entropy"
                ].mean()
            ),
    },

    "vit": {

        "mean_center_activation":
            float(
                audit_df[
                    "vit_center_activation"
                ].mean()
            ),

        "mean_border_activation":
            float(
                audit_df[
                    "vit_border_activation"
                ].mean()
            ),

        "mean_corner_activation":
            float(
                audit_df[
                    "vit_corner_activation"
                ].mean()
            ),

        "mean_peripheral_activation":
            float(
                audit_df[
                    "vit_peripheral_activation"
                ].mean()
            ),

        "mean_entropy":
            float(
                audit_df[
                    "vit_entropy"
                ].mean()
            ),
    }
}


# ============================================================
# 13. SUSPICIOUS CASES
# ============================================================

# High border activation
border_threshold = (
    audit_df[
        [
            "eff_border_activation",
            "vit_border_activation"
        ]
    ].stack().quantile(
        0.95
    )
)

# High corner activation
corner_threshold = (
    audit_df[
        [
            "eff_corner_activation",
            "vit_corner_activation"
        ]
    ].stack().quantile(
        0.95
    )
)


audit_df[
    "suspicious_border"
] = (
    (audit_df[
        "eff_border_activation"
    ] >= border_threshold)
    |
    (audit_df[
        "vit_border_activation"
    ] >= border_threshold)
)


audit_df[
    "suspicious_corner"
] = (
    (audit_df[
        "eff_corner_activation"
    ] >= corner_threshold)
    |
    (audit_df[
        "vit_corner_activation"
    ] >= corner_threshold)
)


suspicious_df = audit_df[
    audit_df[
        "suspicious_border"
    ]
    |
    audit_df[
        "suspicious_corner"
    ]
].copy()


suspicious_csv = (
    OUTPUT_ROOT
    / "suspicious_xai_cases.csv"
)

suspicious_df.to_csv(
    suspicious_csv,
    index=False
)


# ============================================================
# 14. PER-CLASS SUMMARY
# ============================================================

class_summary = (
    audit_df
    .groupby("true_class")
    .agg({

        "eff_center_activation":
            "mean",

        "eff_border_activation":
            "mean",

        "eff_corner_activation":
            "mean",

        "eff_entropy":
            "mean",

        "vit_center_activation":
            "mean",

        "vit_border_activation":
            "mean",

        "vit_corner_activation":
            "mean",

        "vit_entropy":
            "mean",
    })
    .reset_index()
)

class_csv = (
    OUTPUT_ROOT
    / "xai_region_by_class.csv"
)

class_summary.to_csv(
    class_csv,
    index=False
)


# ============================================================
# 15. SUMMARY JSON
# ============================================================

summary[
    "border_95th_percentile_threshold"
] = float(
    border_threshold
)

summary[
    "corner_95th_percentile_threshold"
] = float(
    corner_threshold
)

summary[
    "suspicious_cases"
] = int(
    len(suspicious_df)
)

summary[
    "suspicious_border_cases"
] = int(
    audit_df[
        "suspicious_border"
    ].sum()
)

summary[
    "suspicious_corner_cases"
] = int(
    audit_df[
        "suspicious_corner"
    ].sum()
)

summary_json = (
    OUTPUT_ROOT
    / "xai_region_audit_summary.json"
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
# 16. PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("XAI REGION AUDIT RESULTS")
print("=" * 70)

print()

print("EfficientNet-B0:")
print(
    f"  Center activation     : "
    f"{summary['efficientnet']['mean_center_activation']:.4f}"
)

print(
    f"  Border activation     : "
    f"{summary['efficientnet']['mean_border_activation']:.4f}"
)

print(
    f"  Corner activation     : "
    f"{summary['efficientnet']['mean_corner_activation']:.4f}"
)

print(
    f"  Peripheral activation : "
    f"{summary['efficientnet']['mean_peripheral_activation']:.4f}"
)

print(
    f"  Entropy               : "
    f"{summary['efficientnet']['mean_entropy']:.4f}"
)

print()

print("ViT-B/16:")
print(
    f"  Center activation     : "
    f"{summary['vit']['mean_center_activation']:.4f}"
)

print(
    f"  Border activation     : "
    f"{summary['vit']['mean_border_activation']:.4f}"
)

print(
    f"  Corner activation     : "
    f"{summary['vit']['mean_corner_activation']:.4f}"
)

print(
    f"  Peripheral activation : "
    f"{summary['vit']['mean_peripheral_activation']:.4f}"
)

print(
    f"  Entropy               : "
    f"{summary['vit']['mean_entropy']:.4f}"
)

print()

print(
    f"95th percentile border threshold : "
    f"{border_threshold:.4f}"
)

print(
    f"95th percentile corner threshold : "
    f"{corner_threshold:.4f}"
)

print()

print(
    f"Suspicious cases : "
    f"{len(suspicious_df)}"
)

print()

print("Outputs:")
print(
    f"  Complete audit : {results_csv}"
)

print(
    f"  Suspicious     : {suspicious_csv}"
)

print(
    f"  Per-class      : {class_csv}"
)

print(
    f"  Summary        : {summary_json}"
)

print()

print("=" * 70)
print("STEP 8.11 COMPLETED")
print("=" * 70)

print()
print(
    "✓ Region-level XAI audit completed."
)

print(
    "✓ Border and corner activation analyzed."
)

print(
    "✓ Potential artifact cases identified."
)

print()
print(
    "Next stage:"
)

print(
    "STEP 8.12 — XAI AUDIT VISUALIZATION"
)