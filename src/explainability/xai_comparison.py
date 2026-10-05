# ============================================================
# Respira — STEP 8.9
# Paired XAI Comparison
# EfficientNet Grad-CAM vs ViT Attention Rollout
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

EFF_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
    / "explainability"
    / "grad_cam"
)

VIT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
    / "explainability"
    / "attention"
)

EFF_RESULTS = (
    EFF_ROOT
    / "reports"
    / "grad_cam_results.csv"
)

VIT_RESULTS = (
    VIT_ROOT
    / "reports"
    / "vit_attention_results.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
)

PAIRED_ROOT = (
    OUTPUT_ROOT
    / "paired_examples"
)

REPORT_ROOT = (
    OUTPUT_ROOT
    / "reports"
)

for directory in [
    OUTPUT_ROOT,
    PAIRED_ROOT,
    REPORT_ROOT,
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

TOP_PERCENT = 20


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.9")
print("PAIRED XAI COMPARISON")
print("EfficientNet Grad-CAM vs ViT Attention Rollout")
print("=" * 70)


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

for path in [
    EFF_RESULTS,
    VIT_RESULTS,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# 5. LOAD RESULTS
# ============================================================

print("\nLoading XAI results...")

eff = pd.read_csv(
    EFF_RESULTS
)

vit = pd.read_csv(
    VIT_RESULTS
)

print(
    f"EfficientNet records: {len(eff)}"
)

print(
    f"ViT records         : {len(vit)}"
)


# ============================================================
# 6. NORMALIZE IMAGE PATHS
# ============================================================

eff["image_key"] = (
    eff["image_path"]
    .astype(str)
    .apply(
        lambda x: str(
            Path(x).resolve()
        ).lower()
    )
)

vit["image_key"] = (
    vit["image_path"]
    .astype(str)
    .apply(
        lambda x: str(
            Path(x).resolve()
        ).lower()
    )
)


# ============================================================
# 7. CHECK ALIGNMENT
# ============================================================

eff_keys = set(
    eff["image_key"]
)

vit_keys = set(
    vit["image_key"]
)

common_keys = (
    eff_keys
    & vit_keys
)

print(
    f"\nCommon images: {len(common_keys)}"
)

if (
    len(eff_keys - vit_keys) > 0
    or len(vit_keys - eff_keys) > 0
):

    raise RuntimeError(
        "EfficientNet and ViT image sets are not aligned."
    )


# ============================================================
# 8. MERGE PREDICTIONS
# ============================================================

eff_keep = eff[
    [
        "image_key",
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct_prediction",
        "heatmap_path",
        "overlay_path",
    ]
].copy()

eff_keep = eff_keep.rename(
    columns={
        "predicted_class":
            "eff_predicted_class",

        "confidence":
            "eff_confidence",

        "correct_prediction":
            "eff_correct",

        "heatmap_path":
            "eff_heatmap_path",

        "overlay_path":
            "eff_overlay_path",
    }
)

vit_keep = vit[
    [
        "image_key",
        "predicted_class",
        "confidence",
        "correct_prediction",
        "heatmap_path",
        "overlay_path",
    ]
].copy()

vit_keep = vit_keep.rename(
    columns={
        "predicted_class":
            "vit_predicted_class",

        "confidence":
            "vit_confidence",

        "correct_prediction":
            "vit_correct",

        "heatmap_path":
            "vit_heatmap_path",

        "overlay_path":
            "vit_overlay_path",
    }
)

merged = eff_keep.merge(
    vit_keep,
    on="image_key",
    how="inner"
)

print(
    f"Merged records: {len(merged)}"
)


# ============================================================
# 9. PREDICTION AGREEMENT
# ============================================================

merged["prediction_agreement"] = (
    merged["eff_predicted_class"]
    ==
    merged["vit_predicted_class"]
)

merged["both_correct"] = (
    merged["eff_correct"]
    &
    merged["vit_correct"]
)

merged["eff_correct_vit_wrong"] = (
    merged["eff_correct"]
    &
    ~merged["vit_correct"]
)

merged["eff_wrong_vit_correct"] = (
    ~merged["eff_correct"]
    &
    merged["vit_correct"]
)

merged["both_wrong"] = (
    ~merged["eff_correct"]
    &
    ~merged["vit_correct"]
)


# ============================================================
# 10. IMAGE / HEATMAP FUNCTIONS
# ============================================================

def load_gray_heatmap(path):

    image = Image.open(
        path
    ).convert("L")

    array = np.asarray(
        image
    ).astype(np.float32)

    # Normalize
    array -= array.min()

    maximum = array.max()

    if maximum > 0:
        array /= maximum

    return array


def load_xray(path):

    return np.asarray(
        Image.open(
            path
        ).convert("RGB")
    )


def pearson_similarity(
    a,
    b
):

    a = a.flatten()
    b = b.flatten()

    a_std = a.std()
    b_std = b.std()

    if (
        a_std == 0
        or b_std == 0
    ):
        return 0.0

    value = np.corrcoef(
        a,
        b
    )[0, 1]

    if np.isnan(value):
        return 0.0

    return float(value)


def cosine_similarity(
    a,
    b
):

    a = a.flatten()
    b = b.flatten()

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def top_attention_mask(
    heatmap,
    percent=20
):

    threshold = np.percentile(
        heatmap,
        100 - percent
    )

    return heatmap >= threshold


def attention_iou(
    a,
    b,
    percent=20
):

    mask_a = top_attention_mask(
        a,
        percent
    )

    mask_b = top_attention_mask(
        b,
        percent
    )

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# 11. COMPUTE XAI SIMILARITY
# ============================================================

print("\nComputing XAI similarity...")

xai_records = []

for counter, row in enumerate(
    merged.itertuples(),
    start=1
):

    eff_heatmap = load_gray_heatmap(
        row.eff_heatmap_path
    )

    vit_heatmap = load_gray_heatmap(
        row.vit_heatmap_path
    )

    # Ensure same dimensions
    if (
        eff_heatmap.shape
        !=
        vit_heatmap.shape
    ):

        raise RuntimeError(
            "Heatmap dimensions differ for "
            f"{row.image_path}"
        )

    pearson = pearson_similarity(
        eff_heatmap,
        vit_heatmap
    )

    cosine = cosine_similarity(
        eff_heatmap,
        vit_heatmap
    )

    iou = attention_iou(
        eff_heatmap,
        vit_heatmap,
        TOP_PERCENT
    )

    xai_records.append({

        "image_path":
            row.image_path,

        "true_class":
            row.true_class,

        "eff_predicted_class":
            row.eff_predicted_class,

        "vit_predicted_class":
            row.vit_predicted_class,

        "eff_confidence":
            row.eff_confidence,

        "vit_confidence":
            row.vit_confidence,

        "eff_correct":
            row.eff_correct,

        "vit_correct":
            row.vit_correct,

        "prediction_agreement":
            row.prediction_agreement,

        "both_correct":
            row.both_correct,

        "eff_correct_vit_wrong":
            row.eff_correct_vit_wrong,

        "eff_wrong_vit_correct":
            row.eff_wrong_vit_correct,

        "both_wrong":
            row.both_wrong,

        "pearson_similarity":
            pearson,

        "cosine_similarity":
            cosine,

        "attention_iou":
            iou,

        "eff_heatmap_path":
            row.eff_heatmap_path,

        "vit_heatmap_path":
            row.vit_heatmap_path,

        "eff_overlay_path":
            row.eff_overlay_path,

        "vit_overlay_path":
            row.vit_overlay_path,
    })

    if (
        counter % 250 == 0
        or counter == len(merged)
    ):

        print(
            f"Processed: "
            f"{counter}/{len(merged)}"
        )


comparison = pd.DataFrame(
    xai_records
)


# ============================================================
# 12. SAVE COMPLETE COMPARISON
# ============================================================

comparison_csv = (
    REPORT_ROOT
    / "xai_comparison_results.csv"
)

comparison.to_csv(
    comparison_csv,
    index=False
)


# ============================================================
# 13. OVERALL STATISTICS
# ============================================================

overall = {

    "total_images":
        int(len(comparison)),

    "prediction_agreement_count":
        int(
            comparison[
                "prediction_agreement"
            ].sum()
        ),

    "prediction_agreement_rate":
        float(
            comparison[
                "prediction_agreement"
            ].mean()
        ),

    "both_correct":
        int(
            comparison[
                "both_correct"
            ].sum()
        ),

    "eff_correct_vit_wrong":
        int(
            comparison[
                "eff_correct_vit_wrong"
            ].sum()
        ),

    "eff_wrong_vit_correct":
        int(
            comparison[
                "eff_wrong_vit_correct"
            ].sum()
        ),

    "both_wrong":
        int(
            comparison[
                "both_wrong"
            ].sum()
        ),

    "mean_pearson":
        float(
            comparison[
                "pearson_similarity"
            ].mean()
        ),

    "median_pearson":
        float(
            comparison[
                "pearson_similarity"
            ].median()
        ),

    "mean_cosine":
        float(
            comparison[
                "cosine_similarity"
            ].mean()
        ),

    "median_cosine":
        float(
            comparison[
                "cosine_similarity"
            ].median()
        ),

    "mean_attention_iou":
        float(
            comparison[
                "attention_iou"
            ].mean()
        ),

    "median_attention_iou":
        float(
            comparison[
                "attention_iou"
            ].median()
        ),

    "top_attention_percent":
        TOP_PERCENT,
}


# ============================================================
# 14. PER-CLASS XAI STATISTICS
# ============================================================

per_class = {}

for class_name in CLASS_NAMES:

    class_df = comparison[
        comparison[
            "true_class"
        ] == class_name
    ]

    if len(class_df) == 0:
        continue

    per_class[class_name] = {

        "samples":
            int(len(class_df)),

        "prediction_agreement":
            float(
                class_df[
                    "prediction_agreement"
                ].mean()
            ),

        "mean_pearson":
            float(
                class_df[
                    "pearson_similarity"
                ].mean()
            ),

        "mean_cosine":
            float(
                class_df[
                    "cosine_similarity"
                ].mean()
            ),

        "mean_attention_iou":
            float(
                class_df[
                    "attention_iou"
                ].mean()
            ),

        "both_correct":
            int(
                class_df[
                    "both_correct"
                ].sum()
            ),

        "eff_correct_vit_wrong":
            int(
                class_df[
                    "eff_correct_vit_wrong"
                ].sum()
            ),

        "eff_wrong_vit_correct":
            int(
                class_df[
                    "eff_wrong_vit_correct"
                ].sum()
            ),

        "both_wrong":
            int(
                class_df[
                    "both_wrong"
                ].sum()
            ),
    }


# ============================================================
# 15. CATEGORY COUNTS
# ============================================================

category_counts = {

    "both_correct":
        int(
            comparison[
                "both_correct"
            ].sum()
        ),

    "eff_correct_vit_wrong":
        int(
            comparison[
                "eff_correct_vit_wrong"
            ].sum()
        ),

    "eff_wrong_vit_correct":
        int(
            comparison[
                "eff_wrong_vit_correct"
            ].sum()
        ),

    "both_wrong":
        int(
            comparison[
                "both_wrong"
            ].sum()
        ),
}


# ============================================================
# 16. REPRESENTATIVE CASES
# ============================================================

representative = {}


def select_case(
    dataframe,
    condition,
    sort_column,
    ascending=False
):

    subset = dataframe[
        condition
    ].copy()

    if len(subset) == 0:
        return None

    subset = subset.sort_values(
        sort_column,
        ascending=ascending
    )

    return subset.iloc[0]


# ------------------------------------------------------------
# Both correct + highest agreement
# ------------------------------------------------------------

case = select_case(
    comparison,
    comparison["both_correct"],
    "pearson_similarity",
    ascending=False
)

if case is not None:
    representative[
        "both_correct_high_agreement"
    ] = case.to_dict()


# ------------------------------------------------------------
# Both correct + lowest agreement
# ------------------------------------------------------------

case = select_case(
    comparison,
    comparison["both_correct"],
    "pearson_similarity",
    ascending=True
)

if case is not None:
    representative[
        "both_correct_low_agreement"
    ] = case.to_dict()


# ------------------------------------------------------------
# EfficientNet correct / ViT wrong
# ------------------------------------------------------------

case = select_case(
    comparison,
    comparison["eff_correct_vit_wrong"],
    "pearson_similarity",
    ascending=True
)

if case is not None:
    representative[
        "efficientnet_correct_vit_wrong"
    ] = case.to_dict()


# ------------------------------------------------------------
# ViT correct / EfficientNet wrong
# ------------------------------------------------------------

case = select_case(
    comparison,
    comparison["eff_wrong_vit_correct"],
    "pearson_similarity",
    ascending=True
)

if case is not None:
    representative[
        "vit_correct_efficientnet_wrong"
    ] = case.to_dict()


# ------------------------------------------------------------
# Both wrong
# ------------------------------------------------------------

case = select_case(
    comparison,
    comparison["both_wrong"],
    "pearson_similarity",
    ascending=True
)

if case is not None:
    representative[
        "both_wrong"
    ] = case.to_dict()


# ============================================================
# 17. SAVE SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "STEP 8.9",

    "analysis":
        "Paired XAI Comparison",

    "models": [
        "EfficientNet-B0 Grad-CAM",
        "ViT-B/16 Attention Rollout"
    ],

    "images_compared":
        int(len(comparison)),

    "overall":
        overall,

    "category_counts":
        category_counts,

    "per_class":
        per_class,

    "representative_cases":
        representative
}

summary_json = (
    OUTPUT_ROOT
    / "xai_comparison_summary.json"
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
# 18. GENERATE AGREEMENT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    comparison[
        "pearson_similarity"
    ],
    bins=30
)

plt.xlabel(
    "Pearson Similarity"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "EfficientNet Grad-CAM vs ViT Attention Similarity"
)

plt.tight_layout()

histogram_path = (
    OUTPUT_ROOT
    / "xai_similarity_distribution.png"
)

plt.savefig(
    histogram_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. PER-CLASS SIMILARITY PLOT
# ============================================================

class_plot_data = []

for class_name in CLASS_NAMES:

    class_df = comparison[
        comparison[
            "true_class"
        ] == class_name
    ]

    if len(class_df) == 0:
        continue

    class_plot_data.append({
        "class": class_name,
        "pearson": class_df[
            "pearson_similarity"
        ].mean(),
        "cosine": class_df[
            "cosine_similarity"
        ].mean(),
        "iou": class_df[
            "attention_iou"
        ].mean(),
    })

class_plot_df = pd.DataFrame(
    class_plot_data
)

if len(class_plot_df) > 0:

    plt.figure(
        figsize=(11, 6)
    )

    x = np.arange(
        len(class_plot_df)
    )

    width = 0.25

    plt.bar(
        x - width,
        class_plot_df["pearson"],
        width,
        label="Pearson"
    )

    plt.bar(
        x,
        class_plot_df["cosine"],
        width,
        label="Cosine"
    )

    plt.bar(
        x + width,
        class_plot_df["iou"],
        width,
        label="Top-20% IoU"
    )

    plt.xticks(
        x,
        class_plot_df["class"],
        rotation=25,
        ha="right"
    )

    plt.ylabel(
        "Similarity"
    )

    plt.title(
        "Per-Class XAI Agreement"
    )

    plt.legend()

    plt.tight_layout()

    class_plot_path = (
        OUTPUT_ROOT
        / "per_class_xai_similarity.png"
    )

    plt.savefig(
        class_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 20. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("STEP 8.9 COMPLETED")
print("=" * 70)

print()

print(
    f"Images compared          : "
    f"{len(comparison)}"
)

print(
    f"Prediction agreement     : "
    f"{overall['prediction_agreement_rate']:.4f}"
)

print(
    f"Both correct             : "
    f"{overall['both_correct']}"
)

print(
    f"EfficientNet correct / "
    f"ViT wrong                : "
    f"{overall['eff_correct_vit_wrong']}"
)

print(
    f"ViT correct / "
    f"EfficientNet wrong       : "
    f"{overall['eff_wrong_vit_correct']}"
)

print(
    f"Both wrong               : "
    f"{overall['both_wrong']}"
)

print()

print(
    f"Mean Pearson similarity  : "
    f"{overall['mean_pearson']:.4f}"
)

print(
    f"Median Pearson           : "
    f"{overall['median_pearson']:.4f}"
)

print(
    f"Mean cosine similarity   : "
    f"{overall['mean_cosine']:.4f}"
)

print(
    f"Median cosine            : "
    f"{overall['median_cosine']:.4f}"
)

print(
    f"Mean top-20% IoU         : "
    f"{overall['mean_attention_iou']:.4f}"
)

print(
    f"Median top-20% IoU       : "
    f"{overall['median_attention_iou']:.4f}"
)

print()

print("Outputs:")

print(
    f"Comparison CSV : "
    f"{comparison_csv}"
)

print(
    f"Summary JSON   : "
    f"{summary_json}"
)

print(
    f"Distribution   : "
    f"{histogram_path}"
)

print()

print(
    "✓ EfficientNet and ViT explanations "
    "were compared image-by-image."
)

print(
    "✓ Prediction agreement calculated."
)

print(
    "✓ XAI similarity calculated."
)

print(
    "✓ Per-class XAI analysis calculated."
)

print(
    "✓ Representative cases identified."
)

print()

print(
    "Next stage:"
)

print(
    "STEP 8.10 — REPRESENTATIVE XAI FIGURES"
)