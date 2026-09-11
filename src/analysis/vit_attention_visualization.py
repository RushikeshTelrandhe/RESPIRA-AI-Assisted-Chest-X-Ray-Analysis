# ============================================================
# Respira — STEP 8.8
# ViT ATTENTION VISUALIZATION
# ============================================================

from pathlib import Path
import json
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROSS_ATTENTION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "cross_attention"
    / "test"
)

FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
)

PREDICTIONS_PATH = (
    FINAL_ROOT
    / "predictions"
    / "final_predictions.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "vit_attention"
)

HEATMAP_ROOT = (
    OUTPUT_ROOT
    / "heatmaps"
)

OVERLAY_ROOT = (
    OUTPUT_ROOT
    / "overlays"
)

REPORT_ROOT = (
    OUTPUT_ROOT
    / "reports"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "vit_attention_summary.json"
)

SUMMARY_TEXT_PATH = (
    OUTPUT_ROOT
    / "vit_attention_summary.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

IMAGE_SIZE = 224

PATCH_GRID = 14

NUM_PATCHES = 196

MAX_IMAGES = 50


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
# CREATE DIRECTORIES
# ============================================================

HEATMAP_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OVERLAY_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.8")
print("VIT ATTENTION VISUALIZATION")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Input root   : {CROSS_ATTENTION_ROOT}")
print(f"Output root  : {OUTPUT_ROOT}")


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)


VIT_TO_CNN_PATH = (
    CROSS_ATTENTION_ROOT
    / "vit_to_cnn_attention.pt"
)

CNN_TO_VIT_PATH = (
    CROSS_ATTENTION_ROOT
    / "cnn_to_vit_attention.pt"
)


if not VIT_TO_CNN_PATH.exists():

    raise FileNotFoundError(
        f"\nViT → CNN attention not found:\n"
        f"{VIT_TO_CNN_PATH}"
    )


if not CNN_TO_VIT_PATH.exists():

    raise FileNotFoundError(
        f"\nCNN → ViT attention not found:\n"
        f"{CNN_TO_VIT_PATH}"
    )


print("✓ ViT → CNN attention found")
print(f"  {VIT_TO_CNN_PATH}")

print()
print("✓ CNN → ViT attention found")
print(f"  {CNN_TO_VIT_PATH}")


# ============================================================
# LOAD ATTENTION
# ============================================================

print()
print("=" * 70)
print("LOADING ATTENTION MAPS")
print("=" * 70)


vit_to_cnn = torch.load(
    VIT_TO_CNN_PATH,
    map_location="cpu"
)

cnn_to_vit = torch.load(
    CNN_TO_VIT_PATH,
    map_location="cpu"
)


if not torch.is_tensor(vit_to_cnn):

    raise TypeError(
        "vit_to_cnn_attention.pt does not contain a tensor."
    )


if not torch.is_tensor(cnn_to_vit):

    raise TypeError(
        "cnn_to_vit_attention.pt does not contain a tensor."
    )


print()
print(
    "ViT → CNN shape:",
    tuple(vit_to_cnn.shape)
)

print(
    "CNN → ViT shape:",
    tuple(cnn_to_vit.shape)
)


# ============================================================
# VALIDATE SHAPES
# ============================================================

if vit_to_cnn.ndim != 3:

    raise ValueError(
        "Expected ViT → CNN attention to have 3 dimensions."
    )


if cnn_to_vit.ndim != 3:

    raise ValueError(
        "Expected CNN → ViT attention to have 3 dimensions."
    )


N = min(
    vit_to_cnn.shape[0],
    cnn_to_vit.shape[0]
)


print()
print(f"Samples available: {N}")


# ============================================================
# LOAD PREDICTIONS
# ============================================================

prediction_rows = []


if PREDICTIONS_PATH.exists():

    print()
    print("✓ Loading final predictions")

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        prediction_rows = list(reader)

    print(
        f"Prediction rows: {len(prediction_rows)}"
    )

else:

    print()
    print(
        "⚠ Final prediction CSV not found."
    )

    print(
        "Continuing without prediction metadata."
    )


# ============================================================
# HELPER — NORMALIZE
# ============================================================

def normalize_map(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-12:

        return np.zeros_like(values)

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# HELPER — CREATE IMAGE
# ============================================================

def load_image(index):

    if index >= len(prediction_rows):

        return None

    row = prediction_rows[index]

    image_path = row.get(
        "image_path",
        ""
    )

    if not image_path:

        return None

    path = Path(image_path)

    if not path.exists():

        return None

    try:

        image = Image.open(path).convert(
            "RGB"
        )

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        return np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    except Exception:

        return None


# ============================================================
# CREATE VIT ATTENTION MAP
# ============================================================

def create_vit_map(
    attention
):

    attention = torch.as_tensor(
        attention
    ).float()

    # Expected:
    # [196, 49]
    #
    # Each ViT token attends to
    # CNN spatial tokens.

    if attention.ndim != 2:

        raise ValueError(
            "Expected attention matrix [196, 49]."
        )

    # Aggregate CNN locations for each
    # ViT token.

    token_attention = attention.mean(
        dim=1
    )

    token_attention = token_attention[
        :NUM_PATCHES
    ]

    if token_attention.numel() != NUM_PATCHES:

        raise ValueError(
            f"Expected {NUM_PATCHES} ViT tokens, "
            f"got {token_attention.numel()}."
        )

    token_attention = token_attention.numpy()

    token_attention = normalize_map(
        token_attention
    )

    return token_attention.reshape(
        PATCH_GRID,
        PATCH_GRID
    )


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING VIT ATTENTION VISUALIZATIONS")
print("=" * 70)


num_visualized = min(
    N,
    MAX_IMAGES
)

results = []


for index in range(
    num_visualized
):

    try:

        attention_map = create_vit_map(
            vit_to_cnn[index]
        )

    except Exception as error:

        print(
            f"⚠ Skipping sample {index}: {error}"
        )

        continue


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = load_image(
        index
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    heatmap_path = (
        HEATMAP_ROOT
        / f"vit_attention_{index:05d}.png"
    )


    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        attention_map,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title(
        "ViT Attention Map"
    )

    plt.tight_layout()

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay_path = (
        OVERLAY_ROOT
        / f"vit_attention_overlay_{index:05d}.png"
    )


    if image is not None:

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            image
        )

        plt.imshow(
            attention_map,
            extent=[
                0,
                IMAGE_SIZE,
                IMAGE_SIZE,
                0
            ],
            alpha=0.45,
            interpolation="bilinear"
        )

        plt.axis("off")

        plt.title(
            "Chest X-Ray + ViT Attention"
        )

        plt.tight_layout()

        plt.savefig(
            overlay_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    row = (
        prediction_rows[index]
        if index < len(prediction_rows)
        else {}
    )


    result = {

        "index": index,

        "image_path": row.get(
            "image_path",
            ""
        ),

        "true_class": row.get(
            "true_class",
            ""
        ),

        "predicted_class": row.get(
            "predicted_class",
            ""
        ),

        "confidence": row.get(
            "prediction_confidence",
            ""
        ),

        "heatmap": str(
            heatmap_path
        ),

        "overlay": str(
            overlay_path
        ),

        "mean_attention": float(
            attention_map.mean()
        ),

        "maximum_attention": float(
            attention_map.max()
        ),

        "minimum_attention": float(
            attention_map.min()
        ),
    }


    results.append(
        result
    )


    if (
        index + 1
    ) % 10 == 0:

        print(
            f"Processed {index + 1}/{num_visualized}"
        )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_csv = (
    REPORT_ROOT
    / "vit_attention_results.csv"
)


with open(
    results_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [

        "index",
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "heatmap",
        "overlay",
        "mean_attention",
        "maximum_attention",
        "minimum_attention",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print(
    f"✓ Results saved:"
)
print(
    f"  {results_csv}"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage": "STEP 8.8",

    "pipeline":
        "ViT Attention Visualization",

    "attention_source":
        "Bidirectional Cross-Attention",

    "test_samples":
        int(N),

    "visualized_samples":
        int(len(results)),

    "vit_attention_shape":
        list(
            vit_to_cnn.shape
        ),

    "cnn_to_vit_attention_shape":
        list(
            cnn_to_vit.shape
        ),

    "patch_grid": [
        PATCH_GRID,
        PATCH_GRID
    ],

    "num_vit_patches":
        NUM_PATCHES,

    "outputs": {

        "heatmaps":
            str(HEATMAP_ROOT),

        "overlays":
            str(OVERLAY_ROOT),

        "results":
            str(results_csv),
    }
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(
    SUMMARY_TEXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — STEP 8.8\n"
    )

    f.write(
        "VIT ATTENTION VISUALIZATION\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Test samples       : {N}\n"
    )

    f.write(
        f"Visualized samples : {len(results)}\n"
    )

    f.write(
        f"ViT patches        : {NUM_PATCHES}\n"
    )

    f.write(
        f"Patch grid         : "
        f"{PATCH_GRID} x {PATCH_GRID}\n"
    )

    f.write(
        "\nAttention source:\n"
    )

    f.write(
        "Bidirectional Cross-Attention\n"
    )

    f.write(
        "\nOutputs:\n"
    )

    f.write(
        f"Heatmaps : {HEATMAP_ROOT}\n"
    )

    f.write(
        f"Overlays : {OVERLAY_ROOT}\n"
    )

    f.write(
        f"Results  : {results_csv}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 8.8 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("ViT attention outputs:")
print(
    OUTPUT_ROOT
)

print()
print("Generated:")

print(
    "  ✓ ViT attention heatmaps"
)

print(
    "  ✓ Chest X-Ray attention overlays"
)

print(
    "  ✓ Attention results CSV"
)

print(
    "  ✓ JSON summary"
)

print(
    "  ✓ Text summary"
)

print()
print(
    f"Test samples      : {N}"
)

print(
    f"Visualized samples: {len(results)}"
)

print()
print(
    "✓ ViT attention visualization completed."
)

print()
print(
    "Next stage:"
)

print(
    "STEP 8.9 — FINAL EXPLAINABILITY ANALYSIS"
)