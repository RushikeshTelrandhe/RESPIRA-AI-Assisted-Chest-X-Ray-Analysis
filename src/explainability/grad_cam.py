# ============================================================
# Respira - STEP 8.7
# Grad-CAM Explainability for EfficientNet-B0
# ============================================================
#
# Purpose:
#   Generate Grad-CAM explanations for EfficientNet-B0
#   predictions on the Respira Chest X-Ray test dataset.
#
# Input:
#   data/processed/test/
#
# Model:
#   outputs/efficientnet_b0/checkpoints/best_model.pth
#
# Outputs:
#   outputs/final_analysis/grad_cam/
#
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

from src.models.efficientnet import create_efficientnet_b0


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
    / "grad_cam"
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

for directory in [
    OUTPUT_ROOT,
    HEATMAP_ROOT,
    OVERLAY_ROOT,
    REPORT_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 6

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_IMAGES = None
# None = process complete test set.
#
# For a quick test you can temporarily use:
#
# MAX_IMAGES = 20


# ============================================================
# 3. TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 4. MODEL
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.7")
print("GRAD-CAM EXPLAINABILITY")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Test dataset:")
print(DATA_ROOT)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Output:")
print(OUTPUT_ROOT)

print()
print("Device:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. CHECK FILES
# ============================================================

if not DATA_ROOT.exists():

    raise FileNotFoundError(
        f"\nTest dataset not found:\n{DATA_ROOT}"
    )

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"\nEfficientNet checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# 6. LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

model = create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)

model.eval()

print("✓ Model loaded")


# ============================================================
# 7. FIND TARGET LAYER
# ============================================================
#
# We hook the LAST convolutional feature layer.
#
# EfficientNet-B0:
#
# backbone.features
#       |
#       +-- final feature block
#              |
#              +-- Conv2d
#
# Grad-CAM requires a spatial feature map.
#
# ============================================================

target_layer = None
target_layer_name = None

for name, module in reversed(
    list(model.backbone.features.named_modules())
):

    if isinstance(
        module,
        torch.nn.Conv2d
    ):

        target_layer = module

        target_layer_name = (
            f"backbone.features.{name}"
        )

        break


if target_layer is None:

    raise RuntimeError(
        "Could not locate a Conv2d target layer "
        "for Grad-CAM."
    )


print()
print("Grad-CAM target layer:")
print(target_layer_name)


# ============================================================
# 8. GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(
    module,
    inputs,
    output
):

    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# 9. GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image_tensor,
    target_class
):

    global activations
    global gradients

    activations = None
    gradients = None

    model.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(
        image_tensor
    )

    # --------------------------------------------------------
    # Select target class
    # --------------------------------------------------------

    target_score = logits[
        0,
        target_class
    ]

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    target_score.backward()

    if activations is None:

        raise RuntimeError(
            "Grad-CAM activations were not captured."
        )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients were not captured."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = (
        weights
        * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(
        cam
    )

    # --------------------------------------------------------
    # Resize to image size
    # --------------------------------------------------------

    cam = F.interpolate(
        cam,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()

    cam = (
        cam
        - cam.min()
    )

    max_value = cam.max()

    if max_value > 0:

        cam = (
            cam
            / max_value
        )

    return (
        logits.detach(),
        cam.detach().cpu().numpy()
    )


# ============================================================
# 10. IMAGE DISCOVERY
# ============================================================

print()
print("=" * 70)
print("LOCATING TEST IMAGES")
print("=" * 70)

image_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
}

image_records = []

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    class_dir = (
        DATA_ROOT
        / class_name
    )

    if not class_dir.exists():

        print(
            f"⚠ Missing class directory: "
            f"{class_dir}"
        )

        continue

    for image_path in sorted(
        class_dir.iterdir()
    ):

        if image_path.suffix.lower() not in image_extensions:

            continue

        image_records.append({

            "image_path": str(
                image_path
            ),

            "true_label_index":
                class_index,

            "true_class":
                class_name
        })


if MAX_IMAGES is not None:

    image_records = image_records[
        :MAX_IMAGES
    ]


print()
print(
    "Test images found:",
    len(image_records)
)


if len(image_records) == 0:

    raise RuntimeError(
        "No test images were found."
    )


# ============================================================
# 11. HELPER FUNCTIONS
# ============================================================

def load_original_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    return np.asarray(
        image
    )


def save_heatmap(
    cam,
    path,
    title
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.axis(
        "off"
    )

    plt.title(
        title
    )

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


def save_overlay(
    original,
    cam,
    path,
    title
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        original
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.axis(
        "off"
    )

    plt.title(
        title
    )

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


# ============================================================
# 12. GENERATE GRAD-CAM
# ============================================================

print()
print("=" * 70)
print("GENERATING GRAD-CAM")
print("=" * 70)

results = []

start_time = time.time()

for counter, record in enumerate(
    image_records,
    start=1
):

    image_path = Path(
        record["image_path"]
    )

    true_label_index = (
        record["true_label_index"]
    )

    true_class = (
        record["true_class"]
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original = load_original_image(
        image_path
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # First prediction
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_index = int(
        probabilities.argmax(
            dim=1
        ).item()
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[
            0,
            predicted_index
        ].item()
    )

    # --------------------------------------------------------
    # Grad-CAM for predicted class
    # --------------------------------------------------------

    logits, cam = generate_gradcam(
        tensor,
        predicted_index
    )

    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    stem = image_path.stem

    safe_class = (
        predicted_class
        .replace(" ", "_")
    )

    heatmap_filename = (
        f"{stem}_"
        f"{safe_class}_"
        f"heatmap.png"
    )

    overlay_filename = (
        f"{stem}_"
        f"{safe_class}_"
        f"overlay.png"
    )

    heatmap_path = (
        HEATMAP_ROOT
        / heatmap_filename
    )

    overlay_path = (
        OVERLAY_ROOT
        / overlay_filename
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    save_heatmap(
        cam,
        heatmap_path,
        (
            f"{predicted_class} "
            f"({confidence:.3f})"
        )
    )

    save_overlay(
        original,
        cam,
        overlay_path,
        (
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence:.3f}"
        )
    )

    # --------------------------------------------------------
    # Result record
    # --------------------------------------------------------

    results.append({

        "image_path":
            str(image_path),

        "true_label_index":
            true_label_index,

        "true_class":
            true_class,

        "predicted_label_index":
            predicted_index,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "correct_prediction":
            predicted_index == true_label_index,

        "heatmap_path":
            str(heatmap_path),

        "overlay_path":
            str(overlay_path)
    })

    if (
        counter == 1
        or counter % 100 == 0
        or counter == len(image_records)
    ):

        print(
            f"Processed: "
            f"{counter}/{len(image_records)}"
        )


elapsed = (
    time.time()
    - start_time
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_csv = (
    REPORT_ROOT
    / "grad_cam_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# 14. SUMMARY STATISTICS
# ============================================================

total_images = len(
    results_df
)

correct_count = int(
    results_df[
        "correct_prediction"
    ].sum()
)

incorrect_count = (
    total_images
    - correct_count
)

accuracy = (
    correct_count
    / total_images
    if total_images > 0
    else 0.0
)

mean_confidence = float(
    results_df[
        "confidence"
    ].mean()
)


per_class = {}

for class_name in CLASS_NAMES:

    class_df = results_df[
        results_df[
            "true_class"
        ] == class_name
    ]

    if len(class_df) == 0:

        continue

    per_class[class_name] = {

        "samples":
            int(len(class_df)),

        "correct":
            int(
                class_df[
                    "correct_prediction"
                ].sum()
            ),

        "accuracy":
            float(
                class_df[
                    "correct_prediction"
                ].mean()
            ),

        "mean_confidence":
            float(
                class_df[
                    "confidence"
                ].mean()
            )
    }


summary = {

    "stage":
        "STEP 8.7",

    "analysis":
        "Grad-CAM Explainability",

    "model":
        "EfficientNet-B0",

    "checkpoint":
        str(
            CHECKPOINT_PATH
        ),

    "test_images":
        total_images,

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "image_size":
        IMAGE_SIZE,

    "target_layer":
        target_layer_name,

    "device":
        str(DEVICE),

    "accuracy":
        accuracy,

    "correct_predictions":
        correct_count,

    "incorrect_predictions":
        incorrect_count,

    "mean_confidence":
        mean_confidence,

    "processing_time_seconds":
        elapsed,

    "per_class":
        per_class
}


summary_json = (
    OUTPUT_ROOT
    / "grad_cam_summary.json"
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
# 15. TEXT SUMMARY
# ============================================================

summary_txt = (
    OUTPUT_ROOT
    / "grad_cam_summary.txt"
)

with open(
    summary_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPIRA — STEP 8.7\n"
    )

    f.write(
        "GRAD-CAM EXPLAINABILITY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Model: EfficientNet-B0\n"
    )

    f.write(
        f"Test images: {total_images}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Mean confidence: "
        f"{mean_confidence:.4f}\n"
    )

    f.write(
        f"Target layer: "
        f"{target_layer_name}\n"
    )

    f.write(
        f"Processing time: "
        f"{elapsed:.2f} seconds\n\n"
    )

    f.write(
        "Per-class statistics\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    for class_name, stats in per_class.items():

        f.write(
            f"\n{class_name}\n"
        )

        f.write(
            f"  Samples: "
            f"{stats['samples']}\n"
        )

        f.write(
            f"  Correct: "
            f"{stats['correct']}\n"
        )

        f.write(
            f"  Accuracy: "
            f"{stats['accuracy']:.4f}\n"
        )

        f.write(
            f"  Mean confidence: "
            f"{stats['mean_confidence']:.4f}\n"
        )


# ============================================================
# 16. REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 8.7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()

print(
    "Grad-CAM outputs:"
)

print(
    OUTPUT_ROOT
)

print()

print(
    "Generated:"
)

print(
    "  ✓ Heatmaps:"
)

print(
    f"    {HEATMAP_ROOT}"
)

print(
    "  ✓ Overlays:"
)

print(
    f"    {OVERLAY_ROOT}"
)

print(
    "  ✓ Results:"
)

print(
    f"    {results_csv}"
)

print(
    "  ✓ Summary:"
)

print(
    f"    {summary_json}"
)

print(
    "  ✓ Text report:"
)

print(
    f"    {summary_txt}"
)

print()

print(
    f"Test images      : {total_images}"
)

print(
    f"Correct          : {correct_count}"
)

print(
    f"Incorrect        : {incorrect_count}"
)

print(
    f"Accuracy         : {accuracy:.4f}"
)

print(
    f"Mean confidence  : "
    f"{mean_confidence:.4f}"
)

print()

print(
    "✓ Grad-CAM heatmaps generated."
)

print(
    "✓ Grad-CAM overlays generated."
)

print(
    "✓ Explainability results saved."
)

print()

print(
    "Next stage:"
)

print(
    "STEP 8.8 — ViT ATTENTION VISUALIZATION"
)