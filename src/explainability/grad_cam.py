# ============================================================
# Respira — STEP 8.7
# Grad-CAM Explainability for EfficientNet-B0 512x512
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
    / "Lung_Disease_Preprocessed_512"
    / "test"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
    / "explainability"
    / "grad_cam"
)

HEATMAP_ROOT = OUTPUT_ROOT / "heatmaps"
OVERLAY_ROOT = OUTPUT_ROOT / "overlays"
REPORT_ROOT = OUTPUT_ROOT / "reports"

for directory in [
    OUTPUT_ROOT,
    HEATMAP_ROOT,
    OVERLAY_ROOT,
    REPORT_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 512
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

# IMPORTANT:
# Start with a small number to verify everything.
#
# After successful verification:
# MAX_IMAGES = None
#
MAX_IMAGES = None


# ============================================================
# 3. TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. START
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.7")
print("EFFICIENTNET-B0 512x512 GRAD-CAM")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset      : {DATA_ROOT}")
print(f"Checkpoint   : {CHECKPOINT_PATH}")
print(f"Output       : {OUTPUT_ROOT}")
print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# 5. CHECK PATHS
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{DATA_ROOT}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# 6. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
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
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("✓ EfficientNet-B0 loaded")
print("✓ 512x512 input configured")


# ============================================================
# 7. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

target_layer = None
target_layer_name = None

for name, module in reversed(
    list(model.backbone.features.named_modules())
):
    if isinstance(module, torch.nn.Conv2d):
        target_layer = module
        target_layer_name = (
            f"backbone.features.{name}"
        )
        break

if target_layer is None:
    raise RuntimeError(
        "Could not locate Conv2d target layer."
    )

print(
    f"\nGrad-CAM target layer: "
    f"{target_layer_name}"
)


# ============================================================
# 8. ACTIVATION / GRADIENT STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(module, inputs, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# 9. GRAD-CAM
# ============================================================

def generate_gradcam(
    image_tensor,
    target_class
):

    global activations
    global gradients

    activations = None
    gradients = None

    model.zero_grad(set_to_none=True)

    logits = model(image_tensor)

    target_score = logits[
        0,
        target_class
    ]

    target_score.backward()

    if activations is None:
        raise RuntimeError(
            "Grad-CAM activations not captured."
        )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients not captured."
        )

    # Global average pooling
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted activation maps
    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    # ReLU
    cam = F.relu(cam)

    # Resize to original 512x512 representation
    cam = F.interpolate(
        cam,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()

    # Normalize 0–1
    cam = cam - cam.min()

    max_value = cam.max()

    if max_value > 0:
        cam = cam / max_value

    return (
        logits.detach(),
        cam.detach().cpu().numpy()
    )


# ============================================================
# 10. FIND TEST IMAGES
# ============================================================

print("\n" + "=" * 70)
print("LOCATING TEST IMAGES")
print("=" * 70)

image_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
}

image_records = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = DATA_ROOT / class_name

    if not class_dir.exists():
        print(
            f"⚠ Missing: {class_dir}"
        )
        continue

    for image_path in sorted(
        class_dir.iterdir()
    ):

        if image_path.suffix.lower() not in image_extensions:
            continue

        image_records.append({
            "image_path": str(image_path),
            "true_label_index": class_index,
            "true_class": class_name
        })


if MAX_IMAGES is not None:
    image_records = image_records[:MAX_IMAGES]

print(
    f"\nImages selected: "
    f"{len(image_records)}"
)

if not image_records:
    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# 11. IMAGE HELPERS
# ============================================================

def load_original_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return np.asarray(image)


def save_heatmap(
    cam,
    path,
    title
):

    plt.figure(figsize=(6, 6))

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)

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

    plt.figure(figsize=(6, 6))

    plt.imshow(original)

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)

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

print("\n" + "=" * 70)
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

    # Load image
    original = load_original_image(
        image_path
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Prediction
    model.zero_grad(set_to_none=True)

    logits = model(tensor)

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
        CLASS_NAMES[predicted_index]
    )

    confidence = float(
        probabilities[
            0,
            predicted_index
        ].item()
    )

    # Grad-CAM
    _, cam = generate_gradcam(
        tensor,
        predicted_index
    )

    # Filenames
    stem = image_path.stem

    safe_class = (
        predicted_class
        .replace(" ", "_")
    )

    heatmap_path = (
        HEATMAP_ROOT
        / f"{stem}_{safe_class}_heatmap.png"
    )

    overlay_path = (
        OVERLAY_ROOT
        / f"{stem}_{safe_class}_overlay.png"
    )

    # Save
    save_heatmap(
        cam,
        heatmap_path,
        f"{predicted_class} | "
        f"{confidence:.3f}"
    )

    save_overlay(
        original,
        cam,
        overlay_path,
        f"True: {true_class} | "
        f"Pred: {predicted_class} | "
        f"{confidence:.3f}"
    )

    results.append({
        "image_path": str(image_path),
        "true_class": true_class,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "correct_prediction":
            predicted_index == true_label_index,
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path)
    })

    print(
        f"[{counter:03d}/{len(image_records):03d}] "
        f"True={true_class} | "
        f"Pred={predicted_class} | "
        f"Conf={confidence:.3f}"
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_csv = (
    REPORT_ROOT
    / "grad_cam_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

total = len(results_df)

correct = int(
    results_df[
        "correct_prediction"
    ].sum()
)

accuracy = (
    correct / total
    if total else 0
)

mean_confidence = float(
    results_df["confidence"].mean()
)

elapsed = time.time() - start_time


# ============================================================
# 14. PER-CLASS SUMMARY
# ============================================================

per_class = {}

for class_name in CLASS_NAMES:

    class_df = results_df[
        results_df["true_class"] == class_name
    ]

    if len(class_df) == 0:
        continue

    per_class[class_name] = {
        "samples": int(len(class_df)),
        "correct": int(
            class_df[
                "correct_prediction"
            ].sum()
        ),
        "accuracy": float(
            class_df[
                "correct_prediction"
            ].mean()
        ),
        "mean_confidence": float(
            class_df["confidence"].mean()
        )
    }


# ============================================================
# 15. SUMMARY JSON
# ============================================================

summary = {
    "stage": "STEP 8.7",
    "analysis": "Grad-CAM Explainability",
    "model": "EfficientNet-B0",
    "image_size": "512x512",
    "checkpoint": str(CHECKPOINT_PATH),
    "target_layer": target_layer_name,
    "test_images": total,
    "correct_predictions": correct,
    "accuracy": accuracy,
    "mean_confidence": mean_confidence,
    "processing_time_seconds": elapsed,
    "classes": CLASS_NAMES,
    "per_class": per_class
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
# 16. REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# 17. FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 8.7 COMPLETED")
print("=" * 70)

print(f"\nImages processed : {total}")
print(f"Correct          : {correct}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Mean confidence  : {mean_confidence:.4f}")

print("\nOutputs:")
print(f"Heatmaps : {HEATMAP_ROOT}")
print(f"Overlays : {OVERLAY_ROOT}")
print(f"Results  : {results_csv}")
print(f"Summary  : {summary_json}")

print("\n✓ 512x512 Grad-CAM generated")
print("✓ EfficientNet-B0 explanations saved")
print("\nNext stage: STEP 8.8 — ViT Attention Visualization")