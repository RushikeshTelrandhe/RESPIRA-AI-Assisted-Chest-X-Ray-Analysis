# ============================================================
# Respira — STEP 8.8
# ViT-B/16 Attention Rollout Explainability
# ============================================================
#
# Model:
#   ViT-B/16 — 512x512
#
# Input:
#   data/Lung_Disease_Preprocessed_512/test/
#
# Checkpoint:
#   outputs/vit_512/checkpoints/best_model.pth
#
# Output:
#   outputs/vit_512/explainability/attention/
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

from src.models.vit import VisionTransformerModel


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
    / "vit_512"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
    / "explainability"
    / "attention"
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

IMAGE_SIZE = 512

PATCH_SIZE = 16

PATCH_GRID = IMAGE_SIZE // PATCH_SIZE

NUM_PATCHES = PATCH_GRID * PATCH_GRID

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


# ------------------------------------------------------------
# IMPORTANT:
# First run only 30 images.
#
# After successful verification:
#
MAX_IMAGES = None
# ------------------------------------------------------------

#MAX_IMAGES = 30


# ============================================================
# 3. TRANSFORM
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
# 4. HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.8")
print("ViT-B/16 ATTENTION ROLLOUT")
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
print("Image size:")
print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

print()
print("Patch size:")
print(f"{PATCH_SIZE} x {PATCH_SIZE}")

print()
print("Patch grid:")
print(f"{PATCH_GRID} x {PATCH_GRID}")

print()
print("Patch tokens:")
print(NUM_PATCHES)

print()
print("Total transformer tokens:")
print(NUM_PATCHES + 1)

print()
print("Device:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. CHECK PATHS
# ============================================================

if not DATA_ROOT.exists():

    raise FileNotFoundError(
        f"\nTest dataset not found:\n{DATA_ROOT}"
    )

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"\nViT checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# 6. LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING ViT-B/16 512x512")
print("=" * 70)

model = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=False
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

print("✓ ViT-B/16 loaded")
print("✓ 512x512 configuration verified")


# ============================================================
# 7. ATTENTION STORAGE
# ============================================================

attention_maps = []


# ============================================================
# 8. FIND TRANSFORMER ATTENTION MODULES
# ============================================================
#
# torchvision ViT uses:
#
# backbone.encoder.layers
#       ↓
# EncoderBlock
#       ↓
# self_attention
#
# The original EncoderBlock requests:
#
#     need_weights=False
#
# We temporarily override this through a pre-hook so that
# MultiheadAttention returns attention matrices.
#
# ============================================================

attention_modules = []

if hasattr(model.backbone, "encoder"):

    encoder_layers = (
        model.backbone
        .encoder
        .layers
    )

    for layer_index, layer in enumerate(
        encoder_layers
    ):

        if hasattr(
            layer,
            "self_attention"
        ):

            attention_modules.append(
                (
                    layer_index,
                    layer.self_attention
                )
            )


if len(attention_modules) == 0:

    raise RuntimeError(
        "Could not locate ViT self-attention modules."
    )


print()
print(
    "Transformer attention layers found:",
    len(attention_modules)
)


# ============================================================
# 9. FORCE ATTENTION WEIGHTS
# ============================================================

def attention_pre_hook(
    module,
    args,
    kwargs
):

    kwargs = dict(kwargs)

    kwargs["need_weights"] = True

    # Keep individual attention heads.
    kwargs["average_attn_weights"] = False

    return args, kwargs


def attention_forward_hook(
    layer_index
):

    def hook(
        module,
        inputs,
        output
    ):

        # MultiheadAttention output:
        #
        # output[0] = attention output
        # output[1] = attention weights
        #
        if not isinstance(
            output,
            tuple
        ):

            return

        if len(output) < 2:

            return

        weights = output[1]

        if weights is None:

            return

        # Expected:
        #
        # [batch, heads, tokens, tokens]
        #
        if weights.dim() == 4:

            attention_maps.append(
                (
                    layer_index,
                    weights.detach()
                )
            )

    return hook


pre_handles = []
forward_handles = []

for layer_index, attention_module in attention_modules:

    pre_handle = (
        attention_module.register_forward_pre_hook(
            attention_pre_hook,
            with_kwargs=True
        )
    )

    forward_handle = (
        attention_module.register_forward_hook(
            attention_forward_hook(
                layer_index
            )
        )
    )

    pre_handles.append(pre_handle)
    forward_handles.append(forward_handle)


# ============================================================
# 10. ATTENTION ROLLOUT
# ============================================================

def attention_rollout(
    attention_list
):

    if not attention_list:

        raise RuntimeError(
            "No attention maps were captured."
        )

    # Sort by transformer layer
    attention_list = sorted(
        attention_list,
        key=lambda x: x[0]
    )

    rollout = None

    for layer_index, attention in attention_list:

        # ----------------------------------------------------
        # attention:
        #
        # [B, H, N, N]
        # ----------------------------------------------------

        attention = attention.float()

        # Average heads
        attention = attention.mean(
            dim=1
        )

        # ----------------------------------------------------
        # Add residual connection
        # ----------------------------------------------------

        batch_size = attention.shape[0]

        token_count = attention.shape[-1]

        identity = torch.eye(
            token_count,
            device=attention.device,
            dtype=attention.dtype
        ).unsqueeze(0)

        identity = identity.expand(
            batch_size,
            -1,
            -1
        )

        attention = (
            attention
            + identity
        )

        # ----------------------------------------------------
        # Normalize rows
        # ----------------------------------------------------

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )

        # ----------------------------------------------------
        # Multiply attention matrices
        # ----------------------------------------------------

        if rollout is None:

            rollout = attention

        else:

            rollout = torch.bmm(
                attention,
                rollout
            )

    # --------------------------------------------------------
    # CLS token → image patches
    # --------------------------------------------------------

    cls_attention = rollout[
        :,
        0,
        1:
    ]

    # Expected:
    #
    # [B, 1024]
    #

    expected_patches = NUM_PATCHES

    if cls_attention.shape[1] != expected_patches:

        raise RuntimeError(
            "Unexpected number of patch tokens. "
            f"Expected {expected_patches}, "
            f"got {cls_attention.shape[1]}."
        )

    # --------------------------------------------------------
    # Reshape to 32 × 32
    # --------------------------------------------------------

    attention_map = cls_attention.reshape(
        batch_size,
        PATCH_GRID,
        PATCH_GRID
    )

    # --------------------------------------------------------
    # Normalize each image
    # --------------------------------------------------------

    min_value = (
        attention_map
        .flatten(1)
        .min(dim=1)[0]
        .view(batch_size, 1, 1)
    )

    max_value = (
        attention_map
        .flatten(1)
        .max(dim=1)[0]
        .view(batch_size, 1, 1)
    )

    attention_map = (
        attention_map
        - min_value
    ) / (
        max_value
        - min_value
        + 1e-8
    )

    # --------------------------------------------------------
    # Resize 32×32 → 512×512
    # --------------------------------------------------------

    attention_map = F.interpolate(
        attention_map.unsqueeze(1),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )

    attention_map = (
        attention_map
        .squeeze(1)
    )

    return attention_map


# ============================================================
# 11. IMAGE DISCOVERY
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

            "image_path":
                str(image_path),

            "true_label_index":
                class_index,

            "true_class":
                class_name
        })


if MAX_IMAGES is not None:

    image_records = (
        image_records[:MAX_IMAGES]
    )


print()
print(
    "Images selected:",
    len(image_records)
)

if len(image_records) == 0:

    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# 12. IMAGE HELPERS
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


def save_attention_heatmap(
    attention,
    path,
    title
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        attention,
        cmap="jet"
    )

    plt.axis("off")

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


def save_attention_overlay(
    original,
    attention,
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
        attention,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

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
# 13. GENERATE ATTENTION MAPS
# ============================================================

print()
print("=" * 70)
print("GENERATING ViT ATTENTION ROLLOUT")
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
    # Clear previous attention
    # --------------------------------------------------------

    attention_maps.clear()

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
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

        # Your custom ViT returns:
        #
        # tokens, pooled, logits
        #
        if isinstance(
            output,
            tuple
        ):

            logits = output[-1]

        elif isinstance(
            output,
            dict
        ):

            logits = output["logits"]

        else:

            logits = output

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
    # Attention rollout
    # --------------------------------------------------------

    if len(attention_maps) == 0:

        raise RuntimeError(
            "No attention maps captured for "
            f"{image_path.name}"
        )

    attention = attention_rollout(
        attention_maps
    )

    attention = (
        attention[0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Filenames
    # --------------------------------------------------------

    stem = image_path.stem

    safe_class = (
        predicted_class
        .replace(" ", "_")
    )

    heatmap_path = (
        HEATMAP_ROOT
        / f"{stem}_{safe_class}_attention.png"
    )

    overlay_path = (
        OVERLAY_ROOT
        / f"{stem}_{safe_class}_overlay.png"
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    save_attention_heatmap(
        attention,
        heatmap_path,
        (
            f"{predicted_class} | "
            f"{confidence:.3f}"
        )
    )

    save_attention_overlay(
        original,
        attention,
        overlay_path,
        (
            f"True: {true_class} | "
            f"Pred: {predicted_class} | "
            f"{confidence:.3f}"
        )
    )

    # --------------------------------------------------------
    # Save record
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
            (
                predicted_index
                == true_label_index
            ),

        "attention_layers":
            len(attention_maps),

        "patch_grid":
            f"{PATCH_GRID}x{PATCH_GRID}",

        "patch_tokens":
            NUM_PATCHES,

        "heatmap_path":
            str(heatmap_path),

        "overlay_path":
            str(overlay_path)
    })

    print(
        f"[{counter:03d}/{len(image_records):03d}] "
        f"True={true_class} | "
        f"Pred={predicted_class} | "
        f"Conf={confidence:.3f} | "
        f"Layers={len(attention_maps)}"
    )


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_csv = (
    REPORT_ROOT
    / "vit_attention_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# 15. SUMMARY
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

elapsed = (
    time.time()
    - start_time
)


# ============================================================
# 16. PER-CLASS SUMMARY
# ============================================================

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


# ============================================================
# 17. SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "STEP 8.8",

    "analysis":
        "ViT Attention Rollout",

    "model":
        "ViT-B/16",

    "image_size":
        "512x512",

    "patch_size":
        "16x16",

    "patch_grid":
        "32x32",

    "patch_tokens":
        1024,

    "total_tokens":
        1025,

    "checkpoint":
        str(CHECKPOINT_PATH),

    "transformer_layers":
        len(attention_modules),

    "test_images":
        total_images,

    "correct_predictions":
        correct_count,

    "incorrect_predictions":
        incorrect_count,

    "accuracy":
        accuracy,

    "mean_confidence":
        mean_confidence,

    "processing_time_seconds":
        elapsed,

    "classes":
        CLASS_NAMES,

    "per_class":
        per_class
}


summary_json = (
    OUTPUT_ROOT
    / "vit_attention_summary.json"
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
# 18. REMOVE HOOKS
# ============================================================

for handle in pre_handles:
    handle.remove()

for handle in forward_handles:
    handle.remove()


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 8.8 COMPLETED")
print("=" * 70)

print()

print(
    f"Images processed : {total_images}"
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

print(
    f"Attention layers : "
    f"{len(attention_modules)}"
)

print(
    f"Patch tokens     : "
    f"{NUM_PATCHES}"
)

print()

print("Outputs:")

print(
    f"Heatmaps : {HEATMAP_ROOT}"
)

print(
    f"Overlays : {OVERLAY_ROOT}"
)

print(
    f"Results  : {results_csv}"
)

print(
    f"Summary  : {summary_json}"
)

print()

print(
    "✓ 512x512 ViT attention maps generated."
)

print(
    "✓ Attention rollout generated."
)

print(
    "✓ ViT explainability results saved."
)

print()

print(
    "Next stage:"
)

print(
    "STEP 8.9 — XAI COMPARISON"
)