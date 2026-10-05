# ============================================================
# RESPIRA - EfficientNet + ViT Probability Fusion
# Validation Weight Selection
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from src.models.efficientnet import create_efficientnet_b0
from src.models.vit import VisionTransformerModel


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "Lung_Disease_Preprocessed_512"
)

VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

EFFICIENTNET_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
)

VIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
)

VALIDATION_DIR = OUTPUT_DIR / "validation"

for directory in [
    OUTPUT_DIR,
    VALIDATION_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


EFFICIENTNET_CHECKPOINT = (
    EFFICIENTNET_DIR
    / "checkpoints"
    / "best_model.pth"
)

VIT_CHECKPOINT = (
    VIT_DIR
    / "checkpoints"
    / "best_model.pth"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_CLASSES = 6

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# Search fusion weights in 0.01 increments.
ALPHA_VALUES = np.arange(
    0.00,
    1.01,
    0.01
)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESPIRA - EfficientNet + ViT PROBABILITY FUSION")
print("=" * 70)

print(f"Device : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Validation data : {VAL_DIR}")

print(
    f"EfficientNet checkpoint : "
    f"{EFFICIENTNET_CHECKPOINT}"
)

print(
    f"ViT checkpoint           : "
    f"{VIT_CHECKPOINT}"
)

print("=" * 70)


# ============================================================
# 4. VERIFY PATHS
# ============================================================

for path in [
    VAL_DIR,
    EFFICIENTNET_CHECKPOINT,
    VIT_CHECKPOINT,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# 5. VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 6. LOAD VALIDATION DATASET
# ============================================================

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nValidation dataset:")
print(
    f"Images: {len(val_dataset)}"
)

print(
    f"Classes: {val_dataset.classes}"
)


if val_dataset.classes != CLASS_NAMES:

    raise ValueError(
        "\nClass order mismatch!\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found: {val_dataset.classes}"
    )

print("✓ Class order verified")


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 7. LOAD EFFICIENTNET
# ============================================================

print("\nLoading EfficientNet-B0...")

efficientnet_checkpoint = torch.load(
    EFFICIENTNET_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

efficientnet = create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=0.3,
)

efficientnet.load_state_dict(
    efficientnet_checkpoint[
        "model_state_dict"
    ]
)

efficientnet = efficientnet.to(DEVICE)

efficientnet.eval()

print(
    f"✓ EfficientNet checkpoint "
    f"epoch: "
    f"{efficientnet_checkpoint.get('epoch', 'N/A')}"
)

print(
    f"✓ EfficientNet validation accuracy: "
    f"{efficientnet_checkpoint.get('val_accuracy', 0) * 100:.2f}%"
)


# ============================================================
# 8. LOAD ViT
# ============================================================

print("\nLoading ViT-B/16...")

vit_checkpoint = torch.load(
    VIT_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

vit = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=False,
)

vit.load_state_dict(
    vit_checkpoint[
        "model_state_dict"
    ]
)

vit = vit.to(DEVICE)

vit.eval()

print(
    f"✓ ViT checkpoint epoch: "
    f"{vit_checkpoint.get('epoch', 'N/A')}"
)

print(
    f"✓ ViT validation accuracy: "
    f"{vit_checkpoint.get('val_accuracy', 0) * 100:.2f}%"
)


# ============================================================
# 9. GENERATE VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

all_labels = []

efficientnet_probabilities = []
vit_probabilities = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        val_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # EfficientNet
        # ----------------------------------------------------

        efficientnet_output = (
            efficientnet(images)
        )

        if isinstance(
            efficientnet_output,
            dict
        ):

            efficientnet_logits = (
                efficientnet_output["logits"]
            )

        else:

            efficientnet_logits = (
                efficientnet_output
            )

        efficientnet_probs = F.softmax(
            efficientnet_logits,
            dim=1
        )


        # ----------------------------------------------------
        # ViT
        # ----------------------------------------------------

        vit_output = vit(images)

        if isinstance(
            vit_output,
            dict
        ):

            vit_logits = vit_output["logits"]

        else:

            vit_logits = vit_output

        vit_probs = F.softmax(
            vit_logits,
            dim=1
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        efficientnet_probabilities.extend(
            efficientnet_probs.cpu().numpy()
        )

        vit_probabilities.extend(
            vit_probs.cpu().numpy()
        )


        if batch_index % 25 == 0:

            processed = min(
                batch_index * BATCH_SIZE,
                len(val_dataset)
            )

            print(
                f"  Processed "
                f"{processed}/"
                f"{len(val_dataset)}"
            )


# ============================================================
# 10. CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    all_labels
)

effnet_probs = np.array(
    efficientnet_probabilities
)

vit_probs = np.array(
    vit_probabilities
)


print("\n✓ Validation probabilities generated")

print(
    f"Labels shape       : {y_true.shape}"
)

print(
    f"EfficientNet shape : {effnet_probs.shape}"
)

print(
    f"ViT shape          : {vit_probs.shape}"
)


# ============================================================
# 11. VERIFY ALIGNMENT
# ============================================================

if not (
    len(y_true)
    == len(effnet_probs)
    == len(vit_probs)
):

    raise RuntimeError(
        "Validation prediction lengths do not match."
    )


# ============================================================
# 12. SAVE RAW VALIDATION PROBABILITIES
# ============================================================

validation_data = {

    "true_label": [
        CLASS_NAMES[i]
        for i in y_true
    ],
}


for i, class_name in enumerate(
    CLASS_NAMES
):

    validation_data[
        f"effnet_prob_{class_name}"
    ] = effnet_probs[:, i]


for i, class_name in enumerate(
    CLASS_NAMES
):

    validation_data[
        f"vit_prob_{class_name}"
    ] = vit_probs[:, i]


validation_predictions_df = pd.DataFrame(
    validation_data
)

validation_predictions_df.to_csv(
    VALIDATION_DIR
    / "validation_probabilities.csv",
    index=False
)


# ============================================================
# 13. INDIVIDUAL VALIDATION ACCURACIES
# ============================================================

effnet_predictions = np.argmax(
    effnet_probs,
    axis=1
)

vit_predictions = np.argmax(
    vit_probs,
    axis=1
)

effnet_val_accuracy = accuracy_score(
    y_true,
    effnet_predictions
)

vit_val_accuracy = accuracy_score(
    y_true,
    vit_predictions
)

print("\nValidation baseline:")

print(
    f"EfficientNet-B0 : "
    f"{effnet_val_accuracy * 100:.2f}%"
)

print(
    f"ViT-B/16        : "
    f"{vit_val_accuracy * 100:.2f}%"
)


# ============================================================
# 14. SEARCH FUSION WEIGHT
# ============================================================

print("\n")
print("=" * 70)
print("SEARCHING OPTIMAL FUSION WEIGHT")
print("=" * 70)

results = []

best_alpha = None
best_accuracy = -1.0

best_macro_f1 = -1.0


for alpha in ALPHA_VALUES:

    # --------------------------------------------------------
    # Weighted probability fusion
    #
    # alpha = EfficientNet weight
    # 1-alpha = ViT weight
    # --------------------------------------------------------

    fusion_probs = (
        alpha * effnet_probs
        +
        (1.0 - alpha) * vit_probs
    )

    fusion_predictions = np.argmax(
        fusion_probs,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        fusion_predictions
    )

    (
        precision,
        recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        fusion_predictions,
        average="macro",
        zero_division=0
    )

    results.append({

        "alpha_efficientnet":
            float(alpha),

        "weight_vit":
            float(1.0 - alpha),

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(macro_f1),
    })


    # Primary selection criterion:
    # validation accuracy.
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_alpha = float(alpha)

        best_macro_f1 = float(
            macro_f1
        )


# ============================================================
# 15. SAVE GRID SEARCH
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    VALIDATION_DIR
    / "fusion_weight_search.csv",
    index=False
)


# ============================================================
# 16. BEST RESULT
# ============================================================

best_row = results_df[
    results_df["accuracy"]
    == results_df["accuracy"].max()
].iloc[0]

best_alpha = float(
    best_row["alpha_efficientnet"]
)

best_vit_weight = float(
    best_row["weight_vit"]
)

best_accuracy = float(
    best_row["accuracy"]
)

best_macro_f1 = float(
    best_row["macro_f1"]
)


print("\nBEST VALIDATION FUSION:")
print(
    f"EfficientNet weight : "
    f"{best_alpha:.2f}"
)

print(
    f"ViT weight          : "
    f"{best_vit_weight:.2f}"
)

print(
    f"Validation Accuracy : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{best_macro_f1 * 100:.2f}%"
)


# ============================================================
# 17. COMPARE WITH INDIVIDUAL MODELS
# ============================================================

print("\nValidation comparison:")

print(
    f"EfficientNet-B0 : "
    f"{effnet_val_accuracy * 100:.2f}%"
)

print(
    f"ViT-B/16        : "
    f"{vit_val_accuracy * 100:.2f}%"
)

print(
    f"Fusion          : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Fusion gain vs EfficientNet: "
    f"{(best_accuracy - effnet_val_accuracy) * 100:+.2f} percentage points"
)

print(
    f"Fusion gain vs ViT: "
    f"{(best_accuracy - vit_val_accuracy) * 100:+.2f} percentage points"
)


# ============================================================
# 18. PLOT WEIGHT SEARCH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    results_df["alpha_efficientnet"],
    results_df["accuracy"] * 100,
    marker="o",
    markersize=3,
    label="Fusion Validation Accuracy"
)

plt.axvline(
    best_alpha,
    linestyle="--",
    label=(
        f"Best α = {best_alpha:.2f}"
    )
)

plt.xlabel(
    "EfficientNet Weight (α)"
)

plt.ylabel(
    "Validation Accuracy (%)"
)

plt.title(
    "EfficientNet + ViT Fusion Weight Search"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    VALIDATION_DIR
    / "fusion_weight_search.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. SAVE SELECTED FUSION CONFIGURATION
# ============================================================

fusion_config = {

    "fusion_method":
        "weighted_probability_fusion",

    "primary_selection_metric":
        "validation_accuracy",

    "efficientnet_weight":
        best_alpha,

    "vit_weight":
        best_vit_weight,

    "validation_accuracy":
        best_accuracy,

    "validation_macro_f1":
        best_macro_f1,

    "efficientnet_validation_accuracy":
        float(effnet_val_accuracy),

    "vit_validation_accuracy":
        float(vit_val_accuracy),

    "efficientnet_checkpoint":
        str(EFFICIENTNET_CHECKPOINT),

    "vit_checkpoint":
        str(VIT_CHECKPOINT),

    "image_size":
        "512x512",

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,
}


with open(
    OUTPUT_DIR
    / "fusion_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fusion_config,
        f,
        indent=4
    )


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION FUSION SEARCH COMPLETE")
print("=" * 70)

print(
    f"Selected EfficientNet weight : "
    f"{best_alpha:.2f}"
)

print(
    f"Selected ViT weight          : "
    f"{best_vit_weight:.2f}"
)

print(
    f"Best validation accuracy     : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    "\nSaved:"
)

print(
    VALIDATION_DIR
    / "validation_probabilities.csv"
)

print(
    VALIDATION_DIR
    / "fusion_weight_search.csv"
)

print(
    VALIDATION_DIR
    / "fusion_weight_search.png"
)

print(
    OUTPUT_DIR
    / "fusion_config.json"
)

print("\nIMPORTANT:")
print(
    "The test set has NOT been used to select "
    "the fusion weight."
)

print(
    "The selected weight is now frozen for "
    "the final test fusion."
)

print("=" * 70)