# ============================================================
# RESPIRA — STEP 6.3
# VISION TRANSFORMER TEST EVALUATION
# ============================================================
#
# Purpose:
#   Evaluate the best trained ViT-B/16 checkpoint on the
#   untouched test dataset.
#
# Input:
#   data/processed/test/
#   outputs/vit/checkpoints/best_model.pth
#
# Output:
#   outputs/vit/evaluation/
#       ├── test_predictions.csv
#       ├── classification_report.csv
#       ├── confusion_matrix.csv
#       ├── metrics.json
#       ├── confusion_matrix.png
#       └── roc_curves.png
#
# Run from project root:
#   python -m src.evaluation.evaluate_vit
#
# ============================================================

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# IMPORT MODEL
# ------------------------------------------------------------

from src.models.vit import VisionTransformerModel

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 6

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESPIRA - VISION TRANSFORMER TEST EVALUATION")
print("=" * 70)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

test_dir = DATA_DIR / "test"

if not test_dir.exists():
    raise FileNotFoundError(
        f"Test dataset does not exist:\n{test_dir}"
    )

# ------------------------------------------------------------
# TRANSFORM
# ------------------------------------------------------------

test_transform = transforms.Compose([
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
    ),
])

# ------------------------------------------------------------
# LOAD TEST DATASET
# ------------------------------------------------------------

test_dataset = datasets.ImageFolder(
    root=str(test_dir),
    transform=test_transform
)

print()
print("Test dataset:")
print("  Images:", len(test_dataset))

print()
print("Detected classes:")
print(" ", test_dataset.classes)

# ------------------------------------------------------------
# VERIFY CLASS ORDER
# ------------------------------------------------------------

if test_dataset.classes != CLASS_NAMES:

    raise RuntimeError(
        "\nClass order mismatch!\n\n"
        f"Expected:\n{CLASS_NAMES}\n\n"
        f"Detected:\n{test_dataset.classes}\n"
    )

print()
print("✓ Class order verified.")

# ------------------------------------------------------------
# DATALOADER
# ------------------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print()
print("Loading ViT checkpoint...")

model = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=False,
    #image_size=IMAGE_SIZE,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

# ------------------------------------------------------------
# LOAD STATE DICT
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict' or 'state_dict'."
        )

else:

    state_dict = checkpoint

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()

print("✓ Best ViT checkpoint loaded.")

# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

print()
print("Running test inference...")

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        # ----------------------------------------------------
        # HANDLE MODEL OUTPUT
        # ----------------------------------------------------
        #
        # Some custom ViT implementations return:
        #
        #   tensor
        #
        # while others return:
        #
        #   {"logits": tensor, ...}
        #
        # ----------------------------------------------------

        if isinstance(outputs, dict):

            if "logits" in outputs:

                logits = outputs["logits"]

            else:

                raise RuntimeError(
                    "Model returned a dictionary "
                    "without a 'logits' key."
                )

        else:

            logits = outputs

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

# ------------------------------------------------------------
# CONVERT TO NUMPY
# ------------------------------------------------------------

y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)

print()
print(
    f"Inference completed: {len(y_true)} images"
)

# ------------------------------------------------------------
# BASIC METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

# ------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------

roc_auc_per_class = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    binary_true = (
        y_true == class_index
    ).astype(int)

    try:

        auc = roc_auc_score(
            binary_true,
            y_prob[:, class_index]
        )

        roc_auc_per_class[
            class_name
        ] = float(auc)

    except ValueError:

        roc_auc_per_class[
            class_name
        ] = None

valid_auc = [
    value
    for value in roc_auc_per_class.values()
    if value is not None
]

roc_auc_macro = (
    float(np.mean(valid_auc))
    if valid_auc
    else None
)

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

prediction_rows = []

for i in range(
    len(y_true)
):

    row = {
        "index": i,
        "true_label": CLASS_NAMES[
            y_true[i]
        ],
        "predicted_label": CLASS_NAMES[
            y_pred[i]
        ],
        "correct": bool(
            y_true[i] == y_pred[i]
        ),
    }

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        row[
            f"prob_{class_name}"
        ] = float(
            y_prob[i, class_index]
        )

    prediction_rows.append(row)

predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(
    OUTPUT_DIR
    / "test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# METRICS JSON
# ------------------------------------------------------------

metrics = {
    "model": "ViT-B/16",
    "checkpoint": str(
        CHECKPOINT_PATH
    ),
    "test_images": int(
        len(y_true)
    ),
    "num_classes": NUM_CLASSES,
    "classes": CLASS_NAMES,
    "accuracy": float(
        accuracy
    ),
    "precision_macro": float(
        precision_macro
    ),
    "recall_macro": float(
        recall_macro
    ),
    "f1_macro": float(
        f1_macro
    ),
    "precision_weighted": float(
        precision_weighted
    ),
    "recall_weighted": float(
        recall_weighted
    ),
    "f1_weighted": float(
        f1_weighted
    ),
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_per_class": roc_auc_per_class,
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ViT-B/16 Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

threshold = (
    cm.max() / 2
    if cm.max() > 0
    else 0
)

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# ROC CURVES
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    binary_true = (
        y_true == class_index
    ).astype(int)

    if len(
        np.unique(binary_true)
    ) < 2:

        continue

    fpr, tpr, _ = roc_curve(
        binary_true,
        y_prob[:, class_index]
    )

    auc = roc_auc_per_class[
        class_name
    ]

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ViT-B/16 ROC Curves"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print()
print("=" * 70)
print("VISION TRANSFORMER TEST RESULTS")
print("=" * 70)

print()
print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Precision (Macro)  : {precision_macro:.4f}"
)

print(
    f"Recall (Macro)     : {recall_macro:.4f}"
)

print(
    f"F1 Score (Macro)   : {f1_macro:.4f}"
)

print(
    f"Precision Weighted : {precision_weighted:.4f}"
)

print(
    f"Recall Weighted    : {recall_weighted:.4f}"
)

print(
    f"F1 Score Weighted  : {f1_weighted:.4f}"
)

if roc_auc_macro is not None:

    print(
        f"ROC-AUC (Macro)   : {roc_auc_macro:.4f}"
    )

print()
print("=" * 70)
print("VISION TRANSFORMER EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro F1: {f1_macro:.4f}"
)

print()
print("Evaluation outputs:")
print(OUTPUT_DIR)

print()
print("Files created:")
print("  ✓ test_predictions.csv")
print("  ✓ classification_report.csv")
print("  ✓ confusion_matrix.csv")
print("  ✓ metrics.json")
print("  ✓ confusion_matrix.png")
print("  ✓ roc_curves.png")

print()
print("✓ ViT-B/16 test evaluation finished.")