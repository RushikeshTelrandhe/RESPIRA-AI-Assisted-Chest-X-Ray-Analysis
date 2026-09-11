# ============================================================
# Respira - EfficientNet-B0 Test Evaluation
# ============================================================
#
# Purpose:
#   Evaluate the trained EfficientNet-B0 model on the
#   held-out test dataset.
#
# Outputs:
#   outputs/efficientnet_b0/evaluation/
#       test_predictions.csv
#       classification_report.csv
#       confusion_matrix.csv
#       metrics.json
#       confusion_matrix.png
#       roc_curves.png
#
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

import matplotlib.pyplot as plt

from src.models.efficientnet import (
    create_efficientnet_b0
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "efficientnet_b0"
    / "best_model.pth"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0"
    / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 6

BATCH_SIZE = 32

NUM_WORKERS = 0

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("RESPIRA - EFFICIENTNET-B0 TEST EVALUATION")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# 4. TEST TRANSFORM
# ============================================================

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


# ============================================================
# 5. TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=test_transform
)


print("\nTest dataset:")
print(
    "  Images:",
    len(test_dataset)
)

print(
    "\nDetected classes:"
)

print(
    " ",
    test_dataset.classes
)


# ============================================================
# 6. VERIFY CLASS ORDER
# ============================================================

expected_classes = sorted(
    CLASS_NAMES
)

actual_classes = sorted(
    test_dataset.classes
)

if actual_classes != expected_classes:

    raise RuntimeError(
        "\nDataset class mismatch.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {actual_classes}"
    )

print(
    "\n✓ Class order verified."
)


# ============================================================
# 7. DATA LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 8. LOAD MODEL
# ============================================================

print(
    "\nLoading EfficientNet-B0 checkpoint..."
)

model = create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print(
    "✓ Best EfficientNet-B0 checkpoint loaded."
)


# ============================================================
# 9. TEST INFERENCE
# ============================================================

all_labels = []

all_predictions = []

all_probabilities = []

print(
    "\nRunning test inference..."
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = (
            probabilities.argmax(
                dim=1
            )
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )


y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
)


print(
    "\nInference completed:",
    len(y_true),
    "images"
)


# ============================================================
# 10. BASIC METRICS
# ============================================================

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


# ============================================================
# 11. ROC-AUC
# ============================================================

y_true_onehot = np.eye(
    NUM_CLASSES
)[y_true]

roc_auc_per_class = {}

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    try:

        auc = roc_auc_score(
            y_true_onehot[:, class_index],
            y_prob[:, class_index]
        )

        roc_auc_per_class[
            class_name
        ] = float(auc)

    except ValueError:

        roc_auc_per_class[
            class_name
        ] = None


try:

    roc_auc_macro = roc_auc_score(
        y_true_onehot,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

except ValueError:

    roc_auc_macro = None


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

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
    EVALUATION_DIR
    / "classification_report.csv"
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    )
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    EVALUATION_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# 14. PREDICTIONS CSV
# ============================================================

metadata = test_dataset.samples

prediction_rows = []

for index, (
    image_path,
    true_index
) in enumerate(metadata):

    predicted_index = int(
        y_pred[index]
    )

    row = {

        "image_path":
            str(image_path),

        "true_label":
            CLASS_NAMES[true_index],

        "predicted_label":
            CLASS_NAMES[predicted_index],

        "correct":
            bool(
                true_index ==
                predicted_index
            )
    }

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        row[
            f"prob_{class_name}"
        ] = float(
            y_prob[
                index,
                class_index
            ]
        )

    prediction_rows.append(
        row
    )


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(
    EVALUATION_DIR
    / "test_predictions.csv",
    index=False
)


# ============================================================
# 15. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "EfficientNet-B0 Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    NUM_CLASSES
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

threshold = (
    cm.max() / 2.0
)

for i in range(
    NUM_CLASSES
):

    for j in range(
        NUM_CLASSES
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR
    / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. ROC CURVES
# ============================================================

plt.figure(
    figsize=(9, 7)
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    try:

        fpr, tpr, _ = roc_curve(
            y_true_onehot[
                :, class_index
            ],
            y_prob[
                :, class_index
            ]
        )

        auc = roc_auc_per_class[
            class_name
        ]

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={auc:.3f})"
            )
        )

    except ValueError:

        pass


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
    "EfficientNet-B0 ROC Curves"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR
    / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. METRICS JSON
# ============================================================

metrics = {

    "model":
        "EfficientNet-B0",

    "checkpoint":
        str(CHECKPOINT_PATH),

    "test_images":
        int(len(y_true)),

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "accuracy":
        float(accuracy),

    "precision_macro":
        float(precision_macro),

    "recall_macro":
        float(recall_macro),

    "f1_macro":
        float(f1_macro),

    "precision_weighted":
        float(precision_weighted),

    "recall_weighted":
        float(recall_weighted),

    "f1_weighted":
        float(f1_weighted),

    "roc_auc_macro":
        (
            float(roc_auc_macro)
            if roc_auc_macro is not None
            else None
        ),

    "roc_auc_per_class":
        roc_auc_per_class
}


with open(
    EVALUATION_DIR
    / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EFFICIENTNET-B0 TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy           : "
    f"{accuracy:.4f}"
)

print(
    f"Precision (Macro)  : "
    f"{precision_macro:.4f}"
)

print(
    f"Recall (Macro)     : "
    f"{recall_macro:.4f}"
)

print(
    f"F1 Score (Macro)   : "
    f"{f1_macro:.4f}"
)

print(
    f"Precision Weighted : "
    f"{precision_weighted:.4f}"
)

print(
    f"Recall Weighted    : "
    f"{recall_weighted:.4f}"
)

print(
    f"F1 Score Weighted  : "
    f"{f1_weighted:.4f}"
)

if roc_auc_macro is not None:

    print(
        f"ROC-AUC (Macro)   : "
        f"{roc_auc_macro:.4f}"
    )


# ============================================================
# 19. OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EFFICIENTNET-B0 EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Macro F1: {f1_macro:.4f}"
)

print(
    f"\nEvaluation outputs:"
)

print(
    EVALUATION_DIR
)

print("\nFiles created:")

print(
    "  ✓ test_predictions.csv"
)

print(
    "  ✓ classification_report.csv"
)

print(
    "  ✓ confusion_matrix.csv"
)

print(
    "  ✓ metrics.json"
)

print(
    "  ✓ confusion_matrix.png"
)

print(
    "  ✓ roc_curves.png"
)

print(
    "\n✓ EfficientNet-B0 test evaluation finished."
)