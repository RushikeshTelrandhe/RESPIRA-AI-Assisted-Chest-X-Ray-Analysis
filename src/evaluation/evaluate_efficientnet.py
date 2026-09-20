# ============================================================
# RESPIRA - EfficientNet-B0 512x512 TEST EVALUATION
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
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

from src.models.efficientnet import create_efficientnet_b0


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "Lung_Disease_Preprocessed_512"
)

TEST_DIR = DATA_DIR / "test"

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
)

CHECKPOINT = (
    EXPERIMENT_DIR
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 512
BATCH_SIZE = 16
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


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESPIRA - EfficientNet-B0 512x512 TEST EVALUATION")
print("=" * 70)

print(f"Device     : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU        : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Test data  : {TEST_DIR}")
print(f"Checkpoint : {CHECKPOINT}")
print(f"Output     : {OUTPUT_DIR}")

print("=" * 70)


# ============================================================
# 4. VALIDATE PATHS
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


# ============================================================
# 5. TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
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
# 6. LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("\nTest dataset:")
print(f"Images: {len(test_dataset)}")

print("\nClasses:")
print(test_dataset.classes)


# ============================================================
# 7. VERIFY CLASS ORDER
# ============================================================

if test_dataset.classes != CLASS_NAMES:

    raise ValueError(
        "\nClass order mismatch!\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {test_dataset.classes}"
    )

print("\n✓ Class order verified")


# ============================================================
# 8. TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 9. LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

print(
    f"Checkpoint epoch          : "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Checkpoint validation acc : "
    f"{checkpoint.get('val_accuracy', 0) * 100:.2f}%"
)

print(
    f"Checkpoint validation loss: "
    f"{checkpoint.get('val_loss', 0):.4f}"
)


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=0.3,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("\n✓ EfficientNet-B0 model loaded")
print("✓ Model set to evaluation mode")


# ============================================================
# 11. TEST INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []

print("\nRunning inference on test set...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        probabilities = F.softmax(
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


# ============================================================
# 12. CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ============================================================
# 13. BASIC ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# 14. PRECISION / RECALL / F1
# ============================================================

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# 15. PER-CLASS REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# ============================================================
# 17. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "EfficientNet-B0 512x512 - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")


# Add numbers inside cells
for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. ROC-AUC
# ============================================================

y_true_one_hot = np.eye(
    NUM_CLASSES
)[y_true]

roc_auc_per_class = {}

fpr_dict = {}
tpr_dict = {}

for i, class_name in enumerate(CLASS_NAMES):

    try:

        fpr, tpr, _ = roc_curve(
            y_true_one_hot[:, i],
            y_prob[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        roc_auc_per_class[class_name] = (
            float(class_auc)
        )

        fpr_dict[class_name] = fpr
        tpr_dict[class_name] = tpr

    except ValueError:

        roc_auc_per_class[class_name] = None


# Macro ROC-AUC
try:

    macro_roc_auc = roc_auc_score(
        y_true_one_hot,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

except ValueError:

    macro_roc_auc = None


# ============================================================
# 19. ROC CURVE PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

for class_name in CLASS_NAMES:

    if class_name not in fpr_dict:
        continue

    class_auc = roc_auc_per_class[
        class_name
    ]

    plt.plot(
        fpr_dict[class_name],
        tpr_dict[class_name],
        label=(
            f"{class_name} "
            f"(AUC = {class_auc:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "EfficientNet-B0 512x512 - ROC Curves"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

prediction_data = {
    "true_label": [
        CLASS_NAMES[i]
        for i in y_true
    ],

    "predicted_label": [
        CLASS_NAMES[i]
        for i in y_pred
    ],

    "correct": (
        y_true == y_pred
    ),
}


for i, class_name in enumerate(CLASS_NAMES):

    prediction_data[
        f"prob_{class_name}"
    ] = y_prob[:, i]


predictions_df = pd.DataFrame(
    prediction_data
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 21. METRICS JSON
# ============================================================

metrics = {

    "model": "EfficientNet-B0",

    "input_size": "512x512",

    "dataset": str(TEST_DIR),

    "test_samples": int(len(y_true)),

    "accuracy": float(test_accuracy),

    "accuracy_percent": float(
        test_accuracy * 100
    ),

    "macro_precision": float(
        precision_macro
    ),

    "macro_recall": float(
        recall_macro
    ),

    "macro_f1": float(
        f1_macro
    ),

    "weighted_precision": float(
        precision_weighted
    ),

    "weighted_recall": float(
        recall_weighted
    ),

    "weighted_f1": float(
        f1_weighted
    ),

    "macro_roc_auc": (
        float(macro_roc_auc)
        if macro_roc_auc is not None
        else None
    ),

    "per_class_roc_auc": (
        roc_auc_per_class
    ),

    "classes": CLASS_NAMES,

    "checkpoint": str(
        CHECKPOINT
    ),

    "checkpoint_epoch": int(
        checkpoint.get("epoch", -1)
    ),

    "validation_accuracy": float(
        checkpoint.get(
            "val_accuracy",
            0
        )
    ),

    "validation_loss": float(
        checkpoint.get(
            "val_loss",
            0
        )
    ),
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


# ============================================================
# 22. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EFFICIENTNET-B0 512x512 TEST RESULTS")
print("=" * 70)

print(
    f"Test Samples       : {len(y_true)}"
)

print(
    f"Test Accuracy      : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{f1_macro * 100:.2f}%"
)

print(
    f"Weighted Precision : "
    f"{precision_weighted * 100:.2f}%"
)

print(
    f"Weighted Recall    : "
    f"{recall_weighted * 100:.2f}%"
)

print(
    f"Weighted F1       : "
    f"{f1_weighted * 100:.2f}%"
)

if macro_roc_auc is not None:

    print(
        f"Macro ROC-AUC      : "
        f"{macro_roc_auc:.4f}"
    )

print("\nPer-class ROC-AUC:")

for class_name in CLASS_NAMES:

    value = roc_auc_per_class[
        class_name
    ]

    if value is not None:

        print(
            f"  {class_name:<25} "
            f"{value:.4f}"
        )

    else:

        print(
            f"  {class_name:<25} N/A"
        )


print("\nOutput files:")

print(
    OUTPUT_DIR / "classification_report.csv"
)

print(
    OUTPUT_DIR / "confusion_matrix.csv"
)

print(
    OUTPUT_DIR / "confusion_matrix.png"
)

print(
    OUTPUT_DIR / "metrics.json"
)

print(
    OUTPUT_DIR / "roc_curves.png"
)

print(
    OUTPUT_DIR / "test_predictions.csv"
)

print("=" * 70)