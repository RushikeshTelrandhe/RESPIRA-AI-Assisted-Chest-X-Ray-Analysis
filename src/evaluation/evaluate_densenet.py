# ============================================================
# RESPIRA — DENSENET121 TEST EVALUATION
# ============================================================
#
# Purpose:
#   Evaluate the trained DenseNet121 model on the untouched
#   test dataset.
#
# Input:
#   data/processed/test/
#
# Checkpoint:
#   outputs/densenet/checkpoints/best_model.pth
#
# Outputs:
#   outputs/densenet/evaluation/
#       test_predictions.csv
#       classification_report.csv
#       confusion_matrix.csv
#       metrics.json
#       confusion_matrix.png
#       roc_curves.png
#
# IMPORTANT:
#   - Uses the SAME DenseNet121Model as training.
#   - Does NOT retrain the model.
#   - Does NOT modify the test dataset.
#   - Test data remains untouched.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


# ============================================================
# 2. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 3. PROJECT IMPORTS
# ============================================================

from src.models.densenet import DenseNet121Model


# ============================================================
# 4. CONFIGURATION
# ============================================================

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
    / "checkpoints"
    / "best_model.pth"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "densenet"
    / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. MODEL CONFIGURATION
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

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# 6. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 7. DATASET
# ============================================================

class ChestXrayTestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir,
        class_names,
    ):

        self.root_dir = Path(root_dir)
        self.class_names = class_names

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        self.samples = []

        self._collect_samples()

    # --------------------------------------------------------
    # COLLECT TEST IMAGES
    # --------------------------------------------------------

    def _collect_samples(self):

        if not self.root_dir.exists():

            raise FileNotFoundError(
                f"Test dataset does not exist:\n"
                f"{self.root_dir}"
            )

        for class_name in self.class_names:

            class_dir = (
                self.root_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Required class directory missing:\n"
                    f"{class_dir}"
                )

            image_files = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.is_file()
                    and p.suffix.lower()
                    in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff",
                    }
                ]
            )

            for image_path in image_files:

                self.samples.append(
                    (
                        image_path,
                        self.class_to_idx[class_name]
                    )
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No test images found in:\n"
                f"{self.root_dir}"
            )

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        from PIL import Image

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        )

        # HWC → CHW

        image = image.permute(
            2,
            0,
            1
        ).contiguous()

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            image,
            label,
            str(image_path)
        )


# ============================================================
# 8. LOAD TEST DATASET
# ============================================================

print("=" * 70)
print("RESPIRA - DENSENET121 TEST EVALUATION")
print("=" * 70)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


TEST_DIR = (
    PROCESSED_DATA
    / "test"
)


test_dataset = ChestXrayTestDataset(
    root_dir=TEST_DIR,
    class_names=CLASS_NAMES,
)


print("Test dataset:")
print("  Images:", len(test_dataset))

print()

print("Detected classes:")

detected_classes = sorted(
    set(
        path.parent.name
        for path, _ in test_dataset.samples
    )
)

print(
    " ",
    detected_classes
)


# ============================================================
# 9. VERIFY CLASS ORDER
# ============================================================

if detected_classes != sorted(CLASS_NAMES):

    raise RuntimeError(
        "\nClass mismatch!\n"
        f"Expected: {sorted(CLASS_NAMES)}\n"
        f"Detected: {detected_classes}"
    )

print()

print("✓ Class order verified.")


# ============================================================
# 10. TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 11. LOAD DENSENET121
# ============================================================

model = DenseNet121Model(
    num_classes=NUM_CLASSES,
    pretrained=False,
)

model = model.to(DEVICE)


# ============================================================
# 12. LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Best checkpoint not found:\n"
        f"{CHECKPOINT}"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)


# ------------------------------------------------------------
# Handle different checkpoint formats
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Some checkpoints are directly state_dict
        state_dict = checkpoint

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Remove possible DataParallel prefix
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[len("module."):]

    clean_state_dict[key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


if missing_keys:

    print(
        "\nWARNING: Missing model keys:"
    )

    for key in missing_keys:

        print(
            " ",
            key
        )


if unexpected_keys:

    print(
        "\nWARNING: Unexpected checkpoint keys:"
    )

    for key in unexpected_keys:

        print(
            " ",
            key
        )


model.eval()


print()

print("✓ Best DenseNet121 checkpoint loaded.")


# ============================================================
# 13. TEST INFERENCE
# ============================================================

print()

print("Running test inference...")


all_labels = []

all_predictions = []

all_probabilities = []

all_paths = []


with torch.no_grad():

    for (
        images,
        labels,
        paths
    ) in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # MODEL FORWARD
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # DenseNet121Model returns dictionary
        # ----------------------------------------------------

        if isinstance(
            outputs,
            dict
        ):

            if "logits" not in outputs:

                raise RuntimeError(
                    "DenseNet121Model output does "
                    "not contain 'logits'.\n"
                    f"Available keys: "
                    f"{list(outputs.keys())}"
                )

            logits = outputs[
                "logits"
            ]

        else:

            logits = outputs

        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_paths.extend(
            paths
        )


# ============================================================
# 14. CONVERT TO NUMPY
# ============================================================

y_true = np.asarray(
    all_labels
)

y_pred = np.asarray(
    all_predictions
)

y_prob = np.asarray(
    all_probabilities
)


print()

print(
    "Inference completed:",
    len(y_true),
    "images"
)


# ============================================================
# 15. BASIC VALIDATION
# ============================================================

if len(y_true) != len(test_dataset):

    raise RuntimeError(
        "Number of predictions does not "
        "match test dataset size."
    )


if y_prob.shape != (
    len(test_dataset),
    NUM_CLASSES
):

    raise RuntimeError(
        "Probability matrix has incorrect shape:\n"
        f"{y_prob.shape}"
    )


# ============================================================
# 16. CLASSIFICATION METRICS
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
# 17. PRINT METRICS
# ============================================================

print()
print("=" * 70)
print("DENSENET121 TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy           : {accuracy:.4f}"
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


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

classification_report_df = (
    pd.DataFrame(report)
    .transpose()
)

classification_report_path = (
    EVALUATION_DIR
    / "classification_report.csv"
)

classification_report_df.to_csv(
    classification_report_path
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

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
    EVALUATION_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# 20. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "DenseNet121 — Test Confusion Matrix"
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

threshold = cm.max() / 2.0

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
            horizontalalignment="center",
            verticalalignment="center"
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
# 21. ROC CURVES
# ============================================================
#
# One-vs-rest ROC curves.
#
# This is a multiclass classification experiment.
# Each disease/class is treated as positive against
# all other classes.
# ============================================================

y_true_one_hot = np.eye(
    NUM_CLASSES
)[y_true]


plt.figure(
    figsize=(10, 8)
)


roc_auc_values = {}


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    fpr, tpr, _ = roc_curve(
        y_true_one_hot[:, class_index],
        y_prob[:, class_index]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_auc_values[
        class_name
    ] = float(roc_auc)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC={roc_auc:.3f})"
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
    "DenseNet121 — One-vs-Rest ROC Curves"
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
    EVALUATION_DIR
    / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. PREDICTION CSV
# ============================================================

prediction_rows = []


for i in range(
    len(y_true)
):

    row = {

        "image_path":
            all_paths[i],

        "true_class":
            CLASS_NAMES[
                y_true[i]
            ],

        "predicted_class":
            CLASS_NAMES[
                y_pred[i]
            ],

        "correct":
            bool(
                y_true[i]
                ==
                y_pred[i]
            ),
    }

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        row[
            f"prob_{class_name}"
        ] = float(
            y_prob[
                i,
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
# 23. METRICS JSON
# ============================================================

metrics = {

    "model":
        "DenseNet121",

    "checkpoint":
        str(CHECKPOINT),

    "test_images":
        int(len(test_dataset)),

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

    "roc_auc_per_class":
        roc_auc_values,
}


with open(
    EVALUATION_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DENSENET121 EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    "Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Macro F1:",
    f"{f1_macro:.4f}"
)

print()

print(
    "Evaluation outputs:"
)

print(
    EVALUATION_DIR
)

print()

print("Files created:")

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

print()

print(
    "✓ DenseNet121 test evaluation finished."
)