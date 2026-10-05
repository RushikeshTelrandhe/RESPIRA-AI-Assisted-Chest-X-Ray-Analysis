# ============================================================
# RESPIRA - EfficientNet + ViT FINAL TEST FUSION
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

TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0_512"
    / "checkpoints"
    / "best_model.pth"
)

VIT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "vit_512"
    / "checkpoints"
    / "best_model.pth"
)

FUSION_CONFIG = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "fusion_config.json"
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


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESPIRA - FINAL EfficientNet + ViT TEST FUSION")
print("=" * 70)

print(f"Device : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Test data : {TEST_DIR}"
)

print(
    f"Output    : {OUTPUT_DIR}"
)

print("=" * 70)


# ============================================================
# 4. VERIFY FILES
# ============================================================

required_paths = [
    TEST_DIR,
    EFFICIENTNET_CHECKPOINT,
    VIT_CHECKPOINT,
    FUSION_CONFIG,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# 5. LOAD FROZEN FUSION CONFIG
# ============================================================

with open(
    FUSION_CONFIG,
    "r",
    encoding="utf-8"
) as f:

    fusion_config = json.load(f)


EFFNET_WEIGHT = float(
    fusion_config[
        "efficientnet_weight"
    ]
)

VIT_WEIGHT = float(
    fusion_config[
        "vit_weight"
    ]
)


print("\nFrozen fusion configuration:")

print(
    f"EfficientNet weight : "
    f"{EFFNET_WEIGHT:.2f}"
)

print(
    f"ViT weight          : "
    f"{VIT_WEIGHT:.2f}"
)

if not np.isclose(
    EFFNET_WEIGHT + VIT_WEIGHT,
    1.0
):

    raise ValueError(
        "Fusion weights must sum to 1."
    )


# ============================================================
# 6. TEST TRANSFORM
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
# 7. LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("\nTest dataset:")
print(
    f"Images: {len(test_dataset)}"
)

print(
    f"Classes: {test_dataset.classes}"
)


if test_dataset.classes != CLASS_NAMES:

    raise ValueError(
        "\nClass order mismatch!\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found: {test_dataset.classes}"
    )

print("✓ Class order verified")


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
# 9. LOAD EFFICIENTNET
# ============================================================

print("\nLoading EfficientNet-B0...")

effnet_checkpoint = torch.load(
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
    effnet_checkpoint[
        "model_state_dict"
    ]
)

efficientnet = efficientnet.to(DEVICE)

efficientnet.eval()

print(
    f"✓ EfficientNet epoch: "
    f"{effnet_checkpoint.get('epoch', 'N/A')}"
)

print(
    f"✓ EfficientNet validation accuracy: "
    f"{effnet_checkpoint.get('val_accuracy', 0) * 100:.2f}%"
)


# ============================================================
# 10. LOAD ViT
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
    f"✓ ViT epoch: "
    f"{vit_checkpoint.get('epoch', 'N/A')}"
)

print(
    f"✓ ViT validation accuracy: "
    f"{vit_checkpoint.get('val_accuracy', 0) * 100:.2f}%"
)


# ============================================================
# 11. FINAL TEST INFERENCE
# ============================================================

print("\nRunning final fusion inference...")

all_labels = []

all_effnet_probs = []
all_vit_probs = []
all_fusion_probs = []

batch_count = len(test_loader)

with torch.no_grad():

    for batch_index, (
        images,
        labels
    ) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # EfficientNet
        # ----------------------------------------------------

        effnet_output = efficientnet(
            images
        )

        if isinstance(
            effnet_output,
            dict
        ):

            effnet_logits = (
                effnet_output["logits"]
            )

        else:

            effnet_logits = (
                effnet_output
            )

        effnet_probs = F.softmax(
            effnet_logits,
            dim=1
        )


        # ----------------------------------------------------
        # ViT
        # ----------------------------------------------------

        vit_output = vit(
            images
        )

        if isinstance(
            vit_output,
            dict
        ):

            vit_logits = (
                vit_output["logits"]
            )

        else:

            vit_logits = vit_output

        vit_probs = F.softmax(
            vit_logits,
            dim=1
        )


        # ----------------------------------------------------
        # Weighted probability fusion
        # ----------------------------------------------------

        fusion_probs = (
            EFFNET_WEIGHT * effnet_probs
            +
            VIT_WEIGHT * vit_probs
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_effnet_probs.extend(
            effnet_probs.cpu().numpy()
        )

        all_vit_probs.extend(
            vit_probs.cpu().numpy()
        )

        all_fusion_probs.extend(
            fusion_probs.cpu().numpy()
        )


        if batch_index % 25 == 0:

            processed = min(
                batch_index * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"  Processed "
                f"{processed}/"
                f"{len(test_dataset)}"
            )


# ============================================================
# 12. NUMPY ARRAYS
# ============================================================

y_true = np.array(
    all_labels
)

effnet_probs = np.array(
    all_effnet_probs
)

vit_probs = np.array(
    all_vit_probs
)

fusion_probs = np.array(
    all_fusion_probs
)


# ============================================================
# 13. PREDICTIONS
# ============================================================

effnet_predictions = np.argmax(
    effnet_probs,
    axis=1
)

vit_predictions = np.argmax(
    vit_probs,
    axis=1
)

fusion_predictions = np.argmax(
    fusion_probs,
    axis=1
)


# ============================================================
# 14. ACCURACIES
# ============================================================

effnet_accuracy = accuracy_score(
    y_true,
    effnet_predictions
)

vit_accuracy = accuracy_score(
    y_true,
    vit_predictions
)

fusion_accuracy = accuracy_score(
    y_true,
    fusion_predictions
)


# ============================================================
# 15. FUSION METRICS
# ============================================================

(
    precision_macro,
    recall_macro,
    f1_macro,
    _
) = precision_recall_fscore_support(
    y_true,
    fusion_predictions,
    average="macro",
    zero_division=0
)

(
    precision_weighted,
    recall_weighted,
    f1_weighted,
    _
) = precision_recall_fscore_support(
    y_true,
    fusion_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    fusion_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    fusion_predictions,
    labels=np.arange(NUM_CLASSES)
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


# ============================================================
# 18. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "EfficientNet + ViT 512x512 - Fusion Confusion Matrix"
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

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

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
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. ROC-AUC
# ============================================================

y_true_one_hot = np.eye(
    NUM_CLASSES
)[y_true]

roc_auc_per_class = {}

fpr_dict = {}
tpr_dict = {}

for i, class_name in enumerate(
    CLASS_NAMES
):

    try:

        fpr, tpr, _ = roc_curve(
            y_true_one_hot[:, i],
            fusion_probs[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        roc_auc_per_class[
            class_name
        ] = float(class_auc)

        fpr_dict[
            class_name
        ] = fpr

        tpr_dict[
            class_name
        ] = tpr

    except ValueError:

        roc_auc_per_class[
            class_name
        ] = None


try:

    macro_roc_auc = roc_auc_score(
        y_true_one_hot,
        fusion_probs,
        average="macro",
        multi_class="ovr"
    )

except ValueError:

    macro_roc_auc = None


# ============================================================
# 20. ROC CURVE
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

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "EfficientNet + ViT 512x512 - Fusion ROC Curves"
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
    OUTPUT_DIR
    / "roc_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

prediction_data = {

    "true_label": [
        CLASS_NAMES[i]
        for i in y_true
    ],

    "efficientnet_prediction": [
        CLASS_NAMES[i]
        for i in effnet_predictions
    ],

    "vit_prediction": [
        CLASS_NAMES[i]
        for i in vit_predictions
    ],

    "fusion_prediction": [
        CLASS_NAMES[i]
        for i in fusion_predictions
    ],

    "fusion_correct": (
        y_true == fusion_predictions
    ),
}


# Add probabilities from all models.

for i, class_name in enumerate(
    CLASS_NAMES
):

    prediction_data[
        f"effnet_prob_{class_name}"
    ] = effnet_probs[:, i]

    prediction_data[
        f"vit_prob_{class_name}"
    ] = vit_probs[:, i]

    prediction_data[
        f"fusion_prob_{class_name}"
    ] = fusion_probs[:, i]


predictions_df = pd.DataFrame(
    prediction_data
)

predictions_df.to_csv(
    OUTPUT_DIR
    / "test_predictions.csv",
    index=False
)


# ============================================================
# 22. METRICS JSON
# ============================================================

metrics = {

    "experiment":
        "EfficientNet-B0 + ViT-B/16 weighted probability fusion",

    "input_size":
        "512x512",

    "test_samples":
        int(len(y_true)),

    "fusion_method":
        "weighted_probability_fusion",

    "efficientnet_weight":
        EFFNET_WEIGHT,

    "vit_weight":
        VIT_WEIGHT,

    "efficientnet_test_accuracy":
        float(effnet_accuracy),

    "vit_test_accuracy":
        float(vit_accuracy),

    "fusion_test_accuracy":
        float(fusion_accuracy),

    "fusion_accuracy_percent":
        float(
            fusion_accuracy * 100
        ),

    "macro_precision":
        float(precision_macro),

    "macro_recall":
        float(recall_macro),

    "macro_f1":
        float(f1_macro),

    "weighted_precision":
        float(precision_weighted),

    "weighted_recall":
        float(recall_weighted),

    "weighted_f1":
        float(f1_weighted),

    "macro_roc_auc":
        (
            float(macro_roc_auc)
            if macro_roc_auc is not None
            else None
        ),

    "per_class_roc_auc":
        roc_auc_per_class,

    "classes":
        CLASS_NAMES,

    "efficientnet_checkpoint":
        str(EFFICIENTNET_CHECKPOINT),

    "vit_checkpoint":
        str(VIT_CHECKPOINT),

    "fusion_config":
        str(FUSION_CONFIG),

    "validation_fusion_accuracy":
        float(
            fusion_config[
                "validation_accuracy"
            ]
        ),

    "validation_fusion_macro_f1":
        float(
            fusion_config[
                "validation_macro_f1"
            ]
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
# 23. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST FUSION RESULTS")
print("=" * 70)

print(
    f"Test Samples       : "
    f"{len(y_true)}"
)

print(
    f"EfficientNet-B0    : "
    f"{effnet_accuracy * 100:.2f}%"
)

print(
    f"ViT-B/16           : "
    f"{vit_accuracy * 100:.2f}%"
)

print(
    f"Fusion             : "
    f"{fusion_accuracy * 100:.2f}%"
)

print(
    f"\nFusion weights:"
)

print(
    f"EfficientNet       : "
    f"{EFFNET_WEIGHT:.2f}"
)

print(
    f"ViT                 : "
    f"{VIT_WEIGHT:.2f}"
)

print(
    f"\nFusion improvement "
    f"over EfficientNet : "
    f"{(fusion_accuracy - effnet_accuracy) * 100:+.2f} percentage points"
)

print(
    f"Fusion improvement "
    f"over ViT          : "
    f"{(fusion_accuracy - vit_accuracy) * 100:+.2f} percentage points"
)

print(
    f"\nMacro Precision    : "
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
    f"Weighted F1        : "
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
    OUTPUT_DIR
    / "classification_report.csv"
)

print(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

print(
    OUTPUT_DIR
    / "confusion_matrix.png"
)

print(
    OUTPUT_DIR
    / "metrics.json"
)

print(
    OUTPUT_DIR
    / "roc_curves.png"
)

print(
    OUTPUT_DIR
    / "test_predictions.csv"
)

print("=" * 70)

print(
    "\n✓ FINAL TEST FUSION COMPLETE"
)

print(
    "✓ Fusion weight was selected using validation data only."
)

print(
    "✓ Test set was used only for final evaluation."
)

print("=" * 70)