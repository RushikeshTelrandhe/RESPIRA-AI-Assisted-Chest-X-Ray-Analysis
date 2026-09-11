# ============================================================
# RESPIRA — STEP 7.8
# FINAL PREDICTION + UNCERTAINTY PIPELINE
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FUSION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
)

CLASSIFICATION_ROOT = (
    FUSION_ROOT
    / "classification"
)

RELATIONSHIP_ROOT = (
    FUSION_ROOT
    / "disease_relationship"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_prediction"
)

PREDICTION_ROOT = (
    OUTPUT_ROOT
    / "predictions"
)

UNCERTAINTY_ROOT = (
    OUTPUT_ROOT
    / "uncertainty"
)

REPORT_ROOT = (
    OUTPUT_ROOT
    / "reports"
)

VISUALIZATION_ROOT = (
    OUTPUT_ROOT
    / "visualizations"
)


for directory in [
    OUTPUT_ROOT,
    PREDICTION_ROOT,
    UNCERTAINTY_ROOT,
    REPORT_ROOT,
    VISUALIZATION_ROOT,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

NUM_CLASSES = 6
FEATURE_DIM = 512
HIDDEN_DIM = 256
DROPOUT = 0.3

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INPUT FILES
# ============================================================
CHECKPOINT = (
    CLASSIFICATION_ROOT
    / "checkpoints"
    / "best_model.pth"
)

DISEASE_FEATURES = (
    FUSION_ROOT
    / "disease_attention"
    / "test"
    / "disease_features.pt"
)

RELATIONSHIP_MATRIX = (
    RELATIONSHIP_ROOT
    / "test"
    / "relationship_matrix.pt"
)

LABELS_PATH = (
    RELATIONSHIP_ROOT
    / "test"
    / "labels.npy"
)

METADATA_PATH = (
    RELATIONSHIP_ROOT
    / "test"
    / "metadata.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 7.8")
print("FINAL PREDICTION + UNCERTAINTY PIPELINE")
print("=" * 70)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = {

    "classification checkpoint":
        CHECKPOINT,

    "disease features":
        DISEASE_FEATURES,

    "relationship matrix":
        RELATIONSHIP_MATRIX,

    "labels":
        LABELS_PATH,

    "metadata":
        METADATA_PATH,
}


for name, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )

    print(f"✓ {name}")
    print(f"  {path}")


# ============================================================
# EXACT STEP 7.7 MODEL
# ============================================================

class MultiLabelClassificationHeads(nn.Module):
    """
    Exact architecture used in STEP 7.7.

    Input:
        disease_features : [B, 6, 512]
        relationship     : [B, 6, 6]

    Output:
        logits : [B, 6]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.disease_projection = nn.Sequential(

            nn.Linear(
                feature_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        self.classification_heads = (
            nn.ModuleList()
        )

        for _ in range(num_classes):

            head = nn.Sequential(

                nn.Linear(
                    hidden_dim,
                    hidden_dim // 2
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    hidden_dim // 2,
                    1
                )
            )

            self.classification_heads.append(
                head
            )

    def forward(
        self,
        disease_features,
        relationship
    ):

        # ----------------------------------------------------
        # Project disease representations
        # ----------------------------------------------------

        projected = (
            self.disease_projection(
                disease_features
            )
        )

        # projected:
        # [B, 6, 256]

        # ----------------------------------------------------
        # Relationship-based interaction
        # ----------------------------------------------------

        relationship_context = torch.bmm(
            relationship,
            projected
        )

        # [B, 6, 256]

        enhanced = (
            projected
            +
            relationship_context
        )

        # ----------------------------------------------------
        # Disease-specific heads
        # ----------------------------------------------------

        outputs = []

        for i in range(
            self.num_classes
        ):

            disease_representation = (
                enhanced[:, i, :]
            )

            logits = (
                self.classification_heads[i](
                    disease_representation
                )
            )

            outputs.append(
                logits
            )

        return torch.cat(
            outputs,
            dim=1
        )


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)


disease_features = torch.load(
    DISEASE_FEATURES,
    map_location="cpu",
    weights_only=False,
)

relationship_matrix = torch.load(
    RELATIONSHIP_MATRIX,
    map_location="cpu",
    weights_only=False,
)

labels = np.load(
    LABELS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


print()
print("Loaded shapes:")

print(
    f"  Disease features    : "
    f"{tuple(disease_features.shape)}"
)

print(
    f"  Relationship matrix : "
    f"{tuple(relationship_matrix.shape)}"
)

print(
    f"  Labels              : "
    f"{labels.shape}"
)

print(
    f"  Metadata            : "
    f"{metadata.shape}"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATING INPUT DATA")
print("=" * 70)


N = disease_features.shape[0]


if disease_features.shape != (
    N,
    NUM_CLASSES,
    FEATURE_DIM
):

    raise RuntimeError(
        "Unexpected disease feature shape."
    )


if relationship_matrix.shape != (
    N,
    NUM_CLASSES,
    NUM_CLASSES
):

    raise RuntimeError(
        "Unexpected relationship matrix shape."
    )


if len(labels) != N:

    raise RuntimeError(
        "Labels do not match feature count."
    )


if len(metadata) != N:

    raise RuntimeError(
        "Metadata does not match feature count."
    )


if not torch.isfinite(
    disease_features
).all():

    raise RuntimeError(
        "Disease features contain NaN/Inf."
    )


if not torch.isfinite(
    relationship_matrix
).all():

    raise RuntimeError(
        "Relationship matrix contains NaN/Inf."
    )


print("✓ Disease feature shape valid")
print("✓ Relationship matrix shape valid")
print("✓ Label count valid")
print("✓ Metadata count valid")
print("✓ No NaN / Inf detected")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING STEP 7.7 CHECKPOINT")
print("=" * 70)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


print(
    f"Checkpoint type: "
    f"{type(checkpoint)}"
)


if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


# ============================================================
# CREATE EXACT MODEL
# ============================================================

model = MultiLabelClassificationHeads(
    num_classes=NUM_CLASSES,
    feature_dim=FEATURE_DIM,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
).to(DEVICE)


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


print("✓ Exact STEP 7.7 architecture created")
print("✓ Checkpoint loaded")
print("✓ Model set to evaluation mode")


# ============================================================
# TEST INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING FINAL INFERENCE")
print("=" * 70)


all_logits = []


with torch.no_grad():

    for start in tqdm(
        range(
            0,
            N,
            BATCH_SIZE
        ),
        desc="Final inference"
    ):

        end = min(
            start + BATCH_SIZE,
            N
        )

        batch_features = (
            disease_features[
                start:end
            ].to(DEVICE)
        )

        batch_relationship = (
            relationship_matrix[
                start:end
            ].to(DEVICE)
        )

        batch_logits = model(
            batch_features,
            batch_relationship
        )

        all_logits.append(
            batch_logits.cpu()
        )


logits = torch.cat(
    all_logits,
    dim=0
)


print()
print(
    f"Inference completed: {N} images"
)


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = torch.sigmoid(
    logits
).numpy()


# ============================================================
# FINAL SINGLE-CLASS PREDICTION
# ============================================================

predicted_labels = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# UNCERTAINTY
# ============================================================

epsilon = 1e-8

p = np.clip(
    probabilities,
    epsilon,
    1.0 - epsilon
)


entropy = -(
    p * np.log(p)
    +
    (1.0 - p)
    * np.log(1.0 - p)
)


normalized_entropy = (
    entropy
    / np.log(2.0)
)


sample_uncertainty = (
    normalized_entropy.mean(
        axis=1
    )
)


confidence = (
    probabilities.max(
        axis=1
    )
)


# ============================================================
# TOP-1 / TOP-2 MARGIN
# ============================================================

sorted_probabilities = np.sort(
    probabilities,
    axis=1
)

top1 = sorted_probabilities[
    :, -1
]

top2 = sorted_probabilities[
    :, -2
]

prediction_margin = (
    top1 - top2
)


margin_uncertainty = (
    1.0 - prediction_margin
)


# ============================================================
# UNCERTAINTY LEVEL
# ============================================================

def get_uncertainty_level(
    value
):

    if value < 0.20:

        return "Low"

    elif value < 0.40:

        return "Moderate"

    else:

        return "High"


uncertainty_levels = [

    get_uncertainty_level(
        x
    )

    for x in sample_uncertainty

]


# ============================================================
# CREATE FINAL PREDICTION TABLE
# ============================================================

results = metadata.copy()


results[
    "true_label_index"
] = labels.astype(int)


results[
    "true_class"
] = [
    CLASS_NAMES[i]
    for i in labels
]


results[
    "predicted_label_index"
] = predicted_labels


results[
    "predicted_class"
] = [
    CLASS_NAMES[i]
    for i in predicted_labels
]


results[
    "prediction_confidence"
] = confidence


results[
    "sample_uncertainty"
] = sample_uncertainty


results[
    "margin_uncertainty"
] = margin_uncertainty


results[
    "uncertainty_level"
] = uncertainty_levels


# ============================================================
# ADD CLASS PROBABILITIES
# ============================================================

for i, class_name in enumerate(
    CLASS_NAMES
):

    column_name = (
        "prob_"
        +
        class_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    results[
        column_name
    ] = probabilities[
        :, i
    ]


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

prediction_path = (
    PREDICTION_ROOT
    / "final_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE PROBABILITIES
# ============================================================

probability_path = (
    PREDICTION_ROOT
    / "probabilities.npy"
)

np.save(
    probability_path,
    probabilities
)


# ============================================================
# SAVE UNCERTAINTY
# ============================================================

uncertainty_results = pd.DataFrame({

    "sample_index":
        np.arange(N),

    "predicted_class":
        [
            CLASS_NAMES[i]
            for i in predicted_labels
        ],

    "confidence":
        confidence,

    "sample_uncertainty":
        sample_uncertainty,

    "margin_uncertainty":
        margin_uncertainty,

    "uncertainty_level":
        uncertainty_levels,

})


uncertainty_path = (
    UNCERTAINTY_ROOT
    / "uncertainty_scores.csv"
)

uncertainty_results.to_csv(
    uncertainty_path,
    index=False
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    labels,
    predicted_labels
)

precision_macro = precision_score(
    labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

y_true_onehot = np.eye(
    NUM_CLASSES
)[labels.astype(int)]


try:

    roc_auc_macro = roc_auc_score(
        y_true_onehot,
        probabilities,
        average="macro"
    )

except Exception:

    roc_auc_macro = None


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    predicted_labels,
    labels=np.arange(
        NUM_CLASSES
    )
)


cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_path = (
    REPORT_ROOT
    / "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    labels,
    predicted_labels,
    labels=np.arange(
        NUM_CLASSES
    ),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


report_path = (
    REPORT_ROOT
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# UNCERTAINTY SUMMARY
# ============================================================

low_count = int(
    np.sum(
        sample_uncertainty < 0.20
    )
)

moderate_count = int(
    np.sum(
        (
            sample_uncertainty >= 0.20
        )
        &
        (
            sample_uncertainty < 0.40
        )
    )
)

high_count = int(
    np.sum(
        sample_uncertainty >= 0.40
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "stage":
        "STEP 7.8",

    "pipeline":
        "Final Prediction + Uncertainty",

    "model":
        "MultiLabelClassificationHeads",

    "checkpoint":
        str(CHECKPOINT),

    "test_images":
        int(N),

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "feature_dimension":
        FEATURE_DIM,

    "hidden_dimension":
        HIDDEN_DIM,

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

    "mean_confidence":
        float(
            confidence.mean()
        ),

    "mean_uncertainty":
        float(
            sample_uncertainty.mean()
        ),

    "median_uncertainty":
        float(
            np.median(
                sample_uncertainty
            )
        ),

    "mean_margin_uncertainty":
        float(
            margin_uncertainty.mean()
        ),

    "low_uncertainty_samples":
        low_count,

    "moderate_uncertainty_samples":
        moderate_count,

    "high_uncertainty_samples":
        high_count,

    "correct_predictions":
        int(
            np.sum(
                labels
                ==
                predicted_labels
            )
        ),

    "incorrect_predictions":
        int(
            np.sum(
                labels
                !=
                predicted_labels
            )
        ),

    "device":
        str(DEVICE),

}


# ============================================================
# PER-CLASS PREDICTION SUMMARY
# ============================================================

per_class = {}


for i, class_name in enumerate(
    CLASS_NAMES
):

    mask = (
        predicted_labels == i
    )

    true_mask = (
        labels == i
    )

    per_class[
        class_name
    ] = {

        "predicted_count":
            int(mask.sum()),

        "true_count":
            int(true_mask.sum()),

        "average_probability":
            float(
                probabilities[
                    :, i
                ].mean()
            ),

        "average_confidence":
            float(
                confidence[
                    mask
                ].mean()
            )
            if mask.any()
            else None,

        "average_uncertainty":
            float(
                sample_uncertainty[
                    mask
                ].mean()
            )
            if mask.any()
            else None,
    }


summary[
    "per_class"
] = per_class


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_ROOT
    / "final_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    REPORT_ROOT
    / "final_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 7.8 — FINAL RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy           : "
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
        f"ROC-AUC (Macro)    : "
        f"{roc_auc_macro:.4f}"
    )

print(
    f"Mean Confidence    : "
    f"{confidence.mean():.4f}"
)

print(
    f"Mean Uncertainty   : "
    f"{sample_uncertainty.mean():.4f}"
)

print()
print(
    f"Low uncertainty     : "
    f"{low_count}"
)

print(
    f"Moderate uncertainty: "
    f"{moderate_count}"
)

print(
    f"High uncertainty    : "
    f"{high_count}"
)

print()
print("=" * 70)
print("STEP 7.8 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Outputs:")

print(
    f"  Predictions : "
    f"{PREDICTION_ROOT}"
)

print(
    f"  Uncertainty : "
    f"{UNCERTAINTY_ROOT}"
)

print(
    f"  Reports     : "
    f"{REPORT_ROOT}"
)

print(
    f"  Summary     : "
    f"{summary_path}"
)

print()
print("✓ Final predictions generated.")
print("✓ Disease probabilities generated.")
print("✓ Uncertainty scores generated.")
print("✓ Classification report generated.")
print("✓ Confusion matrix generated.")
print("✓ Final summary generated.")