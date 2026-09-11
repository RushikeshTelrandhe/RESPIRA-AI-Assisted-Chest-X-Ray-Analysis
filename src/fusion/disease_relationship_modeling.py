# ============================================================
# Respira - STEP 7.6
# Disease Relationship Modeling
# ============================================================
#
# Input:
#   outputs/fusion/disease_attention/<split>/disease_features.pt
#
# Shape:
#   (N, 6, 512)
#
# Output:
#   outputs/fusion/disease_relationship/<split>/
#
#   relationship_features.pt
#   relationship_matrix.pt
#   labels.npy
#   metadata.csv
#   relationship_summary.json
#
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "disease_attention"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "disease_relationship"
)

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

NUM_HEADS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 7.6")
print("DISEASE RELATIONSHIP MODELING")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Input:")
print(INPUT_ROOT)

print()
print("Output:")
print(OUTPUT_ROOT)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)


# ============================================================
# RELATIONSHIP MODEL
# ============================================================

class DiseaseRelationshipModel(nn.Module):
    """
    Models relationships between disease-specific
    representations using multi-head self-attention.

    Input:
        [B, 6, 512]

    Output:
        enhanced disease representations:
        [B, 6, 512]

        attention matrix:
        [B, 6, 6]
    """

    def __init__(
        self,
        feature_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(feature_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):

        # ----------------------------------------------------
        # Disease-to-disease self attention
        # ----------------------------------------------------

        attended, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = self.norm1(
            x + attended
        )

        # ----------------------------------------------------
        # Feed-forward transformation
        # ----------------------------------------------------

        ff = self.feed_forward(x)

        x = self.norm2(
            x + ff
        )

        return x, attention_weights


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    split_dir = INPUT_ROOT / split

    disease_path = (
        split_dir / "disease_features.pt"
    )

    labels_path = (
        split_dir / "labels.npy"
    )

    metadata_path = (
        split_dir / "metadata.csv"
    )

    if not disease_path.exists():
        raise FileNotFoundError(
            f"\nDisease feature file missing:\n"
            f"{disease_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"\nLabels file missing:\n"
            f"{labels_path}"
        )

    print()
    print("=" * 70)
    print(
        f"LOADING DISEASE FEATURES — "
        f"{split.upper()}"
    )
    print("=" * 70)

    disease_features = torch.load(
        disease_path,
        map_location="cpu",
        weights_only=True,
    )

    labels = np.load(
        labels_path
    )

    metadata = None

    if metadata_path.exists():
        metadata = pd.read_csv(
            metadata_path
        )

    print()
    print("Disease features:")
    print(
        " ",
        tuple(disease_features.shape)
    )

    print("Labels:")
    print(
        " ",
        tuple(labels.shape)
    )

    if metadata is not None:
        print("Metadata:")
        print(
            " ",
            tuple(metadata.shape)
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if disease_features.ndim != 3:
        raise ValueError(
            "Disease features must have "
            "shape (N, 6, 512)."
        )

    if disease_features.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} disease "
            f"representations, got "
            f"{disease_features.shape[1]}."
        )

    if disease_features.shape[2] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension "
            f"{FEATURE_DIM}, got "
            f"{disease_features.shape[2]}."
        )

    if len(labels) != disease_features.shape[0]:
        raise ValueError(
            "Feature/label sample count mismatch."
        )

    if not torch.isfinite(
        disease_features
    ).all():
        raise ValueError(
            "Disease features contain NaN/Inf."
        )

    print()
    print("✓ Shape validation passed")
    print("✓ Sample count validation passed")
    print("✓ Label validation passed")
    print("✓ NaN / Inf validation passed")

    return (
        disease_features,
        labels,
        metadata,
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

@torch.no_grad()
def process_split(
    split,
    model,
):

    disease_features, labels, metadata = (
        load_split(split)
    )

    num_samples = (
        disease_features.shape[0]
    )

    output_dir = (
        OUTPUT_ROOT / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    enhanced_features = []

    attention_matrices = []

    print()
    print(
        f"Processing {num_samples} samples..."
    )

    start_time = time.time()

    for start in tqdm(
        range(
            0,
            num_samples,
            BATCH_SIZE,
        ),
        desc=f"Relationship modeling {split}",
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples,
        )

        batch = disease_features[
            start:end
        ].to(DEVICE)

        enhanced, attention = model(
            batch
        )

        enhanced_features.append(
            enhanced.cpu()
        )

        attention_matrices.append(
            attention.cpu()
        )

    enhanced_features = torch.cat(
        enhanced_features,
        dim=0,
    )

    attention_matrices = torch.cat(
        attention_matrices,
        dim=0,
    )

    elapsed = (
        time.time() - start_time
    )

    # ========================================================
    # ATTENTION HEAD AVERAGE
    # ========================================================

    # Shape:
    # [N, heads, 6, 6]

    relationship_matrix = (
        attention_matrices.mean(
            dim=1
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("Output validation:")

    print(
        "  Enhanced features:",
        tuple(
            enhanced_features.shape
        ),
    )

    print(
        "  Attention matrices:",
        tuple(
            attention_matrices.shape
        ),
    )

    print(
        "  Relationship matrix:",
        tuple(
            relationship_matrix.shape
        ),
    )

    expected_feature_shape = (
        num_samples,
        NUM_CLASSES,
        FEATURE_DIM,
    )

    expected_relationship_shape = (
        num_samples,
        NUM_CLASSES,
        NUM_CLASSES,
    )

    if tuple(
        enhanced_features.shape
    ) != expected_feature_shape:

        raise RuntimeError(
            "Enhanced feature shape mismatch."
        )

    if tuple(
        relationship_matrix.shape
    ) != expected_relationship_shape:

        raise RuntimeError(
            "Relationship matrix shape mismatch."
        )

    if not torch.isfinite(
        enhanced_features
    ).all():

        raise RuntimeError(
            "Enhanced features contain NaN/Inf."
        )

    if not torch.isfinite(
        relationship_matrix
    ).all():

        raise RuntimeError(
            "Relationship matrix contains NaN/Inf."
        )

    # --------------------------------------------------------
    # Attention normalization
    # --------------------------------------------------------

    row_sums = (
        relationship_matrix.sum(
            dim=-1
        )
    )

    normalization_error = torch.max(
        torch.abs(
            row_sums - 1.0
        )
    ).item()

    print()
    print(
        "Maximum relationship "
        "attention normalization error:",
        f"{normalization_error:.8f}",
    )

    if normalization_error > 1e-4:

        raise RuntimeError(
            "Relationship attention is not normalized."
        )

    print(
        "  ✓ Relationship attention normalized"
    )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    relationship_features_path = (
        output_dir
        / "relationship_features.pt"
    )

    relationship_matrix_path = (
        output_dir
        / "relationship_matrix.pt"
    )

    labels_path = (
        output_dir
        / "labels.npy"
    )

    metadata_output_path = (
        output_dir
        / "metadata.csv"
    )

    summary_path = (
        output_dir
        / "relationship_summary.json"
    )

    torch.save(
        enhanced_features,
        relationship_features_path,
    )

    torch.save(
        relationship_matrix,
        relationship_matrix_path,
    )

    np.save(
        labels_path,
        labels,
    )

    if metadata is not None:
        metadata.to_csv(
            metadata_output_path,
            index=False,
        )

    # ========================================================
    # CLASS RELATIONSHIP SUMMARY
    # ========================================================

    average_relationship = (
        relationship_matrix.mean(
            dim=0
        )
        .numpy()
    )

    relationship_dict = {}

    for i, source_class in enumerate(
        CLASS_NAMES
    ):

        relationship_dict[
            source_class
        ] = {}

        for j, target_class in enumerate(
            CLASS_NAMES
        ):

            relationship_dict[
                source_class
            ][target_class] = float(
                average_relationship[i, j]
            )

    # --------------------------------------------------------
    # Most influential relationships
    # --------------------------------------------------------

    relation_pairs = []

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            if i == j:
                continue

            relation_pairs.append(
                (
                    CLASS_NAMES[i],
                    CLASS_NAMES[j],
                    float(
                        average_relationship[
                            i, j
                        ]
                    ),
                )
            )

    relation_pairs.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    top_relationships = [
        {
            "source": item[0],
            "target": item[1],
            "attention": item[2],
        }
        for item in relation_pairs[:10]
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "step": "7.6",

        "module":
            "Disease Relationship Modeling",

        "split": split,

        "num_samples":
            int(num_samples),

        "num_diseases":
            NUM_CLASSES,

        "feature_dimension":
            FEATURE_DIM,

        "num_attention_heads":
            NUM_HEADS,

        "input_shape":
            list(
                disease_features.shape
            ),

        "relationship_features_shape":
            list(
                enhanced_features.shape
            ),

        "relationship_matrix_shape":
            list(
                relationship_matrix.shape
            ),

        "classes":
            CLASS_NAMES,

        "average_relationship_matrix":
            relationship_dict,

        "top_relationships":
            top_relationships,

        "normalization_error":
            float(
                normalization_error
            ),

        "processing_time_seconds":
            float(elapsed),

        "files": {

            "relationship_features":
                str(
                    relationship_features_path
                ),

            "relationship_matrix":
                str(
                    relationship_matrix_path
                ),

            "labels":
                str(labels_path),

            "metadata":
                str(
                    metadata_output_path
                ),
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print(
        "✓ Saved relationship features:"
    )

    print(
        relationship_features_path
    )

    print()
    print(
        "✓ Saved relationship matrix:"
    )

    print(
        relationship_matrix_path
    )

    print()
    print(
        "✓ Saved labels:"
    )

    print(
        labels_path
    )

    if metadata is not None:

        print()
        print(
            "✓ Saved metadata:"
        )

        print(
            metadata_output_path
        )

    print()
    print(
        "✓ Saved summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Top disease relationships:"
    )

    for item in top_relationships[:5]:

        print(
            f"  {item['source']} "
            f"→ {item['target']} : "
            f"{item['attention']:.4f}"
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "INITIALIZING RELATIONSHIP MODEL"
    )

    print(
        "=" * 70
    )

    model = DiseaseRelationshipModel(
        feature_dim=FEATURE_DIM,
        num_heads=NUM_HEADS,
    )

    model = model.to(DEVICE)

    model.eval()

    print()
    print(
        "✓ Multi-head disease relationship "
        "attention initialized"
    )

    print(
        f"  Feature dimension: {FEATURE_DIM}"
    )

    print(
        f"  Disease classes: {NUM_CLASSES}"
    )

    print(
        f"  Attention heads: {NUM_HEADS}"
    )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    splits = [
        "train",
        "val",
        "test",
    ]

    summaries = {}

    for split in splits:

        summaries[split] = process_split(
            split,
            model,
        )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    global_summary_path = (
        OUTPUT_ROOT
        / "relationship_modeling_summary.json"
    )

    global_summary = {

        "step": "7.6",

        "module":
            "Disease Relationship Modeling",

        "classes":
            CLASS_NAMES,

        "feature_dimension":
            FEATURE_DIM,

        "attention_heads":
            NUM_HEADS,

        "splits":
            summaries,
    }

    with open(
        global_summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEP 7.6 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "Disease relationship outputs:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Representations:"
    )

    print(
        "  Disease features:",
        "(N, 6, 512)",
    )

    print(
        "  Relationship matrix:",
        "(N, 6, 6)",
    )

    print()
    print(
        "✓ Train relationship features generated."
    )

    print(
        "✓ Validation relationship features generated."
    )

    print(
        "✓ Test relationship features generated."
    )

    print(
        "✓ Disease-to-disease attention generated."
    )

    print()
    print(
        "Summary:"
    )

    print(
        global_summary_path
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "STEP 7.7 — MULTI-LABEL CLASSIFICATION HEADS"
    )


if __name__ == "__main__":
    main()