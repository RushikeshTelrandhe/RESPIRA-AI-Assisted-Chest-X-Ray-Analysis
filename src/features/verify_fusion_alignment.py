# ============================================================
# Respira - Fusion Feature Alignment Verification
# ============================================================
#
# STEP 7.1
#
# Purpose:
#   Verify that EfficientNet-B0 and Vision Transformer
#   extracted features correspond to exactly the same
#   images in exactly the same order.
#
# This verification MUST pass before starting feature fusion.
#
# Expected representations:
#
# EfficientNet-B0:
#   Spatial : (N, 1280, 7, 7)
#   Pooled  : (N, 1280)
#
# ViT-B/16:
#   Tokens  : (N, 196, 768)
#   CLS     : (N, 768)
#   Pooled  : (N, 768)
#
# Dataset:
#   6 classes
#
# ============================================================

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EFFICIENTNET_FEATURES = (
    OUTPUTS_DIR
    / "efficientnet_b0"
    / "features"
)

VIT_FEATURES = (
    OUTPUTS_DIR
    / "vit"
    / "features"
)

RESULTS_DIR = (
    OUTPUTS_DIR
    / "fusion"
    / "alignment"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SPLITS = [
    "train",
    "val",
    "test",
]

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# EXPECTED FEATURE SHAPES
# ============================================================

EXPECTED_SHAPES = {
    "efficientnet_spatial": (1280, 7, 7),
    "efficientnet_pooled": (1280,),
    "vit_tokens": (196, 768),
    "vit_cls": (768,),
    "vit_pooled": (768,),
}


# ============================================================
# TERMINAL HELPERS
# ============================================================

def print_header(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file does not exist:\n"
            f"{path}\n\n"
            f"Missing: {description}"
        )


# ============================================================
# LOAD TENSOR
# ============================================================

def load_tensor(
    path: Path,
    description: str
):
    check_file(
        path,
        description
    )

    try:
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        tensor = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{description} is not a torch.Tensor:\n"
            f"{path}"
        )

    return tensor


# ============================================================
# LOAD SPLIT FEATURES
# ============================================================

def load_features(split: str):

    print(f"\nLoading EfficientNet features...")

    efficient_spatial = load_tensor(
        EFFICIENTNET_FEATURES
        / split
        / "spatial_features.pt",
        f"EfficientNet spatial features ({split})",
    )

    efficient_pooled = load_tensor(
        EFFICIENTNET_FEATURES
        / split
        / "pooled_features.pt",
        f"EfficientNet pooled features ({split})",
    )

    efficient_labels_path = (
        EFFICIENTNET_FEATURES
        / split
        / "labels.npy"
    )

    check_file(
        efficient_labels_path,
        f"EfficientNet labels ({split})",
    )

    efficient_labels = np.load(
        efficient_labels_path
    )

    print("  ✓ EfficientNet spatial loaded")
    print("  ✓ EfficientNet pooled loaded")
    print("  ✓ EfficientNet labels loaded")

    print("\nLoading ViT features...")

    vit_tokens = load_tensor(
        VIT_FEATURES
        / split
        / "token_features.pt",
        f"ViT patch tokens ({split})",
    )

    vit_cls = load_tensor(
        VIT_FEATURES
        / split
        / "cls_features.pt",
        f"ViT CLS features ({split})",
    )

    vit_pooled = load_tensor(
        VIT_FEATURES
        / split
        / "pooled_features.pt",
        f"ViT pooled features ({split})",
    )

    vit_labels_path = (
        VIT_FEATURES
        / split
        / "labels.npy"
    )

    check_file(
        vit_labels_path,
        f"ViT labels ({split})",
    )

    vit_labels = np.load(
        vit_labels_path
    )

    print("  ✓ ViT patch tokens loaded")
    print("  ✓ ViT CLS features loaded")
    print("  ✓ ViT pooled features loaded")
    print("  ✓ ViT labels loaded")

    return {
        "efficient_spatial": efficient_spatial,
        "efficient_pooled": efficient_pooled,
        "efficient_labels": efficient_labels,
        "vit_tokens": vit_tokens,
        "vit_cls": vit_cls,
        "vit_pooled": vit_pooled,
        "vit_labels": vit_labels,
    }


# ============================================================
# SHAPE VALIDATION
# ============================================================

def validate_shapes(
    features,
    split: str
):

    print("\nFeature shapes:")

    efficient_spatial = features[
        "efficient_spatial"
    ]

    efficient_pooled = features[
        "efficient_pooled"
    ]

    vit_tokens = features[
        "vit_tokens"
    ]

    vit_cls = features[
        "vit_cls"
    ]

    vit_pooled = features[
        "vit_pooled"
    ]

    print(
        f"  EfficientNet spatial: "
        f"{tuple(efficient_spatial.shape)}"
    )

    print(
        f"  EfficientNet pooled : "
        f"{tuple(efficient_pooled.shape)}"
    )

    print(
        f"  ViT tokens          : "
        f"{tuple(vit_tokens.shape)}"
    )

    print(
        f"  ViT CLS             : "
        f"{tuple(vit_cls.shape)}"
    )

    print(
        f"  ViT pooled          : "
        f"{tuple(vit_pooled.shape)}"
    )

    print("\nShape validation:")

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if tuple(efficient_spatial.shape[1:]) != (
        EXPECTED_SHAPES[
            "efficientnet_spatial"
        ]
    ):
        raise RuntimeError(
            f"{split}: Invalid EfficientNet spatial shape."
        )

    print(
        "  ✓ EfficientNet spatial shape valid"
    )

    if tuple(efficient_pooled.shape[1:]) != (
        EXPECTED_SHAPES[
            "efficientnet_pooled"
        ]
    ):
        raise RuntimeError(
            f"{split}: Invalid EfficientNet pooled shape."
        )

    print(
        "  ✓ EfficientNet pooled shape valid"
    )

    if tuple(vit_tokens.shape[1:]) != (
        EXPECTED_SHAPES[
            "vit_tokens"
        ]
    ):
        raise RuntimeError(
            f"{split}: Invalid ViT token shape."
        )

    print(
        "  ✓ ViT token shape valid"
    )

    if tuple(vit_cls.shape[1:]) != (
        EXPECTED_SHAPES[
            "vit_cls"
        ]
    ):
        raise RuntimeError(
            f"{split}: Invalid ViT CLS shape."
        )

    print(
        "  ✓ ViT CLS shape valid"
    )

    if tuple(vit_pooled.shape[1:]) != (
        EXPECTED_SHAPES[
            "vit_pooled"
        ]
    ):
        raise RuntimeError(
            f"{split}: Invalid ViT pooled shape."
        )

    print(
        "  ✓ ViT pooled shape valid"
    )


# ============================================================
# SAMPLE COUNT VALIDATION
# ============================================================

def validate_sample_counts(
    features,
    split: str
):

    efficient_spatial = features[
        "efficient_spatial"
    ]

    efficient_pooled = features[
        "efficient_pooled"
    ]

    efficient_labels = features[
        "efficient_labels"
    ]

    vit_tokens = features[
        "vit_tokens"
    ]

    vit_cls = features[
        "vit_cls"
    ]

    vit_pooled = features[
        "vit_pooled"
    ]

    vit_labels = features[
        "vit_labels"
    ]

    n_eff_spatial = len(
        efficient_spatial
    )

    n_eff_pooled = len(
        efficient_pooled
    )

    n_eff_labels = len(
        efficient_labels
    )

    n_vit_tokens = len(
        vit_tokens
    )

    n_vit_cls = len(
        vit_cls
    )

    n_vit_pooled = len(
        vit_pooled
    )

    n_vit_labels = len(
        vit_labels
    )

    counts = [
        n_eff_spatial,
        n_eff_pooled,
        n_eff_labels,
        n_vit_tokens,
        n_vit_cls,
        n_vit_pooled,
        n_vit_labels,
    ]

    if len(set(counts)) != 1:

        raise RuntimeError(
            f"{split}: Feature sample counts do not match.\n"
            f"Counts: {counts}"
        )

    print("\nSample alignment:")

    print(
        f"  ✓ EfficientNet samples: "
        f"{n_eff_spatial}"
    )

    print(
        f"  ✓ ViT samples: "
        f"{n_vit_tokens}"
    )

    print(
        "  ✓ All feature sample counts match"
    )

    return n_eff_spatial


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(
    features,
    split: str
):

    efficient_labels = features[
        "efficient_labels"
    ]

    vit_labels = features[
        "vit_labels"
    ]

    print("\nLabel alignment:")

    if len(efficient_labels) != len(
        vit_labels
    ):
        raise RuntimeError(
            f"{split}: Label count mismatch."
        )

    if not np.array_equal(
        efficient_labels,
        vit_labels,
    ):

        mismatch_indices = np.where(
            efficient_labels
            != vit_labels
        )[0]

        print(
            f"  ✗ Label mismatch at "
            f"{len(mismatch_indices)} samples."
        )

        print(
            "\n  First mismatches:"
        )

        for idx in mismatch_indices[:10]:

            print(
                f"    Index {idx}: "
                f"EfficientNet={efficient_labels[idx]} "
                f"ViT={vit_labels[idx]}"
            )

        raise RuntimeError(
            f"{split}: EfficientNet and ViT "
            f"label ordering mismatch."
        )

    print(
        "  ✓ EfficientNet labels == ViT labels"
    )


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

def validate_numerical_values(
    features,
    split: str
):

    print("\nNumerical validation:")

    tensors = {
        "EfficientNet spatial":
            features["efficient_spatial"],

        "EfficientNet pooled":
            features["efficient_pooled"],

        "ViT tokens":
            features["vit_tokens"],

        "ViT CLS":
            features["vit_cls"],

        "ViT pooled":
            features["vit_pooled"],
    }

    for name, tensor in tensors.items():

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{split}: {name} contains "
                f"NaN or Inf values."
            )

        print(
            f"  ✓ {name}: no NaN/Inf"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def validate_class_distribution(
    features,
    split: str
):

    labels = features[
        "efficient_labels"
    ]

    print("\nClass distribution:")

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        label_int = int(label)

        if (
            label_int < 0
            or label_int >= len(CLASS_NAMES)
        ):
            class_name = "UNKNOWN"

        else:
            class_name = CLASS_NAMES[
                label_int
            ]

        print(
            f"  {label_int}: "
            f"{class_name:<22} "
            f"{count}"
        )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(
    split: str
):

    efficient_path = (
        EFFICIENTNET_FEATURES
        / split
        / "metadata.csv"
    )

    vit_path = (
        VIT_FEATURES
        / split
        / "metadata.csv"
    )

    check_file(
        efficient_path,
        f"EfficientNet metadata ({split})",
    )

    check_file(
        vit_path,
        f"ViT metadata ({split})",
    )

    efficient_metadata = pd.read_csv(
        efficient_path
    )

    vit_metadata = pd.read_csv(
        vit_path
    )

    print("\nLoading metadata...")

    print(
        f"  EfficientNet metadata: "
        f"{efficient_metadata.shape}"
    )

    print(
        f"  ViT metadata: "
        f"{vit_metadata.shape}"
    )

    return (
        efficient_metadata,
        vit_metadata,
    )


# ============================================================
# NORMALIZE PATH
# ============================================================

def normalize_path(value):

    return (
        str(value)
        .strip()
        .replace("\\", "/")
        .lower()
    )


# ============================================================
# METADATA ALIGNMENT
# ============================================================

def compare_metadata(
    efficient_metadata,
    vit_metadata,
    split: str
):

    print("\nMetadata alignment:")

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    if len(efficient_metadata) != len(
        vit_metadata
    ):

        raise RuntimeError(
            f"{split}: Metadata sample count mismatch.\n"
            f"EfficientNet: "
            f"{len(efficient_metadata)}\n"
            f"ViT: "
            f"{len(vit_metadata)}"
        )

    print(
        f"  ✓ Sample count identical: "
        f"{len(efficient_metadata)}"
    )

    # --------------------------------------------------------
    # EfficientNet path column
    # --------------------------------------------------------

    if "image_path" in (
        efficient_metadata.columns
    ):

        efficient_path_col = (
            "image_path"
        )

    elif "path" in (
        efficient_metadata.columns
    ):

        efficient_path_col = "path"

    else:

        efficient_path_col = None

    # --------------------------------------------------------
    # ViT path column
    # --------------------------------------------------------

    if "path" in (
        vit_metadata.columns
    ):

        vit_path_col = "path"

    elif "image_path" in (
        vit_metadata.columns
    ):

        vit_path_col = (
            "image_path"
        )

    else:

        vit_path_col = None

    # --------------------------------------------------------
    # Path verification
    # --------------------------------------------------------

    if (
        efficient_path_col is None
        or vit_path_col is None
    ):

        raise RuntimeError(
            f"{split}: Could not identify image "
            f"path columns.\n\n"
            f"EfficientNet columns: "
            f"{efficient_metadata.columns.tolist()}\n"
            f"ViT columns: "
            f"{vit_metadata.columns.tolist()}"
        )

    print(
        f"  EfficientNet path column: "
        f"{efficient_path_col}"
    )

    print(
        f"  ViT path column: "
        f"{vit_path_col}"
    )

    efficient_paths = [
        normalize_path(x)
        for x in efficient_metadata[
            efficient_path_col
        ]
    ]

    vit_paths = [
        normalize_path(x)
        for x in vit_metadata[
            vit_path_col
        ]
    ]

    # --------------------------------------------------------
    # Exact order comparison
    # --------------------------------------------------------

    mismatch_indices = []

    for i, (
        efficient_path,
        vit_path
    ) in enumerate(
        zip(
            efficient_paths,
            vit_paths
        )
    ):

        if efficient_path != vit_path:

            mismatch_indices.append(i)

    if mismatch_indices:

        print(
            f"\n  ✗ Image path mismatch: "
            f"{len(mismatch_indices)} samples"
        )

        print(
            "\n  First mismatches:"
        )

        for idx in mismatch_indices[:10]:

            print(
                f"\n    Index {idx}"
            )

            print(
                f"      EfficientNet: "
                f"{efficient_paths[idx]}"
            )

            print(
                f"      ViT:          "
                f"{vit_paths[idx]}"
            )

        raise RuntimeError(
            f"{split}: EfficientNet and ViT "
            f"image ordering mismatch."
        )

    print(
        f"  ✓ Image paths match exactly: "
        f"{len(efficient_paths)}/"
        f"{len(vit_paths)}"
    )

    # --------------------------------------------------------
    # Metadata labels
    # --------------------------------------------------------

    # EfficientNet
    if "label_index" in (
        efficient_metadata.columns
    ):

        efficient_meta_labels = (
            efficient_metadata[
                "label_index"
            ]
            .astype(int)
            .to_numpy()
        )

    elif "label" in (
        efficient_metadata.columns
    ):

        efficient_meta_labels = (
            efficient_metadata[
                "label"
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        efficient_meta_labels = None

    # ViT
    if "label" in (
        vit_metadata.columns
    ):

        vit_meta_labels = (
            vit_metadata[
                "label"
            ]
            .astype(int)
            .to_numpy()
        )

    elif "class_name" in (
        vit_metadata.columns
    ):

        vit_meta_labels = (
            vit_metadata[
                "class_name"
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        vit_meta_labels = None

    # --------------------------------------------------------
    # Compare metadata labels
    # --------------------------------------------------------

    if (
        efficient_meta_labels is not None
        and vit_meta_labels is not None
    ):

        if not np.array_equal(
            efficient_meta_labels,
            vit_meta_labels,
        ):

            raise RuntimeError(
                f"{split}: Metadata label ordering "
                f"does not match."
            )

        print(
            "  ✓ Metadata label ordering matches"
        )

    # --------------------------------------------------------
    # Compare image filenames
    # --------------------------------------------------------

    efficient_filenames = [
        Path(x).name
        for x in efficient_paths
    ]

    vit_filenames = [
        Path(x).name
        for x in vit_paths
    ]

    if efficient_filenames != vit_filenames:

        raise RuntimeError(
            f"{split}: Image filename ordering mismatch."
        )

    print(
        "  ✓ Image filename ordering matches"
    )

    print(
        "  ✓ EfficientNet ↔ ViT sample "
        "alignment verified"
    )


# ============================================================
# VERIFY ONE SPLIT
# ============================================================

def verify_split(
    split: str
):

    print_header(
        f"VERIFYING FUSION ALIGNMENT — {split.upper()}"
    )

    features = load_features(
        split
    )

    validate_shapes(
        features,
        split
    )

    sample_count = validate_sample_counts(
        features,
        split
    )

    validate_labels(
        features,
        split
    )

    validate_numerical_values(
        features,
        split
    )

    validate_class_distribution(
        features,
        split
    )

    (
        efficient_metadata,
        vit_metadata,
    ) = load_metadata(
        split
    )

    compare_metadata(
        efficient_metadata,
        vit_metadata,
        split
    )

    return {
        "status": "PASS",
        "samples": sample_count,
        "efficientnet_spatial_shape":
            list(
                features[
                    "efficient_spatial"
                ].shape
            ),
        "efficientnet_pooled_shape":
            list(
                features[
                    "efficient_pooled"
                ].shape
            ),
        "vit_token_shape":
            list(
                features[
                    "vit_tokens"
                ].shape
            ),
        "vit_cls_shape":
            list(
                features[
                    "vit_cls"
                ].shape
            ),
        "vit_pooled_shape":
            list(
                features[
                    "vit_pooled"
                ].shape
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "RESPIRA — FUSION FEATURE ALIGNMENT VERIFICATION"
    )

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nEfficientNet features:\n"
        f"{EFFICIENTNET_FEATURES}"
    )

    print(
        f"\nViT features:\n"
        f"{VIT_FEATURES}"
    )

    print(
        f"\nAlignment results:\n"
        f"{RESULTS_DIR}"
    )

    results = {}

    # --------------------------------------------------------
    # Verify all splits
    # --------------------------------------------------------

    for split in SPLITS:

        results[split] = verify_split(
            split
        )

        print_header(
            f"{split.upper()} ALIGNMENT PASSED"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "project": "Respira",
        "step": "7.1",
        "purpose":
            "EfficientNet-B0 and ViT feature alignment verification",
        "classes": CLASS_NAMES,
        "splits": results,
        "alignment_status": "PASS",
    }

    summary_path = (
        RESULTS_DIR
        / "alignment_summary.json"
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

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_header(
        "FUSION FEATURE ALIGNMENT COMPLETE"
    )

    print(
        "\nTrain:"
    )

    print(
        "  ✓ EfficientNet + ViT aligned"
    )

    print(
        "\nValidation:"
    )

    print(
        "  ✓ EfficientNet + ViT aligned"
    )

    print(
        "\nTest:"
    )

    print(
        "  ✓ EfficientNet + ViT aligned"
    )

    print(
        "\nAll checks passed:"
    )

    print(
        "  ✓ Feature shapes"
    )

    print(
        "  ✓ Sample counts"
    )

    print(
        "  ✓ Labels"
    )

    print(
        "  ✓ Image paths"
    )

    print(
        "  ✓ Image ordering"
    )

    print(
        "  ✓ Metadata"
    )

    print(
        "  ✓ NaN / Inf"
    )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print(
        "\n✓ STEP 7.1 COMPLETED SUCCESSFULLY."
    )

    print(
        "\nNext stage:"
    )

    print(
        "STEP 7.2 — FEATURE PROJECTION & "
        "DIMENSION ALIGNMENT"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 70)
        print("FUSION ALIGNMENT FAILED")
        print("=" * 70)

        print(
            f"\nError:\n{exc}"
        )

        print(
            "\nFix the issue above before "
            "proceeding to Step 7.2."
        )

        sys.exit(1)