# ============================================================
# Respira - STEP 7.2
# Feature Projection & Dimension Alignment
# ============================================================
#
# Purpose:
#   Project EfficientNet-B0 spatial features and ViT patch
#   tokens into a common embedding dimension.
#
# Input:
#
#   EfficientNet spatial:
#       (N, 1280, 7, 7)
#
#   ViT patch tokens:
#       (N, 196, 768)
#
# Output:
#
#   EfficientNet projected tokens:
#       (N, 49, 512)
#
#   ViT projected tokens:
#       (N, 196, 512)
#
# This prepares both branches for STEP 7.3:
#   Bidirectional Cross-Attention
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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EFFICIENTNET_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "efficientnet_b0"
    / "features"
)

VIT_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
    / "features"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "projected_features"
)


# ============================================================
# CONFIGURATION
# ============================================================

SPLITS = [
    "train",
    "val",
    "test",
]

INPUT_EFFICIENTNET_DIM = 1280
INPUT_VIT_DIM = 768

PROJECTION_DIM = 512

EXPECTED_EFFICIENTNET_H = 7
EXPECTED_EFFICIENTNET_W = 7

EXPECTED_VIT_TOKENS = 196


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FEATURE PROJECTOR
# ============================================================

class FeatureProjection(nn.Module):
    """
    Projects EfficientNet and ViT features into a
    common embedding dimension.

    EfficientNet:
        1280 -> 512

    ViT:
        768 -> 512
    """

    def __init__(
        self,
        efficientnet_dim=1280,
        vit_dim=768,
        projection_dim=512,
    ):
        super().__init__()

        # ----------------------------------------------------
        # EfficientNet projection
        # ----------------------------------------------------

        self.efficientnet_projection = nn.Sequential(
            nn.Linear(
                efficientnet_dim,
                projection_dim
            ),
            nn.LayerNorm(
                projection_dim
            ),
            nn.GELU()
        )

        # ----------------------------------------------------
        # ViT projection
        # ----------------------------------------------------

        self.vit_projection = nn.Sequential(
            nn.Linear(
                vit_dim,
                projection_dim
            ),
            nn.LayerNorm(
                projection_dim
            ),
            nn.GELU()
        )

    def forward(
        self,
        efficientnet_tokens,
        vit_tokens,
    ):
        """
        Args:
            efficientnet_tokens:
                [B, 49, 1280]

            vit_tokens:
                [B, 196, 768]

        Returns:
            projected_efficientnet:
                [B, 49, 512]

            projected_vit:
                [B, 196, 512]
        """

        efficientnet_projected = (
            self.efficientnet_projection(
                efficientnet_tokens
            )
        )

        vit_projected = (
            self.vit_projection(
                vit_tokens
            )
        )

        return (
            efficientnet_projected,
            vit_projected,
        )


# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_output_directories():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in SPLITS:

        (
            OUTPUT_ROOT / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(split):

    print()
    print("=" * 70)
    print(f"LOADING FEATURES — {split.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------

    efficientnet_dir = (
        EFFICIENTNET_FEATURES / split
    )

    efficientnet_path = (
        efficientnet_dir
        / "spatial_features.pt"
    )

    efficientnet_labels_path = (
        efficientnet_dir
        / "labels.npy"
    )

    efficientnet_metadata_path = (
        efficientnet_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # ViT
    # --------------------------------------------------------

    vit_dir = (
        VIT_FEATURES / split
    )

    vit_tokens_path = (
        vit_dir
        / "token_features.pt"
    )

    vit_labels_path = (
        vit_dir
        / "labels.npy"
    )

    vit_metadata_path = (
        vit_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        efficientnet_path,
        efficientnet_labels_path,
        efficientnet_metadata_path,
        vit_tokens_path,
        vit_labels_path,
        vit_metadata_path,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file does not exist:\n{path}"
            )

    # --------------------------------------------------------
    # Load tensors
    # --------------------------------------------------------

    efficientnet_spatial = torch.load(
        efficientnet_path,
        map_location="cpu",
        weights_only=True
    )

    vit_tokens = torch.load(
        vit_tokens_path,
        map_location="cpu",
        weights_only=True
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    efficientnet_labels = np.load(
        efficientnet_labels_path
    )

    vit_labels = np.load(
        vit_labels_path
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    efficientnet_metadata = pd.read_csv(
        efficientnet_metadata_path
    )

    vit_metadata = pd.read_csv(
        vit_metadata_path
    )

    print(
        f"\nEfficientNet spatial: "
        f"{tuple(efficientnet_spatial.shape)}"
    )

    print(
        f"ViT patch tokens    : "
        f"{tuple(vit_tokens.shape)}"
    )

    print(
        f"EfficientNet labels : "
        f"{efficientnet_labels.shape}"
    )

    print(
        f"ViT labels          : "
        f"{vit_labels.shape}"
    )

    return (
        efficientnet_spatial,
        vit_tokens,
        efficientnet_labels,
        vit_labels,
        efficientnet_metadata,
        vit_metadata,
    )


# ============================================================
# VALIDATE INPUT FEATURES
# ============================================================

def validate_inputs(
    split,
    efficientnet_spatial,
    vit_tokens,
    efficientnet_labels,
    vit_labels,
    efficientnet_metadata,
    vit_metadata,
):

    print()
    print("Input validation:")

    # --------------------------------------------------------
    # EfficientNet shape
    # --------------------------------------------------------

    if efficientnet_spatial.ndim != 4:

        raise RuntimeError(
            f"{split}: EfficientNet feature must be 4D. "
            f"Found {efficientnet_spatial.ndim}D."
        )

    n_eff, channels, height, width = (
        efficientnet_spatial.shape
    )

    if channels != INPUT_EFFICIENTNET_DIM:

        raise RuntimeError(
            f"{split}: Expected EfficientNet "
            f"channels={INPUT_EFFICIENTNET_DIM}, "
            f"found {channels}."
        )

    if height != EXPECTED_EFFICIENTNET_H:
        raise RuntimeError(
            f"{split}: Expected EfficientNet height=7, "
            f"found {height}."
        )

    if width != EXPECTED_EFFICIENTNET_W:
        raise RuntimeError(
            f"{split}: Expected EfficientNet width=7, "
            f"found {width}."
        )

    print(
        "  ✓ EfficientNet spatial shape valid"
    )

    # --------------------------------------------------------
    # ViT shape
    # --------------------------------------------------------

    if vit_tokens.ndim != 3:

        raise RuntimeError(
            f"{split}: ViT tokens must be 3D. "
            f"Found {vit_tokens.ndim}D."
        )

    n_vit, num_tokens, vit_dim = (
        vit_tokens.shape
    )

    if num_tokens != EXPECTED_VIT_TOKENS:

        raise RuntimeError(
            f"{split}: Expected "
            f"{EXPECTED_VIT_TOKENS} ViT tokens, "
            f"found {num_tokens}."
        )

    if vit_dim != INPUT_VIT_DIM:

        raise RuntimeError(
            f"{split}: Expected ViT dimension "
            f"{INPUT_VIT_DIM}, found {vit_dim}."
        )

    print(
        "  ✓ ViT token shape valid"
    )

    # --------------------------------------------------------
    # Sample count
    # --------------------------------------------------------

    if n_eff != n_vit:

        raise RuntimeError(
            f"{split}: EfficientNet and ViT sample "
            f"counts differ: {n_eff} vs {n_vit}"
        )

    print(
        f"  ✓ Sample count: {n_eff}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if not np.array_equal(
        efficientnet_labels,
        vit_labels
    ):

        raise RuntimeError(
            f"{split}: EfficientNet and ViT labels "
            f"are not aligned."
        )

    print(
        "  ✓ Labels aligned"
    )

    # --------------------------------------------------------
    # Metadata sample count
    # --------------------------------------------------------

    if len(efficientnet_metadata) != n_eff:

        raise RuntimeError(
            f"{split}: EfficientNet metadata count "
            f"does not match features."
        )

    if len(vit_metadata) != n_vit:

        raise RuntimeError(
            f"{split}: ViT metadata count "
            f"does not match features."
        )

    print(
        "  ✓ Metadata counts valid"
    )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    if not torch.isfinite(
        efficientnet_spatial
    ).all():

        raise RuntimeError(
            f"{split}: EfficientNet contains NaN/Inf."
        )

    if not torch.isfinite(
        vit_tokens
    ).all():

        raise RuntimeError(
            f"{split}: ViT contains NaN/Inf."
        )

    print(
        "  ✓ No NaN / Inf detected"
    )


# ============================================================
# PROJECT FEATURES
# ============================================================

@torch.no_grad()
def project_features(
    projector,
    efficientnet_spatial,
    vit_tokens,
):

    # --------------------------------------------------------
    # EfficientNet:
    #
    # [N, 1280, 7, 7]
    #
    # -> [N, 7, 7, 1280]
    #
    # -> [N, 49, 1280]
    # --------------------------------------------------------

    efficientnet_tokens = (
        efficientnet_spatial
        .permute(
            0,
            2,
            3,
            1
        )
        .contiguous()
        .view(
            efficientnet_spatial.shape[0],
            49,
            INPUT_EFFICIENTNET_DIM
        )
    )

    # --------------------------------------------------------
    # Process in batches to reduce GPU memory
    # --------------------------------------------------------

    batch_size = 256

    projected_eff_list = []
    projected_vit_list = []

    total = efficientnet_tokens.shape[0]

    for start in tqdm(
        range(
            0,
            total,
            batch_size
        ),
        desc="Projecting features"
    ):

        end = min(
            start + batch_size,
            total
        )

        eff_batch = (
            efficientnet_tokens[
                start:end
            ]
            .to(DEVICE)
            .float()
        )

        vit_batch = (
            vit_tokens[
                start:end
            ]
            .to(DEVICE)
            .float()
        )

        eff_projected, vit_projected = (
            projector(
                eff_batch,
                vit_batch
            )
        )

        projected_eff_list.append(
            eff_projected.cpu()
        )

        projected_vit_list.append(
            vit_projected.cpu()
        )

    projected_efficientnet = torch.cat(
        projected_eff_list,
        dim=0
    )

    projected_vit = torch.cat(
        projected_vit_list,
        dim=0
    )

    return (
        projected_efficientnet,
        projected_vit
    )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    split,
    projected_efficientnet,
    projected_vit,
    labels,
    efficientnet_metadata,
):

    output_dir = (
        OUTPUT_ROOT / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    efficientnet_path = (
        output_dir
        / "efficientnet_projected.pt"
    )

    vit_path = (
        output_dir
        / "vit_projected.pt"
    )

    labels_path = (
        output_dir
        / "labels.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        projected_efficientnet,
        efficientnet_path
    )

    torch.save(
        projected_vit,
        vit_path
    )

    np.save(
        labels_path,
        labels
    )

    efficientnet_metadata.to_csv(
        metadata_path,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("Saved:")

    print(
        f"  ✓ EfficientNet:"
        f"\n    {efficientnet_path}"
    )

    print(
        f"  ✓ ViT:"
        f"\n    {vit_path}"
    )

    print(
        f"  ✓ Labels:"
        f"\n    {labels_path}"
    )

    print(
        f"  ✓ Metadata:"
        f"\n    {metadata_path}"
    )


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    split,
    projector
):

    start_time = time.time()

    (
        efficientnet_spatial,
        vit_tokens,
        efficientnet_labels,
        vit_labels,
        efficientnet_metadata,
        vit_metadata,
    ) = load_features(split)

    validate_inputs(
        split,
        efficientnet_spatial,
        vit_tokens,
        efficientnet_labels,
        vit_labels,
        efficientnet_metadata,
        vit_metadata,
    )

    print()
    print("Projecting into common dimension...")
    print(
        f"  EfficientNet: "
        f"{INPUT_EFFICIENTNET_DIM} -> "
        f"{PROJECTION_DIM}"
    )

    print(
        f"  ViT: "
        f"{INPUT_VIT_DIM} -> "
        f"{PROJECTION_DIM}"
    )

    (
        projected_efficientnet,
        projected_vit
    ) = project_features(
        projector,
        efficientnet_spatial,
        vit_tokens
    )

    # --------------------------------------------------------
    # Output validation
    # --------------------------------------------------------

    expected_eff_shape = (
        efficientnet_spatial.shape[0],
        49,
        PROJECTION_DIM
    )

    expected_vit_shape = (
        vit_tokens.shape[0],
        196,
        PROJECTION_DIM
    )

    if tuple(
        projected_efficientnet.shape
    ) != expected_eff_shape:

        raise RuntimeError(
            f"{split}: Invalid EfficientNet "
            f"output shape."
        )

    if tuple(
        projected_vit.shape
    ) != expected_vit_shape:

        raise RuntimeError(
            f"{split}: Invalid ViT "
            f"output shape."
        )

    if not torch.isfinite(
        projected_efficientnet
    ).all():

        raise RuntimeError(
            f"{split}: Projected EfficientNet "
            f"contains NaN/Inf."
        )

    if not torch.isfinite(
        projected_vit
    ).all():

        raise RuntimeError(
            f"{split}: Projected ViT "
            f"contains NaN/Inf."
        )

    print()
    print("Projected shapes:")

    print(
        f"  EfficientNet: "
        f"{tuple(projected_efficientnet.shape)}"
    )

    print(
        f"  ViT         : "
        f"{tuple(projected_vit.shape)}"
    )

    save_features(
        split,
        projected_efficientnet,
        projected_vit,
        efficientnet_labels,
        efficientnet_metadata
    )

    elapsed = time.time() - start_time

    return {
        "samples": int(efficientnet_spatial.shape[0]),
        "efficientnet_input_shape": list(
            efficientnet_spatial.shape
        ),
        "vit_input_shape": list(
            vit_tokens.shape
        ),
        "efficientnet_output_shape": list(
            projected_efficientnet.shape
        ),
        "vit_output_shape": list(
            projected_vit.shape
        ),
        "projection_dimension": PROJECTION_DIM,
        "efficientnet_tokens": 49,
        "vit_tokens": 196,
        "extraction_time_seconds": elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RESPIRA — STEP 7.2"
    )
    print(
        "FEATURE PROJECTION & DIMENSION ALIGNMENT"
    )
    print("=" * 70)

    print()
    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print()
    print(
        f"EfficientNet features:\n"
        f"{EFFICIENTNET_FEATURES}"
    )

    print()
    print(
        f"ViT features:\n"
        f"{VIT_FEATURES}"
    )

    print()
    print(
        f"Output:\n{OUTPUT_ROOT}"
    )

    print()
    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "GPU not available. "
            "Using CPU."
        )

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Create projector
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING FEATURE PROJECTOR")
    print("=" * 70)

    projector = FeatureProjection(
        efficientnet_dim=INPUT_EFFICIENTNET_DIM,
        vit_dim=INPUT_VIT_DIM,
        projection_dim=PROJECTION_DIM,
    )

    projector = projector.to(
        DEVICE
    )

    projector.eval()

    print()
    print(
        "Projection configuration:"
    )

    print(
        f"  EfficientNet input : "
        f"{INPUT_EFFICIENTNET_DIM}"
    )

    print(
        f"  ViT input          : "
        f"{INPUT_VIT_DIM}"
    )

    print(
        f"  Common dimension   : "
        f"{PROJECTION_DIM}"
    )

    print(
        "  EfficientNet tokens: 49"
    )

    print(
        "  ViT tokens         : 196"
    )

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    summary = {}

    for split in SPLITS:

        summary[split] = process_split(
            split,
            projector
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_ROOT
        / "projection_summary.json"
    )

    full_summary = {
        "step": "7.2",
        "description": (
            "Feature Projection and "
            "Dimension Alignment"
        ),
        "model_branches": {
            "cnn": "EfficientNet-B0",
            "transformer": "ViT-B/16",
        },
        "input_dimensions": {
            "efficientnet": INPUT_EFFICIENTNET_DIM,
            "vit": INPUT_VIT_DIM,
        },
        "projection_dimension": PROJECTION_DIM,
        "token_counts": {
            "efficientnet": 49,
            "vit": 196,
        },
        "splits": summary,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            full_summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 7.2 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "Projected features saved to:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Final representations:"
    )

    print(
        "  EfficientNet: "
        "(N, 49, 512)"
    )

    print(
        "  ViT         : "
        "(N, 196, 512)"
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "STEP 7.3 — BIDIRECTIONAL "
        "CROSS-ATTENTION"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    