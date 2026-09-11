# ============================================================
# Respira - Step 7.3
# Bidirectional Cross-Attention
# ============================================================
#
# Purpose:
#   Fuse EfficientNet spatial tokens with ViT patch tokens
#   using bidirectional cross-attention.
#
# Input:
#
#   EfficientNet:
#       (N, 49, 512)
#
#   ViT:
#       (N, 196, 512)
#
# Processing:
#
#   EfficientNet -> ViT
#       CNN queries attend to ViT tokens
#
#   ViT -> EfficientNet
#       ViT queries attend to CNN tokens
#
# Output:
#
#   CNN-enhanced representation:
#       (N, 49, 512)
#
#   ViT-enhanced representation:
#       (N, 196, 512)
#
#   Fused representation:
#       (N, 245, 512)
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

PROJECTED_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "projected_features"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "cross_attention"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMBED_DIM = 512

NUM_HEADS = 8

DROPOUT = 0.1

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


SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# CROSS-ATTENTION BLOCK
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block.

    Query comes from one modality.

    Key and Value come from the other modality.

    Example:

        CNN -> ViT

        Q = CNN tokens
        K = ViT tokens
        V = ViT tokens

    Output retains the query modality's token count.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            ),

            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_tokens,
        context_tokens,
    ):
        """
        Parameters
        ----------
        query_tokens:
            [B, Nq, D]

        context_tokens:
            [B, Nk, D]

        Returns
        -------
        output:
            [B, Nq, D]

        attention_weights:
            [B, Nq, Nk]
        """

        # ----------------------------------------------------
        # Cross attention
        # ----------------------------------------------------

        attended, attention_weights = (
            self.attention(
                query=query_tokens,
                key=context_tokens,
                value=context_tokens,
                need_weights=True,
            )
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = self.norm1(
            query_tokens + attended
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        ff = self.feed_forward(x)

        x = self.norm2(
            x + ff
        )

        return x, attention_weights


# ============================================================
# BIDIRECTIONAL CROSS-ATTENTION MODEL
# ============================================================

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional fusion between EfficientNet and ViT.

    Direction 1:
        EfficientNet queries -> ViT context

    Direction 2:
        ViT queries -> EfficientNet context
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------------------------------
        # EfficientNet -> ViT
        # ----------------------------------------------------

        self.cnn_to_vit = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # ViT -> EfficientNet
        # ----------------------------------------------------

        self.vit_to_cnn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        cnn_tokens,
        vit_tokens,
    ):
        """
        Parameters
        ----------
        cnn_tokens:
            [B, 49, 512]

        vit_tokens:
            [B, 196, 512]

        Returns
        -------
        cnn_enhanced:
            [B, 49, 512]

        vit_enhanced:
            [B, 196, 512]

        cnn_to_vit_attention:
            [B, 49, 196]

        vit_to_cnn_attention:
            [B, 196, 49]
        """

        # ----------------------------------------------------
        # Direction 1
        #
        # CNN queries attend to ViT tokens
        # ----------------------------------------------------

        cnn_enhanced, cnn_to_vit_attention = (
            self.cnn_to_vit(
                query_tokens=cnn_tokens,
                context_tokens=vit_tokens,
            )
        )

        # ----------------------------------------------------
        # Direction 2
        #
        # ViT queries attend to CNN tokens
        # ----------------------------------------------------

        vit_enhanced, vit_to_cnn_attention = (
            self.vit_to_cnn(
                query_tokens=vit_tokens,
                context_tokens=cnn_tokens,
            )
        )

        return (
            cnn_enhanced,
            vit_enhanced,
            cnn_to_vit_attention,
            vit_to_cnn_attention,
        )


# ============================================================
# LOAD PROJECTED FEATURES
# ============================================================

def load_projected_features(
    split
):
    """
    Load Step 7.2 projected features.
    """

    split_dir = (
        PROJECTED_ROOT
        / split
    )

    # --------------------------------------------------------
    # Expected filenames
    # --------------------------------------------------------

    cnn_path = (
        split_dir
        / "efficientnet_projected.pt"
    )

    vit_path = (
        split_dir
        / "vit_projected.pt"
    )

    labels_path = (
        split_dir
        / "labels.npy"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not cnn_path.exists():
        raise FileNotFoundError(
            f"\nEfficientNet projected features not found:\n"
            f"{cnn_path}\n\n"
            f"Run STEP 7.2 first."
        )

    if not vit_path.exists():
        raise FileNotFoundError(
            f"\nViT projected features not found:\n"
            f"{vit_path}\n\n"
            f"Run STEP 7.2 first."
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"\nLabels not found:\n"
            f"{labels_path}"
        )

    # --------------------------------------------------------
    # Load tensors
    # --------------------------------------------------------

    cnn_features = torch.load(
        cnn_path,
        map_location="cpu",
        weights_only=True,
    )

    vit_features = torch.load(
        vit_path,
        map_location="cpu",
        weights_only=True,
    )

    labels = np.load(
        labels_path
    )

    return (
        cnn_features,
        vit_features,
        labels,
    )


# ============================================================
# FIND FEATURE FILES
# ============================================================

def find_feature_file(
    directory,
    candidates,
):
    """
    Find the first existing file from candidates.
    """

    for filename in candidates:

        path = directory / filename

        if path.exists():
            return path

    return None


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    split
):
    """
    Load projected EfficientNet and ViT features.

    Supports the common filename variants generated
    by different Step 7.2 versions.
    """

    split_dir = (
        PROJECTED_ROOT
        / split
    )

    cnn_path = find_feature_file(
        split_dir,
        [
            "efficientnet_projected.pt",
            "efficientnet_features.pt",
            "cnn_projected.pt",
            "cnn_features.pt",
        ],
    )

    vit_path = find_feature_file(
        split_dir,
        [
            "vit_projected.pt",
            "vit_features.pt",
            "vit_projected_features.pt",
        ],
    )

    labels_path = (
        split_dir
        / "labels.npy"
    )

    if cnn_path is None:

        raise FileNotFoundError(
            f"\nCould not find EfficientNet projected features "
            f"for {split}.\n"
            f"Directory:\n{split_dir}\n\n"
            f"Expected one of:\n"
            f"  efficientnet_projected.pt\n"
            f"  efficientnet_features.pt\n"
            f"  cnn_projected.pt"
        )

    if vit_path is None:

        raise FileNotFoundError(
            f"\nCould not find ViT projected features "
            f"for {split}.\n"
            f"Directory:\n{split_dir}\n\n"
            f"Expected one of:\n"
            f"  vit_projected.pt\n"
            f"  vit_features.pt"
        )

    if not labels_path.exists():

        raise FileNotFoundError(
            f"\nLabels not found:\n"
            f"{labels_path}"
        )

    print(
        f"  EfficientNet: {cnn_path.name}"
    )

    print(
        f"  ViT         : {vit_path.name}"
    )

    cnn_features = torch.load(
        cnn_path,
        map_location="cpu",
        weights_only=True,
    )

    vit_features = torch.load(
        vit_path,
        map_location="cpu",
        weights_only=True,
    )

    labels = np.load(
        labels_path
    )

    return (
        cnn_features,
        vit_features,
        labels,
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    cnn,
    vit,
    labels,
    split,
):
    """
    Validate feature shapes and alignment.
    """

    print("\nShape validation:")

    # CNN

    if cnn.ndim != 3:

        raise RuntimeError(
            f"{split}: EfficientNet tensor must be 3D."
        )

    if cnn.shape[1:] != (
        49,
        EMBED_DIM,
    ):

        raise RuntimeError(
            f"{split}: Invalid EfficientNet shape: "
            f"{tuple(cnn.shape)}"
        )

    print(
        f"  ✓ EfficientNet: {tuple(cnn.shape)}"
    )

    # ViT

    if vit.ndim != 3:

        raise RuntimeError(
            f"{split}: ViT tensor must be 3D."
        )

    if vit.shape[1:] != (
        196,
        EMBED_DIM,
    ):

        raise RuntimeError(
            f"{split}: Invalid ViT shape: "
            f"{tuple(vit.shape)}"
        )

    print(
        f"  ✓ ViT: {tuple(vit.shape)}"
    )

    # Sample count

    if (
        cnn.shape[0]
        != vit.shape[0]
        or cnn.shape[0]
        != len(labels)
    ):

        raise RuntimeError(
            f"{split}: Sample count mismatch."
        )

    print(
        f"  ✓ Samples: {cnn.shape[0]}"
    )

    # NaN / Inf

    if not torch.isfinite(cnn).all():

        raise RuntimeError(
            f"{split}: EfficientNet contains NaN/Inf."
        )

    if not torch.isfinite(vit).all():

        raise RuntimeError(
            f"{split}: ViT contains NaN/Inf."
        )

    print(
        "  ✓ No NaN / Inf"
    )


# ============================================================
# SAVE TENSOR SAFELY
# ============================================================

def save_tensor(
    tensor,
    path,
):
    """
    Save tensor using CPU storage.
    """

    tensor = tensor.cpu()

    torch.save(
        tensor,
        path,
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

@torch.no_grad()
def process_split(
    split,
    model,
):
    """
    Run bidirectional cross-attention
    for one dataset split.
    """

    print("\n")
    print("=" * 70)
    print(
        f"STEP 7.3 — CROSS-ATTENTION — {split.upper()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading projected features...")

    (
        cnn_features,
        vit_features,
        labels,
    ) = load_features(split)

    print(
        f"  EfficientNet: {tuple(cnn_features.shape)}"
    )

    print(
        f"  ViT         : {tuple(vit_features.shape)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        cnn_features,
        vit_features,
        labels,
        split,
    )

    num_samples = cnn_features.shape[0]

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    cnn_outputs = []

    vit_outputs = []

    cnn_vit_attention = []

    vit_cnn_attention = []

    # --------------------------------------------------------
    # Batch processing
    # --------------------------------------------------------

    print(
        f"\nRunning cross-attention..."
    )

    print(
        f"Samples: {num_samples}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    start_time = time.time()

    for start in tqdm(
        range(
            0,
            num_samples,
            BATCH_SIZE
        ),
        desc=f"Cross-attention {split}",
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples
        )

        cnn_batch = (
            cnn_features[start:end]
            .to(DEVICE)
        )

        vit_batch = (
            vit_features[start:end]
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Bidirectional attention
        # ----------------------------------------------------

        (
            cnn_enhanced,
            vit_enhanced,
            cnn_to_vit,
            vit_to_cnn,
        ) = model(
            cnn_batch,
            vit_batch,
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        cnn_outputs.append(
            cnn_enhanced.cpu()
        )

        vit_outputs.append(
            vit_enhanced.cpu()
        )

        cnn_vit_attention.append(
            cnn_to_vit.cpu()
        )

        vit_cnn_attention.append(
            vit_to_cnn.cpu()
        )

    extraction_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    cnn_enhanced = torch.cat(
        cnn_outputs,
        dim=0,
    )

    vit_enhanced = torch.cat(
        vit_outputs,
        dim=0,
    )

    cnn_to_vit_attention = torch.cat(
        cnn_vit_attention,
        dim=0,
    )

    vit_to_cnn_attention = torch.cat(
        vit_cnn_attention,
        dim=0,
    )

    # --------------------------------------------------------
    # Combined representation
    # --------------------------------------------------------

    fused_tokens = torch.cat(
        [
            cnn_enhanced,
            vit_enhanced,
        ],
        dim=1,
    )

    # Shape:

    # (N, 49 + 196, 512)
    #
    # = (N, 245, 512)

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print("\nCross-attention outputs:")

    print(
        f"  CNN enhanced:"
        f" {tuple(cnn_enhanced.shape)}"
    )

    print(
        f"  ViT enhanced:"
        f" {tuple(vit_enhanced.shape)}"
    )

    print(
        f"  CNN → ViT attention:"
        f" {tuple(cnn_to_vit_attention.shape)}"
    )

    print(
        f"  ViT → CNN attention:"
        f" {tuple(vit_to_cnn_attention.shape)}"
    )

    print(
        f"  Fused tokens:"
        f" {tuple(fused_tokens.shape)}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving outputs...")

    save_tensor(
        cnn_enhanced,
        output_dir
        / "cnn_enhanced.pt",
    )

    print(
        "  ✓ CNN enhanced features saved"
    )

    save_tensor(
        vit_enhanced,
        output_dir
        / "vit_enhanced.pt",
    )

    print(
        "  ✓ ViT enhanced features saved"
    )

    save_tensor(
        fused_tokens,
        output_dir
        / "fused_tokens.pt",
    )

    print(
        "  ✓ Fused tokens saved"
    )

    save_tensor(
        cnn_to_vit_attention,
        output_dir
        / "cnn_to_vit_attention.pt",
    )

    print(
        "  ✓ CNN → ViT attention saved"
    )

    save_tensor(
        vit_to_cnn_attention,
        output_dir
        / "vit_to_cnn_attention.pt",
    )

    print(
        "  ✓ ViT → CNN attention saved"
    )

    np.save(
        output_dir
        / "labels.npy",
        labels,
    )

    print(
        "  ✓ Labels saved"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = (
        PROJECTED_ROOT
        / split
        / "metadata.csv"
    )

    if metadata_path.exists():

        metadata = pd.read_csv(
            metadata_path
        )

        metadata.to_csv(
            output_dir
            / "metadata.csv",
            index=False,
        )

        print(
            "  ✓ Metadata saved"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    summary = {

        "split": split,

        "num_samples": int(
            num_samples
        ),

        "embed_dim": EMBED_DIM,

        "num_heads": NUM_HEADS,

        "cnn_tokens": 49,

        "vit_tokens": 196,

        "fused_tokens": 245,

        "cnn_enhanced_shape": list(
            cnn_enhanced.shape
        ),

        "vit_enhanced_shape": list(
            vit_enhanced.shape
        ),

        "cnn_to_vit_attention_shape": list(
            cnn_to_vit_attention.shape
        ),

        "vit_to_cnn_attention_shape": list(
            vit_to_cnn_attention.shape
        ),

        "fused_tokens_shape": list(
            fused_tokens.shape
        ),

        "processing_time_seconds":
            extraction_time,

        "device": str(
            DEVICE
        ),

        "classes": CLASS_NAMES,
    }

    with open(
        output_dir
        / "cross_attention_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print(
        "  ✓ Summary saved"
    )

    return summary


# ============================================================
# MODEL TEST
# ============================================================

def test_model(
    model
):
    """
    Verify the cross-attention model before
    processing the complete dataset.
    """

    print("\n")
    print("=" * 70)
    print("CROSS-ATTENTION MODEL TEST")
    print("=" * 70)

    dummy_cnn = torch.randn(
        2,
        49,
        EMBED_DIM,
        device=DEVICE,
    )

    dummy_vit = torch.randn(
        2,
        196,
        EMBED_DIM,
        device=DEVICE,
    )

    with torch.no_grad():

        (
            cnn_output,
            vit_output,
            cnn_vit_attn,
            vit_cnn_attn,
        ) = model(
            dummy_cnn,
            dummy_vit,
        )

    print(
        f"\nInput CNN:"
        f" {tuple(dummy_cnn.shape)}"
    )

    print(
        f"Input ViT:"
        f" {tuple(dummy_vit.shape)}"
    )

    print(
        f"\nCNN enhanced:"
        f" {tuple(cnn_output.shape)}"
    )

    print(
        f"ViT enhanced:"
        f" {tuple(vit_output.shape)}"
    )

    print(
        f"\nCNN → ViT attention:"
        f" {tuple(cnn_vit_attn.shape)}"
    )

    print(
        f"ViT → CNN attention:"
        f" {tuple(vit_cnn_attn.shape)}"
    )

    assert cnn_output.shape == (
        2,
        49,
        512,
    )

    assert vit_output.shape == (
        2,
        196,
        512,
    )

    assert cnn_vit_attn.shape == (
        2,
        49,
        196,
    )

    assert vit_cnn_attn.shape == (
        2,
        196,
        49,
    )

    print(
        "\n✓ All cross-attention shapes verified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RESPIRA — STEP 7.3"
    )
    print(
        "BIDIRECTIONAL CROSS-ATTENTION"
    )
    print("=" * 70)

    print(
        f"\nProject root:"
        f"\n{PROJECT_ROOT}"
    )

    print(
        f"\nInput features:"
        f"\n{PROJECTED_ROOT}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_ROOT}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"\nEmbedding dimension: {EMBED_DIM}"
    )

    print(
        f"Attention heads: {NUM_HEADS}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BidirectionalCrossAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "\n✓ Bidirectional cross-attention model created."
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_model(
        model
    )

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    all_summaries = {}

    for split in SPLITS:

        all_summaries[split] = (
            process_split(
                split,
                model,
            )
        )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_summary = {

        "step": "7.3",

        "module":
            "Bidirectional Cross-Attention",

        "embed_dim":
            EMBED_DIM,

        "num_heads":
            NUM_HEADS,

        "cnn_input_tokens":
            49,

        "vit_input_tokens":
            196,

        "fused_tokens":
            245,

        "classes":
            CLASS_NAMES,

        "device":
            str(DEVICE),

        "splits":
            all_summaries,
    }

    with open(
        OUTPUT_ROOT
        / "cross_attention_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "STEP 7.3 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        "\nCross-attention outputs:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nRepresentation:"
    )

    print(
        "  EfficientNet enhanced:"
        " (N, 49, 512)"
    )

    print(
        "  ViT enhanced:"
        " (N, 196, 512)"
    )

    print(
        "  Combined:"
        " (N, 245, 512)"
    )

    print(
        "\nAttention maps:"
    )

    print(
        "  CNN → ViT:"
        " (N, 49, 196)"
    )

    print(
        "  ViT → CNN:"
        " (N, 196, 49)"
    )

    print(
        "\nNext stage:"
    )

    print(
        "STEP 7.4 — DISEASE-CONDITIONED ATTENTION"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()