# ============================================================
# Respira - Step 7.4
# Disease-Conditioned Attention
# ============================================================
#
# Input:
#   outputs/fusion/cross_attention/{split}/fused_tokens.pt
#
# Input shape:
#   (N, 245, 512)
#
# Output:
#   Disease-specific representations
#   (N, 6, 512)
#
# Diseases:
#   0 - Atelectasis
#   1 - Bacterial Pneumonia
#   2 - Normal
#   3 - Pulmonary Edema
#   4 - Tuberculosis
#   5 - Viral Pneumonia
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
    / "cross_attention"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "disease_attention"
)

SPLITS = ["train", "val", "test"]

NUM_DISEASES = 6
EMBED_DIM = 512

NUM_HEADS = 8

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 7.4")
print("DISEASE-CONDITIONED ATTENTION")
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
    print(
        "CUDA:",
        torch.version.cuda
    )

print()
print("Configuration:")
print("  Diseases:", NUM_DISEASES)
print("  Embedding dimension:", EMBED_DIM)
print("  Attention heads:", NUM_HEADS)
print("  Batch size:", BATCH_SIZE)

print()
print("Disease classes:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

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
# DISEASE-CONDITIONED ATTENTION MODEL
# ============================================================

class DiseaseConditionedAttention(nn.Module):
    """
    Disease-conditioned attention module.

    A learnable query is maintained for each disease.

    Each disease query attends independently to the
    fused CNN + ViT token representation.

    Input:
        fused_tokens:
            [B, 245, 512]

    Output:
        disease_features:
            [B, 6, 512]

        attention_maps:
            [B, 6, 245]
    """

    def __init__(
        self,
        num_diseases=6,
        embed_dim=512,
        num_heads=8,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Learnable disease queries
        # ----------------------------------------------------

        self.disease_queries = nn.Parameter(
            torch.randn(
                num_diseases,
                embed_dim
            )
            * 0.02
        )

        # ----------------------------------------------------
        # Query / key / value projections
        # ----------------------------------------------------

        self.query_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.key_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.value_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        # ----------------------------------------------------
        # Multi-head attention
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Feed-forward refinement
        # ----------------------------------------------------

        self.ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * 2
            ),
            nn.GELU(),
            nn.Linear(
                embed_dim * 2,
                embed_dim
            )
        )

        self.ffn_norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        fused_tokens
    ):
        """
        Args:
            fused_tokens:
                [B, 245, 512]

        Returns:
            disease_features:
                [B, 6, 512]

            attention_maps:
                [B, 6, 245]
        """

        batch_size = fused_tokens.size(0)

        # ----------------------------------------------------
        # Expand disease queries for every image
        # ----------------------------------------------------

        queries = self.disease_queries.unsqueeze(0)

        queries = queries.expand(
            batch_size,
            -1,
            -1
        )

        # ----------------------------------------------------
        # Project Q/K/V
        # ----------------------------------------------------

        q = self.query_projection(
            queries
        )

        k = self.key_projection(
            fused_tokens
        )

        v = self.value_projection(
            fused_tokens
        )

        # ----------------------------------------------------
        # Multi-head attention
        # ----------------------------------------------------

        attended, attention_weights = (
            self.attention(
                q,
                k,
                v,
                need_weights=True,
                average_attn_weights=True
            )
        )

        # attention_weights:
        #
        # [B, 6, 245]
        #

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        attended = self.norm(
            attended + queries
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        refined = self.ffn(
            attended
        )

        refined = self.ffn_norm(
            refined + attended
        )

        return (
            refined,
            attention_weights
        )


# ============================================================
# LOAD TENSOR
# ============================================================

def load_tensor(path):

    print(
        f"Loading: {path.name}"
    )

    tensor = torch.load(
        path,
        map_location="cpu"
    )

    if not isinstance(
        tensor,
        torch.Tensor
    ):
        raise TypeError(
            f"Expected Tensor but received "
            f"{type(tensor)}"
        )

    return tensor


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    features,
    split
):

    if features.ndim != 3:

        raise RuntimeError(
            f"{split}: Expected 3D tensor, "
            f"got {features.shape}"
        )

    n, tokens, dim = features.shape

    if tokens != 245:

        raise RuntimeError(
            f"{split}: Expected 245 tokens, "
            f"got {tokens}"
        )

    if dim != EMBED_DIM:

        raise RuntimeError(
            f"{split}: Expected embedding "
            f"dimension {EMBED_DIM}, "
            f"got {dim}"
        )

    if not torch.isfinite(
        features
    ).all():

        raise RuntimeError(
            f"{split}: NaN/Inf detected."
        )

    print()
    print("Feature validation:")
    print(f"  ✓ Shape: {features.shape}")
    print("  ✓ 245 fused tokens")
    print("  ✓ 512-dimensional embeddings")
    print("  ✓ No NaN / Inf")


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(
    split,
    expected_count
):

    path = (
        INPUT_ROOT
        / split
        / "labels.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Labels not found:\n{path}"
        )

    labels = np.load(
        path
    )

    if len(labels) != expected_count:

        raise RuntimeError(
            f"{split}: Label count "
            f"{len(labels)} != "
            f"{expected_count}"
        )

    return labels


# ============================================================
# PROCESS SPLIT
# ============================================================

@torch.no_grad()
def process_split(
    split,
    model
):

    print()
    print("=" * 70)
    print(
        f"DISEASE ATTENTION — "
        f"{split.upper()}"
    )
    print("=" * 70)

    split_input = (
        INPUT_ROOT
        / split
    )

    split_output = (
        OUTPUT_ROOT
        / split
    )

    fused_path = (
        split_input
        / "fused_tokens.pt"
    )

    if not fused_path.exists():

        raise FileNotFoundError(
            f"\nFused token file not found:\n"
            f"{fused_path}\n\n"
            f"Run Step 7.3 first."
        )

    # --------------------------------------------------------
    # Load fused tokens
    # --------------------------------------------------------

    fused_tokens = load_tensor(
        fused_path
    )

    n_samples = fused_tokens.shape[0]

    validate_features(
        fused_tokens,
        split
    )

    print()
    print("Images:", n_samples)
    print(
        "Input shape:",
        tuple(fused_tokens.shape)
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = load_labels(
        split,
        n_samples
    )

    print(
        "Labels:",
        labels.shape
    )

    # --------------------------------------------------------
    # Output tensors
    # --------------------------------------------------------

    disease_features = torch.empty(
        (
            n_samples,
            NUM_DISEASES,
            EMBED_DIM
        ),
        dtype=torch.float32
    )

    attention_maps = torch.empty(
        (
            n_samples,
            NUM_DISEASES,
            245
        ),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Batch inference
    # --------------------------------------------------------

    start_time = time.time()

    num_batches = (
        n_samples + BATCH_SIZE - 1
    ) // BATCH_SIZE

    progress = tqdm(
        range(num_batches),
        desc=f"Processing {split}"
    )

    for batch_idx in progress:

        start = (
            batch_idx
            * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            n_samples
        )

        batch = (
            fused_tokens[start:end]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        features, attn = model(
            batch
        )

        disease_features[
            start:end
        ] = features.cpu()

        attention_maps[
            start:end
        ] = attn.cpu()

        del batch
        del features
        del attn

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    print()
    print("Output validation:")

    print(
        "  Disease features:",
        tuple(disease_features.shape)
    )

    print(
        "  Attention maps:",
        tuple(attention_maps.shape)
    )

    if not torch.isfinite(
        disease_features
    ).all():

        raise RuntimeError(
            "NaN/Inf detected in "
            "disease features."
        )

    if not torch.isfinite(
        attention_maps
    ).all():

        raise RuntimeError(
            "NaN/Inf detected in "
            "attention maps."
        )

    print(
        "  ✓ Disease features valid"
    )

    print(
        "  ✓ Attention maps valid"
    )

    # --------------------------------------------------------
    # Verify attention normalization
    # --------------------------------------------------------

    attention_sum = (
        attention_maps.sum(
            dim=-1
        )
    )

    max_error = (
        torch.abs(
            attention_sum - 1.0
        ).max().item()
    )

    print()
    print(
        "Attention normalization:"
    )

    print(
        f"  Maximum error: "
        f"{max_error:.8f}"
    )

    if max_error < 1e-4:
        print(
            "  ✓ Attention weights "
            "properly normalized"
        )
    else:
        print(
            "  ⚠ Attention normalization "
            "difference detected"
        )

    # --------------------------------------------------------
    # Save disease features
    # --------------------------------------------------------

    feature_path = (
        split_output
        / "disease_features.pt"
    )

    torch.save(
        disease_features,
        feature_path
    )

    print()
    print(
        "✓ Saved disease features:"
    )

    print(
        feature_path
    )

    # --------------------------------------------------------
    # Save attention maps
    # --------------------------------------------------------

    attention_path = (
        split_output
        / "disease_attention_maps.pt"
    )

    torch.save(
        attention_maps,
        attention_path
    )

    print()
    print(
        "✓ Saved disease attention maps:"
    )

    print(
        attention_path
    )

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    labels_path = (
        split_output
        / "labels.npy"
    )

    np.save(
        labels_path,
        labels
    )

    print()
    print(
        "✓ Saved labels:"
    )

    print(
        labels_path
    )

    # --------------------------------------------------------
    # Copy metadata
    # --------------------------------------------------------

    metadata_source = (
        split_input
        / "metadata.csv"
    )

    metadata_destination = (
        split_output
        / "metadata.csv"
    )

    if metadata_source.exists():

        metadata = pd.read_csv(
            metadata_source
        )

        metadata.to_csv(
            metadata_destination,
            index=False
        )

        print()
        print(
            "✓ Saved metadata:"
        )

        print(
            metadata_destination
        )

    # --------------------------------------------------------
    # Disease statistics
    # --------------------------------------------------------

    mean_attention = attention_maps.mean(dim=0)

    max_attention = mean_attention.max(dim=-1).values

    summary = {

        "split": split,

        "samples": int(
            n_samples
        ),

        "input_shape": [
            int(x)
            for x in fused_tokens.shape
        ],

        "disease_feature_shape": [
            int(x)
            for x in disease_features.shape
        ],

        "attention_map_shape": [
            int(x)
            for x in attention_maps.shape
        ],

        "num_diseases": NUM_DISEASES,

        "embedding_dimension": EMBED_DIM,

        "num_tokens": 245,

        "batch_size": BATCH_SIZE,

        "processing_time_seconds": elapsed,

        "diseases": CLASS_NAMES,

        "mean_attention_per_disease": {
            CLASS_NAMES[i]: float(
                mean_attention[i].mean()
            )
            for i in range(
                NUM_DISEASES
            )
        },

        "maximum_mean_attention_per_disease": {
            CLASS_NAMES[i]: float(
                max_attention[i]
            )
            for i in range(
                NUM_DISEASES
            )
        },

        "attention_normalization_error": max_error,
    }

    summary_path = (
        split_output
        / "disease_attention_summary.json"
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

    print()
    print(
        "✓ Saved summary:"
    )

    print(
        summary_path
    )

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del fused_tokens
    del disease_features
    del attention_maps

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "INITIALIZING DISEASE-CONDITIONED "
        "ATTENTION MODEL"
    )
    print("=" * 70)

    model = DiseaseConditionedAttention(
        num_diseases=NUM_DISEASES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "✓ Disease queries initialized."
    )

    print(
        "✓ Attention module initialized."
    )

    print(
        "✓ Model moved to:",
        DEVICE
    )

    print()
    print(
        "Important:"
    )

    print(
        "Disease queries are initialized "
        "and used to generate disease-specific "
        "representations."
    )

    summaries = {}

    total_start = time.time()

    # --------------------------------------------------------
    # Process train / val / test
    # --------------------------------------------------------

    for split in SPLITS:

        summaries[split] = process_split(
            split,
            model
        )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall_summary = {

        "stage": "Step 7.4",

        "name":
            "Disease-Conditioned Attention",

        "input_representation":
            "(N, 245, 512)",

        "output_representation":
            "(N, 6, 512)",

        "attention_representation":
            "(N, 6, 245)",

        "num_diseases":
            NUM_DISEASES,

        "embedding_dimension":
            EMBED_DIM,

        "num_tokens":
            245,

        "num_attention_heads":
            NUM_HEADS,

        "batch_size":
            BATCH_SIZE,

        "device":
            str(DEVICE),

        "classes":
            CLASS_NAMES,

        "splits":
            summaries,

        "total_processing_time_seconds":
            total_time,

        "status":
            "completed",
    }

    overall_path = (
        OUTPUT_ROOT
        / "disease_attention_summary.json"
    )

    with open(
        overall_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            overall_summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 7.4 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "Disease-conditioned outputs:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Representation:"
    )

    print(
        "  Disease features: (N, 6, 512)"
    )

    print(
        "  Attention maps   : (N, 6, 245)"
    )

    print()
    print(
        "Disease-specific representations:"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {i}: {name} → 512"
        )

    print()
    print(
        "Summary:"
    )

    print(
        overall_path
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "STEP 7.5 — "
        "ADAPTIVE DISEASE-SPECIFIC FUSION"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()