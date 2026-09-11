# ============================================================
# RESPIRA - VISION TRANSFORMER MODEL
# ============================================================
#
# File:
#   src/models/vit.py
#
# Purpose:
#   Vision Transformer (ViT-B/16) model for the Respira project.
#
# Pipeline position:
#
#   Preprocessed Chest X-Ray
#             |
#             v
#        Vision Transformer
#             |
#             +--------------------+
#             |                    |
#             v                    v
#       ViT Tokens           Global Feature
#       (spatial/global)         Vector
#             |
#             v
#       Cross-Attention
#             |
#             v
#       DenseNet Fusion
#
# Input:
#   (B, 3, 224, 224)
#
# Output:
#   logits       -> (B, 6)
#   tokens       -> (B, 196, 768)
#   pooled       -> (B, 768)
#
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6
IMAGE_SIZE = 224

MODEL_NAME = "ViT-B/16"

# ViT-B/16 embedding dimension
EMBED_DIM = 768

# 224 / 16 = 14
# 14 x 14 = 196 patch tokens
NUM_PATCHES = 196


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# VISION TRANSFORMER MODEL
# ============================================================

class VisionTransformerModel(nn.Module):
    """
    Respira Vision Transformer model.

    Backbone:
        torchvision ViT-B/16

    Input:
        B x 3 x 224 x 224

    Outputs:
        logits:
            B x 6

        tokens:
            B x 196 x 768

        pooled:
            B x 768

    The token representation is retained because it will
    later participate in DenseNet <-> ViT cross-attention.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # ----------------------------------------------------
        # Load pretrained ViT-B/16
        # ----------------------------------------------------

        weights = (
            ViT_B_16_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = vit_b_16(
            weights=weights
        )

        # ----------------------------------------------------
        # ViT embedding dimension
        # ----------------------------------------------------

        self.embed_dim = self.backbone.hidden_dim

        # ----------------------------------------------------
        # Replace original classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.embed_dim,
                num_classes
            )
        )

        # Remove the original torchvision classification head.
        #
        # We don't use:
        #
        #   self.backbone.heads
        #
        # because we want explicit access to the
        # Transformer token representation.
        self.backbone.heads = nn.Identity()

    # ========================================================
    # TOKEN EXTRACTION
    # ========================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ViT token and global representations.

        Parameters
        ----------
        x:
            Tensor of shape:
            (B, 3, 224, 224)

        Returns
        -------
        tokens:
            (B, 196, 768)

        pooled:
            (B, 768)
        """

        # ----------------------------------------------------
        # ViT patch embedding
        # ----------------------------------------------------

        x = self.backbone._process_input(x)

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Class token
        # ----------------------------------------------------

        class_token = (
            self.backbone.class_token
            .expand(batch_size, -1, -1)
        )

        # ----------------------------------------------------
        # Add class token
        # ----------------------------------------------------

        x = torch.cat(
            [class_token, x],
            dim=1
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        x = self.backbone.encoder(x)

        # ----------------------------------------------------
        # Separate CLS token and patch tokens
        # ----------------------------------------------------

        pooled = x[:, 0]

        tokens = x[:, 1:]

        return tokens, pooled

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns a dictionary rather than only logits so
        downstream fusion can directly access ViT features.
        """

        tokens, pooled = self.forward_features(x)

        logits = self.classifier(
            pooled
        )

        return {
            "logits": logits,
            "tokens": tokens,
            "pooled": pooled,
        }


# ============================================================
# FREEZE / UNFREEZE HELPERS
# ============================================================

def freeze_backbone(
    model: VisionTransformerModel,
) -> None:
    """
    Freeze the ViT backbone.

    Useful when extracting features.
    """

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_backbone(
    model: VisionTransformerModel,
) -> None:
    """
    Unfreeze the complete ViT model.

    Useful for fine-tuning.
    """

    for parameter in model.parameters():
        parameter.requires_grad = True


# ============================================================
# PARAMETER INFORMATION
# ============================================================

def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return:

        total parameters
        trainable parameters
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# MODEL TEST
# ============================================================

def test_model() -> None:

    print("=" * 70)
    print("RESPIRA - VISION TRANSFORMER MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nLoading Vision Transformer...")

    model = VisionTransformerModel(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Parameter information
    # --------------------------------------------------------

    total_params, trainable_params = (
        count_parameters(model)
    )

    print("\nModel:")
    print("  Architecture:", MODEL_NAME)
    print("  Pretrained: True")
    print("  Classes:", NUM_CLASSES)
    print("  Embedding dimension:", EMBED_DIM)
    print("  Patch tokens:", NUM_PATCHES)
    print("  Image size:", IMAGE_SIZE)
    print("  Total parameters:", f"{total_params:,}")
    print(
        "  Trainable parameters:",
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    print("\nInput:")
    print(" ", tuple(dummy_input.shape))

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            dummy_input
        )

    logits = output["logits"]
    tokens = output["tokens"]
    pooled = output["pooled"]

    # --------------------------------------------------------
    # Display shapes
    # --------------------------------------------------------

    print("\nViT tokens:")
    print(" ", tuple(tokens.shape))

    print("\nPooled features:")
    print(" ", tuple(pooled.shape))

    print("\nClassification logits:")
    print(" ", tuple(logits.shape))

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_tokens = (
        2,
        NUM_PATCHES,
        EMBED_DIM,
    )

    expected_pooled = (
        2,
        EMBED_DIM,
    )

    expected_logits = (
        2,
        NUM_CLASSES,
    )

    assert tuple(tokens.shape) == expected_tokens, (
        f"Unexpected token shape: {tokens.shape}"
    )

    assert tuple(pooled.shape) == expected_pooled, (
        f"Unexpected pooled shape: {pooled.shape}"
    )

    assert tuple(logits.shape) == expected_logits, (
        f"Unexpected logits shape: {logits.shape}"
    )

    print("\n✓ ViT token shape verified.")
    print("✓ ViT pooled feature shape verified.")
    print("✓ Classification output verified.")

    print("\n" + "=" * 70)
    print("VISION TRANSFORMER MODEL IS READY")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_model()