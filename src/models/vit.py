# ============================================================
# RESPIRA - VISION TRANSFORMER MODEL
# ============================================================
#
# Model:
#   ViT-B/16
#
# Purpose:
#   512x512 Chest X-Ray classification
#
# Input:
#   (B, 3, 512, 512)
#
# Patch size:
#   16x16
#
# Patch grid:
#   32 x 32
#
# Patch tokens:
#   1024
#
# Embedding dimension:
#   768
#
# Output:
#   logits  -> (B, 6)
#   tokens  -> (B, 1024, 768)
#   pooled  -> (B, 768)
#
# IMPORTANT:
#   Pretrained ImageNet ViT-B/16 weights are retained.
#   The pretrained 14x14 positional embeddings are
#   interpolated to the required 32x32 grid.
#
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

IMAGE_SIZE = 512

PATCH_SIZE = 16

MODEL_NAME = "ViT-B/16"

EMBED_DIM = 768

# 512 / 16 = 32
# 32 x 32 = 1024
NUM_PATCHES = (
    IMAGE_SIZE // PATCH_SIZE
) ** 2


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
    Respira ViT-B/16 classifier adapted for 512x512 images.

    Architecture:

        512x512 Chest X-Ray
                |
                v
          16x16 patching
                |
                v
        1024 patch tokens
                |
                v
          + CLS token
                |
                v
        Transformer Encoder
                |
          +-----+-----+
          |           |
          v           v
       Tokens       CLS
    [B,1024,768] [B,768]
                      |
                      v
                Classification
                      |
                      v
                  6 logits
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        self.image_size = IMAGE_SIZE
        self.patch_size = PATCH_SIZE
        self.embed_dim = EMBED_DIM

        # ----------------------------------------------------
        # Load pretrained torchvision ViT-B/16
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
        # Make torchvision input-size validation accept 512
        #
        # The patch projection itself remains 16x16.
        # ----------------------------------------------------

        self.backbone.image_size = IMAGE_SIZE

        # ----------------------------------------------------
        # ViT embedding dimension
        # ----------------------------------------------------

        self.embed_dim = (
            self.backbone.hidden_dim
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                self.embed_dim,
                num_classes
            )
        )

        # We don't use torchvision's original classifier.
        self.backbone.heads = nn.Identity()

    # ========================================================
    # POSITIONAL EMBEDDING INTERPOLATION
    # ========================================================

    def _interpolate_positional_embedding(
        self,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Convert the pretrained 14x14 positional embedding
        grid into the required positional embedding grid.

        Original pretrained ViT-B/16:

            224x224
              |
              v
            14x14
              |
              v
            196 patches

        Respira:

            512x512
              |
              v
            32x32
              |
              v
            1024 patches

        The CLS token positional embedding is retained
        separately.
        """

        pos_embedding = (
            self.backbone.encoder.pos_embedding
        )

        # Shape:
        #   [1, 197, 768]
        #
        # First token = CLS
        # Remaining 196 = 14x14 patch positions

        cls_pos = pos_embedding[
            :, :1, :
        ]

        patch_pos = pos_embedding[
            :, 1:, :
        ]

        original_grid = int(
            patch_pos.shape[1] ** 0.5
        )

        new_grid = int(
            num_patches ** 0.5
        )

        if (
            original_grid == new_grid
        ):
            return pos_embedding

        # ----------------------------------------------------
        # [1, 196, 768]
        # ->
        # [1, 768, 14, 14]
        # ----------------------------------------------------

        patch_pos = patch_pos.reshape(
            1,
            original_grid,
            original_grid,
            self.embed_dim,
        )

        patch_pos = patch_pos.permute(
            0,
            3,
            1,
            2,
        )

        # ----------------------------------------------------
        # Interpolate:
        #
        # 14x14 -> 32x32
        # ----------------------------------------------------

        patch_pos = F.interpolate(
            patch_pos,
            size=(
                new_grid,
                new_grid,
            ),
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # [1, 768, 32, 32]
        # ->
        # [1, 1024, 768]
        # ----------------------------------------------------

        patch_pos = patch_pos.permute(
            0,
            2,
            3,
            1,
        )

        patch_pos = patch_pos.reshape(
            1,
            new_grid * new_grid,
            self.embed_dim,
        )

        # ----------------------------------------------------
        # Add CLS positional embedding back
        # ----------------------------------------------------

        new_pos_embedding = torch.cat(
            [
                cls_pos,
                patch_pos,
            ],
            dim=1,
        )

        return new_pos_embedding

    # ========================================================
    # TOKEN EXTRACTION
    # ========================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Extract ViT patch tokens and CLS representation.

        Input:
            [B, 3, 512, 512]

        Returns:
            tokens:
                [B, 1024, 768]

            pooled:
                [B, 768]
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape "
                "[B, 3, H, W], "
                f"got {tuple(x.shape)}"
            )

        if (
            x.shape[-2] != IMAGE_SIZE
            or x.shape[-1] != IMAGE_SIZE
        ):
            raise ValueError(
                f"ViT-B/16 expects "
                f"{IMAGE_SIZE}x{IMAGE_SIZE} input. "
                f"Got {x.shape[-2]}x{x.shape[-1]}."
            )

        # ----------------------------------------------------
        # Patch projection
        #
        # torchvision ViT:
        #
        # 512x512
        #   ↓
        # 32x32 patches
        #   ↓
        # 1024 patches
        #   ↓
        # 768-dimensional embeddings
        # ----------------------------------------------------

        x = self.backbone._process_input(x)

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        class_token = (
            self.backbone.class_token
            .expand(
                batch_size,
                -1,
                -1
            )
        )

        # ----------------------------------------------------
        # Add CLS token
        # ----------------------------------------------------

        x = torch.cat(
            [
                class_token,
                x
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Interpolate positional embeddings
        # ----------------------------------------------------

        positional_embedding = (
            self._interpolate_positional_embedding(
                x.shape[1] - 1
            )
        )

        x = (
            x
            + positional_embedding
        )

        # ----------------------------------------------------
        # Transformer encoder
        #
        # We cannot call:
        #
        #   self.backbone.encoder(x)
        #
        # directly because torchvision's encoder adds
        # its original 197-token positional embedding.
        #
        # Therefore we apply the encoder operations manually
        # using the interpolated positional embedding.
        # ----------------------------------------------------

        x = self.backbone.encoder.dropout(
            x
        )

        x = self.backbone.encoder.layers(
            x
        )

        x = self.backbone.encoder.ln(
            x
        )

        # ----------------------------------------------------
        # Separate CLS and patch tokens
        # ----------------------------------------------------

        pooled = x[:, 0]

        tokens = x[:, 1:]

        return (
            tokens,
            pooled
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:

            logits:
                [B, 6]

            tokens:
                [B, 1024, 768]

            pooled:
                [B, 768]
        """

        tokens, pooled = (
            self.forward_features(x)
        )

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
    Freeze ViT backbone while keeping the classifier
    trainable.
    """

    for parameter in (
        model.backbone.parameters()
    ):
        parameter.requires_grad = False

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True


def unfreeze_backbone(
    model: VisionTransformerModel,
) -> None:
    """
    Unfreeze the complete ViT model.
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

    return (
        total,
        trainable
    )


# ============================================================
# MODEL TEST
# ============================================================

def test_model() -> None:

    print("=" * 70)
    print(
        "RESPIRA - ViT-B/16 512x512 MODEL TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(" ", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nLoading pretrained ViT-B/16..."
    )

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
    print(
        "  Architecture:",
        MODEL_NAME
    )

    print(
        "  Pretrained:",
        True
    )

    print(
        "  Image size:",
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        "  Patch size:",
        f"{PATCH_SIZE}x{PATCH_SIZE}"
    )

    print(
        "  Patch grid:",
        f"{IMAGE_SIZE // PATCH_SIZE}x"
        f"{IMAGE_SIZE // PATCH_SIZE}"
    )

    print(
        "  Patch tokens:",
        NUM_PATCHES
    )

    print(
        "  Embedding dimension:",
        EMBED_DIM
    )

    print(
        "  Classes:",
        NUM_CLASSES
    )

    print(
        "  Total parameters:",
        f"{total_params:,}"
    )

    print(
        "  Trainable parameters:",
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Dummy 512x512 input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    print("\nInput:")
    print(
        " ",
        tuple(dummy_input.shape)
    )

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
    print(
        " ",
        tuple(tokens.shape)
    )

    print("\nPooled features:")
    print(
        " ",
        tuple(pooled.shape)
    )

    print("\nClassification logits:")
    print(
        " ",
        tuple(logits.shape)
    )

    # --------------------------------------------------------
    # Expected shapes
    # --------------------------------------------------------

    expected_tokens = (
        2,
        1024,
        768,
    )

    expected_pooled = (
        2,
        768,
    )

    expected_logits = (
        2,
        6,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert (
        tuple(tokens.shape)
        == expected_tokens
    ), (
        "Unexpected token shape: "
        f"{tokens.shape}"
    )

    assert (
        tuple(pooled.shape)
        == expected_pooled
    ), (
        "Unexpected pooled shape: "
        f"{pooled.shape}"
    )

    assert (
        tuple(logits.shape)
        == expected_logits
    ), (
        "Unexpected logits shape: "
        f"{logits.shape}"
    )

    print(
        "\n✓ 512x512 input verified."
    )

    print(
        "✓ 1024 patch tokens verified."
    )

    print(
        "✓ 768-dimensional features verified."
    )

    print(
        "✓ 6-class output verified."
    )

    print("\n" + "=" * 70)
    print(
        "ViT-B/16 512x512 MODEL IS READY"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_model()