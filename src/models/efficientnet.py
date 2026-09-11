# ============================================================
# Respira - EfficientNet-B0 Model
# ============================================================
#
# Purpose:
#   EfficientNet-B0 baseline model for Chest X-Ray
#   multi-class classification.
#
# Dataset:
#   6 classes
#
# Input:
#   224 x 224 RGB image
#
# Output:
#   Class logits
#
# ============================================================

from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 classifier for the Respira project.

    Architecture:

        Input Image
             |
             v
        EfficientNet-B0
             |
             v
        Feature Extractor
             |
             v
        Global Average Pooling
             |
             v
        Dropout
             |
             v
        Linear Classifier
             |
             v
        Class Logits
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Load EfficientNet-B0
        # ----------------------------------------------------

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = efficientnet_b0(
            weights=weights
        )

        # ----------------------------------------------------
        # EfficientNet-B0 classifier input size
        # ----------------------------------------------------

        in_features = (
            self.backbone.classifier[1].in_features
        )

        # ----------------------------------------------------
        # Replace original ImageNet classifier
        # ----------------------------------------------------

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                in_features,
                num_classes
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.backbone(x)

    def extract_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract EfficientNet feature vectors before
        the final classification layer.

        Output:
            [batch_size, 1280]
        """

        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


def create_efficientnet_b0(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> EfficientNetB0Classifier:

    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("EfficientNet-B0 Model Test")
    print("=" * 70)

    model = create_efficientnet_b0(
        num_classes=6,
        pretrained=True
    )

    print(model)

    # Dummy 224x224 RGB batch
    dummy = torch.randn(
        2,
        3,
        224,
        224
    )

    with torch.no_grad():

        logits = model(dummy)

        features = model.extract_features(
            dummy
        )

    print("\nInput shape:")
    print(dummy.shape)

    print("\nFeature shape:")
    print(features.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nExpected:")
    print("Features : [2, 1280]")
    print("Logits   : [2, 6]")

    print("\n✓ EfficientNet-B0 model test completed.")