# ============================================================
# Respira - MobileNetV3-Large Model
# ============================================================
#
# Purpose:
#   Lightweight CNN baseline for Chest X-Ray classification.
#
# Input:
#   224 x 224 RGB
#
# Output:
#   6 class logits
#
# Classes:
#   Atelectasis
#   Bacterial Pneumonia
#   Normal
#   Pulmonary Edema
#   Tuberculosis
#   Viral Pneumonia
#
# ============================================================

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class MobileNetV3Classifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Load pretrained MobileNetV3-Large
        # ----------------------------------------------------

        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = mobilenet_v3_large(
            weights=weights
        )

        # ----------------------------------------------------
        # Get classifier input features
        # ----------------------------------------------------

        in_features = (
            self.backbone.classifier[0].in_features
        )

        # ----------------------------------------------------
        # Replace ImageNet classifier
        # ----------------------------------------------------

        self.backbone.classifier = nn.Sequential(

            nn.Linear(
                in_features,
                1280
            ),

            nn.Hardswish(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                1280,
                num_classes
            )
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.backbone(x)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def extract_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract MobileNetV3 feature vector before
        the classification head.

        Output:
            [batch_size, 960]
        """

        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


def create_mobilenet_v3(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.3,
):

    model = MobileNetV3Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    return model


# ============================================================
# Standalone Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MobileNetV3-Large Model Test")
    print("=" * 70)

    model = create_mobilenet_v3(
        num_classes=6,
        pretrained=True
    )

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

    print("\nInput:")
    print(dummy.shape)

    print("\nFeature vector:")
    print(features.shape)

    print("\nLogits:")
    print(logits.shape)

    print("\nExpected:")
    print("Features : [2, 960]")
    print("Logits   : [2, 6]")

    print("\nTotal parameters:")

    print(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("\n✓ MobileNetV3-Large model test successful.")