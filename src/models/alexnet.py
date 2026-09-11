# ============================================================
# Respira - AlexNet Model
# ============================================================
#
# Purpose:
#   AlexNet CNN baseline for Chest X-Ray classification.
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
    alexnet,
    AlexNet_Weights,
)


class AlexNetClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Load AlexNet
        # ----------------------------------------------------

        weights = (
            AlexNet_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = alexnet(
            weights=weights
        )

        # ----------------------------------------------------
        # Original AlexNet classifier:
        #
        # 9216 -> 4096 -> 4096 -> 1000
        #
        # Replace final layer with 6 classes.
        # ----------------------------------------------------

        in_features = (
            self.backbone.classifier[-1].in_features
        )

        self.backbone.classifier[-1] = nn.Linear(
            in_features,
            num_classes
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
        Extract flattened AlexNet feature vector
        before the classifier.

        Output:
            [batch_size, 9216]
        """

        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


def create_alexnet(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.5,
):

    model = AlexNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    return model


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AlexNet Model Test")
    print("=" * 70)

    model = create_alexnet(
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
    print("Features : [2, 9216]")
    print("Logits   : [2, 6]")

    print("\nTotal parameters:")

    print(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("\n✓ AlexNet model test successful.")