"""
============================================================
RESPIRA
DenseNet121 Model
============================================================

STEP 3: DENSENET121 MODEL DEFINITION

Purpose
-------
This module defines the DenseNet121 CNN branch of the
Respira project.

Architecture:

                224 x 224 RGB Image
                         |
                         v
                  DenseNet121
                         |
                         v
                CNN Feature Map
                         |
                         +--------------------+
                         |                    |
                         v                    v
                  Spatial Features      Global Pooling
                         |                    |
                         |                    v
                         |              Feature Vector
                         |                    |
                         |                    v
                         |             Classification
                         |                    |
                         |                    v
                         |                  Logits
                         |                    |
                         |                    v
                         |                Prediction
                         |
                         v
              Later Fusion Architecture

The spatial feature map is preserved because it will
eventually be used by the multimodal fusion architecture.

Current standalone experiment:

    Image
      |
      v
    DenseNet121
      |
      v
    Features
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
    6 Class Logits

Classes
-------
0 - Atelectasis
1 - Bacterial Pneumonia
2 - Normal
3 - Pulmonary Edema
4 - Tuberculosis
5 - Viral Pneumonia

Important
---------
This file DOES NOT:
    - load the dataset
    - train the model
    - evaluate the model
    - save checkpoints

Those responsibilities belong to separate modules.

============================================================
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 6

IMAGE_SIZE = 224

FEATURE_CHANNELS = 1024


CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]


# ============================================================
# 3. DENSENET121 MODEL
# ============================================================

class DenseNet121Model(nn.Module):
    """
    DenseNet121 classifier with access to CNN spatial
    feature maps.

    Parameters
    ----------
    num_classes:
        Number of output classes.

    pretrained:
        If True, load ImageNet pretrained weights.

    dropout:
        Dropout probability before the classifier.

    freeze_backbone:
        If True, freeze DenseNet convolutional layers.

    Returns
    -------
    During forward():

        {
            "logits": classification logits,
            "features": CNN spatial feature map,
            "pooled_features": global feature vector
        }

    Shapes for 224x224 input:

        input:
            [B, 3, 224, 224]

        features:
            [B, 1024, 7, 7]

        pooled_features:
            [B, 1024]

        logits:
            [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Validate number of classes
        # ----------------------------------------------------

        if num_classes <= 0:

            raise ValueError(
                "num_classes must be greater than zero."
            )

        # ----------------------------------------------------
        # Load DenseNet121
        # ----------------------------------------------------

        if pretrained:

            weights = (
                DenseNet121_Weights.DEFAULT
            )

        else:

            weights = None

        self.backbone = densenet121(
            weights=weights
        )

        # ----------------------------------------------------
        # DenseNet feature extractor
        # ----------------------------------------------------

        self.features = (
            self.backbone.features
        )

        # ----------------------------------------------------
        # DenseNet121 output channels
        # ----------------------------------------------------

        self.feature_channels = (
            FEATURE_CHANNELS
        )

        # ----------------------------------------------------
        # Optional backbone freezing
        # ----------------------------------------------------

        if freeze_backbone:

            self.freeze_backbone()

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            self.feature_channels,
            num_classes,
        )

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.num_classes = num_classes

        self.pretrained = pretrained

        self.dropout_probability = dropout

        self.freeze_backbone_flag = (
            freeze_backbone
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape:

                [B, 3, 224, 224]

        Returns
        -------
        Dictionary containing:

            features
                CNN spatial feature map.

            pooled_features
                Global feature vector.

            logits
                Classification logits.
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "\nDenseNet expects a 4D tensor.\n"
                "Expected:\n"
                "    [batch, channels, height, width]\n"
                f"Received: {tuple(x.shape)}"
            )

        if x.shape[1] != 3:

            raise ValueError(
                "\nDenseNet expects 3-channel RGB input.\n"
                f"Received {x.shape[1]} channels."
            )

        # ----------------------------------------------------
        # CNN feature extraction
        # ----------------------------------------------------

        features = self.features(x)

        # ----------------------------------------------------
        # DenseNet normally expects ReLU after features
        # ----------------------------------------------------

        features = torch.relu(
            features
        )

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        pooled_features = torch.nn.functional.adaptive_avg_pool2d(
            features,
            output_size=(1, 1),
        )

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        pooled_features = torch.flatten(
            pooled_features,
            start_dim=1,
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        classifier_input = self.dropout(
            pooled_features
        )

        logits = self.classifier(
            classifier_input
        )

        # ----------------------------------------------------
        # Return everything required by the project
        # ----------------------------------------------------

        return {

            "features": features,

            "pooled_features":
                pooled_features,

            "logits": logits,
        }

    # ========================================================
    # FREEZE BACKBONE
    # ========================================================

    def freeze_backbone(self) -> None:

        """
        Freeze DenseNet convolutional layers.

        Useful for initial transfer-learning experiments.
        """

        for parameter in self.features.parameters():

            parameter.requires_grad = False

        self.freeze_backbone_flag = True

    # ========================================================
    # UNFREEZE BACKBONE
    # ========================================================

    def unfreeze_backbone(self) -> None:

        """
        Unfreeze DenseNet convolutional layers.

        Useful for fine-tuning.
        """

        for parameter in self.features.parameters():

            parameter.requires_grad = True

        self.freeze_backbone_flag = False

    # ========================================================
    # TRAINABLE PARAMETERS
    # ========================================================

    def trainable_parameters(self):

        """
        Return only parameters that require gradients.
        """

        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    def model_summary(self) -> Dict[str, object]:

        """
        Return model configuration information.
        """

        total_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return {

            "architecture":
                "DenseNet121",

            "pretrained":
                self.pretrained,

            "num_classes":
                self.num_classes,

            "image_size":
                IMAGE_SIZE,

            "feature_channels":
                self.feature_channels,

            "dropout":
                self.dropout_probability,

            "backbone_frozen":
                self.freeze_backbone_flag,

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "classes":
                CLASS_NAMES.copy(),
        }


# ============================================================
# 4. MODEL FACTORY
# ============================================================

def create_densenet121(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> DenseNet121Model:

    """
    Factory function for creating DenseNet121.

    Example
    -------

        model = create_densenet121(
            pretrained=True
        )
    """

    return DenseNet121Model(

        num_classes=num_classes,

        pretrained=pretrained,

        dropout=dropout,

        freeze_backbone=freeze_backbone,
    )


# ============================================================
# 5. DEBUG / MANUAL TEST
# ============================================================

def test_model() -> None:

    """
    Run a complete forward-pass test.

    This does NOT train the model.
    """

    print("=" * 70)

    print(
        "RESPIRA - DENSENET121 MODEL TEST"
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

    print(
        "\nDevice:",
        device
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_densenet121(

        num_classes=NUM_CLASSES,

        pretrained=True,

        dropout=0.2,

        freeze_backbone=False,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        model.model_summary()
    )

    print(
        "\nModel:"
    )

    print(
        "  Architecture:",
        summary["architecture"]
    )

    print(
        "  Pretrained:",
        summary["pretrained"]
    )

    print(
        "  Classes:",
        summary["num_classes"]
    )

    print(
        "  Feature channels:",
        summary["feature_channels"]
    )

    print(
        "  Image size:",
        summary["image_size"]
    )

    print(
        "  Total parameters:",
        f"{summary['total_parameters']:,}"
    )

    print(
        "  Trainable parameters:",
        f"{summary['trainable_parameters']:,}"
    )

    # --------------------------------------------------------
    # Dummy image
    # --------------------------------------------------------

    dummy_input = torch.randn(

        2,

        3,

        IMAGE_SIZE,

        IMAGE_SIZE,
    ).to(device)

    print(
        "\nInput:"
    )

    print(
        " ",
        tuple(dummy_input.shape)
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(
            dummy_input
        )

    # --------------------------------------------------------
    # Feature map
    # --------------------------------------------------------

    features = (
        outputs["features"]
    )

    print(
        "\nCNN feature map:"
    )

    print(
        " ",
        tuple(features.shape)
    )

    # --------------------------------------------------------
    # Pooled feature
    # --------------------------------------------------------

    pooled_features = (
        outputs["pooled_features"]
    )

    print(
        "\nPooled features:"
    )

    print(
        " ",
        tuple(
            pooled_features.shape
        )
    )

    # --------------------------------------------------------
    # Logits
    # --------------------------------------------------------

    logits = outputs["logits"]

    print(
        "\nClassification logits:"
    )

    print(
        " ",
        tuple(logits.shape)
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_features = (
        2,
        FEATURE_CHANNELS,
        7,
        7,
    )

    expected_pooled = (
        2,
        FEATURE_CHANNELS,
    )

    expected_logits = (
        2,
        NUM_CLASSES,
    )

    if tuple(features.shape) != (
        expected_features
    ):

        raise RuntimeError(
            "\nUnexpected feature map shape.\n"
            f"Expected: {expected_features}\n"
            f"Actual: {tuple(features.shape)}"
        )

    if tuple(pooled_features.shape) != (
        expected_pooled
    ):

        raise RuntimeError(
            "\nUnexpected pooled feature shape.\n"
            f"Expected: {expected_pooled}\n"
            f"Actual: {tuple(pooled_features.shape)}"
        )

    if tuple(logits.shape) != (
        expected_logits
    ):

        raise RuntimeError(
            "\nUnexpected logits shape.\n"
            f"Expected: {expected_logits}\n"
            f"Actual: {tuple(logits.shape)}"
        )

    print(
        "\n✓ Feature map shape verified."
    )

    print(
        "✓ Pooled feature shape verified."
    )

    print(
        "✓ Classification output verified."
    )

    print(
        "\nDenseNet121 model is ready."
    )

    print(
        "=" * 70
    )


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    test_model()
    