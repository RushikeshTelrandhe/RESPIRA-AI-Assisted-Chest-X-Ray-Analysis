"""
RESPIRA - Grad-CAM Explainability

Generates Grad-CAM heatmaps for the EfficientNet-B0 branch
using the same preprocessing and checkpoint as the training
pipeline (target layer: last Conv2d of backbone.features).

Produces the original image, the heatmap, and an overlay.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.backend import config
from app.backend.preprocessing import preprocess_pil

from src.models.efficientnet import create_efficientnet_b0


class GradCAMGenerator:
    """Grad-CAM explainability for the EfficientNet-B0 branch."""

    def __init__(
        self,
        device: torch.device = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.target_layer = None
        self.target_layer_name = None
        self.activations = None
        self.gradients = None
        self._handles = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> "GradCAMGenerator":
        if not config.EFFICIENTNET_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"EfficientNet checkpoint missing:\n"
                f"{config.EFFICIENTNET_CHECKPOINT}"
            )

        self.model = create_efficientnet_b0(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            dropout=0.3,
        )

        checkpoint = torch.load(
            config.EFFICIENTNET_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self._find_target_layer()
        self._register_hooks()

        return self

    def _find_target_layer(self) -> None:
        """Last Conv2d in backbone.features (spatial feature map)."""
        target_layer = None
        target_layer_name = None

        for name, module in reversed(
            list(self.model.backbone.features.named_modules())
        ):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_layer_name = f"backbone.features.{name}"
                break

        if target_layer is None:
            raise RuntimeError(
                "Could not locate a Conv2d target layer "
                "for Grad-CAM."
            )

        self.target_layer = target_layer
        self.target_layer_name = target_layer_name

    def _register_hooks(self) -> None:
        self._handles.append(
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    @torch.no_grad()
    def _preprocess(self, image: Image.Image):
        final_array, _ = preprocess_pil(image)
        pil_image = Image.fromarray(final_array)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device), pil_image

    @torch.enable_grad()
    def generate(
        self,
        image: Image.Image,
        target_class: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image:
            PIL image (RGB or grayscale).
        target_class:
            Disease class index to explain.

        Returns
        -------
        (original_rgb, heatmap, overlay)
        """
        if self.model is None:
            raise RuntimeError(
                "GradCAMGenerator not loaded. Call load() first."
            )

        tensor, original = self._preprocess(image)

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        logits = self.model(tensor)

        target_score = logits[0, target_class]
        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )
        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        # Base images for overlay
        original_np = np.array(original.convert("RGB"))
        heatmap_np = (np.uint8(255 * cam))
        overlay_np = self._blend(original_np, cam)

        return original_np, heatmap_np, overlay_np

    @staticmethod
    def _blend(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        colored = np.stack(
            [heatmap, heatmap, heatmap],
            axis=-1,
        )
        # Jet-style coloring
        colored = self_and_jet(colored)
        return np.uint8(
            alpha * colored + (1.0 - alpha) * image
        )


JET_MAP = (
    np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    * 255.0
)


def self_and_jet(grayscale_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale (H, W, 3) array with 0-1 values
    into a jet-colored array with float values 0-255.
    """
    intensity = grayscale_rgb[..., 0]
    scaled = intensity * (len(JET_MAP) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(JET_MAP) - 1)
    frac = scaled - low

    colors = (
        JET_MAP[low] * (1.0 - frac[..., None])
        + JET_MAP[high] * frac[..., None]
    )
    return colors


def encode_png(array: np.ndarray) -> str:
    """Encode an RGB uint8 array as a PNG data URL."""
    import base64
    import io

    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.fromarray(array).save(
        buffer,
        format="PNG",
    )
    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")
    return f"data:image/png;base64,{encoded}"