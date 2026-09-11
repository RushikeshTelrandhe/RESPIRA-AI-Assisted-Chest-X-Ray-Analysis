"""
RESPIRA - ViT Attention Visualization

Generates ViT-to-CNN cross-attention patch maps for
explainability, matching the methodology used in the
final analysis (STEP 8.8):

    1. vit_to_cnn cross-attention: (196, 49)
    2. Mean over CNN axis -> (196,) importance per ViT patch
    3. Reshape -> (14, 14) spatial heatmap
    4. Overlay on original image

Requires the full fusion pipeline (same preprocessing,
same checkpoint loading).
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image


PATCH_GRID = 14
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 196


def _normalize(
    arr: np.ndarray,
) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return arr


def _apply_jet(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel [0, 1] heatmap to a
    colored [0, 255] uint8 image using a simplified
    Jet colormap.
    """
    intensity = np.clip(heatmap, 0.0, 1.0)

    # Simple 3-stop Jet: blue -> cyan -> yellow -> red
    low = np.clip(intensity * 2.0, 0.0, 1.0)
    high = np.clip((intensity - 0.5) * 2.0, 0.0, 1.0)
    mid = 1.0 - np.abs(intensity - 0.5) * 2.0

    r = np.uint8(np.clip((1.0 - low) * 255.0, 0, 255))
    g = np.uint8(np.clip(mid * 255.0 + low * 0.0, 0, 255))
    b = np.uint8(np.clip(low * 255.0, 0, 255))

    # Fix bands
    r2 = np.where(intensity < 0.5, r, np.uint8(255.0))
    b2 = np.where(intensity > 0.5, np.uint8(0), b)
    g2 = np.where(intensity > 0.75, np.uint8(255.0 * (1.0 - (intensity - 0.75) * 4.0)), g)

    colored = np.stack([r2, g2, b2], axis=-1)
    return colored


def create_vit_attention_map(
    vit_to_cnn_attention: torch.Tensor,
) -> np.ndarray:
    """
    Convert a vit_to_cnn cross-attention matrix to a
    (14, 14) normalized spatial heatmap.

    Parameters
    ----------
    vit_to_cnn_attention:
        [196, 49] tensor (single sample, no batch dim).
        Average over CNN axis -> (196,) -> reshape (14, 14).

    Returns
    -------
    (14, 14) numpy array, values in [0, 1].
    """
    if vit_to_cnn_attention.ndim == 3:
        vit_to_cnn_attention = vit_to_cnn_attention[0]

    if vit_to_cnn_attention.numel() == 0:
        return np.zeros((PATCH_GRID, PATCH_GRID))

    token_attention = vit_to_cnn_attention.mean(dim=1)

    token_attention = token_attention[:NUM_PATCHES]

    token_attention = _normalize(token_attention.cpu().numpy())

    return token_attention.reshape(PATCH_GRID, PATCH_GRID)


def compute_vit_attention(
    vit_to_cnn_attention: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (original_placeholder, heatmap, overlay) arrays.

    This is a local-only function; no original image is needed
    to compute the heatmap, but the caller provides the
    original PIL image separately for the overlay.

    Returns
    -------
    (attention_map_14x14, heatmap_rgb, overlay_rgb)
    where the latter two are uint8 [0, 255].
    """
    attention_map = create_vit_attention_map(
        vit_to_cnn_attention
    )

    # Upscale to 224 for heatmap and overlay
    heatmap_224 = np.uint8(
        np.array(
            Image.fromarray(
                (attention_map * 255).astype(np.uint8),
                mode="L",
            ).resize(
                (224, 224),
                resample=Image.BILINEAR,
            )
        )
    )

    heatmap_rgb = np.stack(
        [heatmap_224, heatmap_224, heatmap_224],
        axis=-1,
    )

    return attention_map, heatmap_224, heatmap_rgb