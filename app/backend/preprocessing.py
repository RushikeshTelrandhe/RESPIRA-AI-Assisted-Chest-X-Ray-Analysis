"""
RESPIRA - Preprocessing
Reproduces the exact dataset preprocessing used to train the models:
CLAHE -> Lung mask -> Resize 224x224 -> RGB.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from app.backend.config import (
    IMAGE_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)


# ============================================================
# CLAHE ENHANCEMENT
# ============================================================
# Parameters identical to preprocessing_pipeline_FIXED.ipynb:
#   clip limit  : 2.0
#   tile grid   : (8, 8)


def apply_clahe(
    gray: np.ndarray,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


# ============================================================
# LUNG MASK (BYPASS / UNMASKED)
# ============================================================
# Masking removed to strictly match the 512x512 unmasked dataset:
# Preserves complete chest radiograph anatomy (ribs, apex, costophrenic angles).


def create_lung_mask(
    gray: np.ndarray,
) -> np.ndarray:
    """Returns full unmasked mask (all 255) to preserve complete chest anatomy."""
    h, w = gray.shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)


def apply_lung_mask(
    gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unmasked bypass:
    Returns the unmasked CLAHE enhanced image and full-field mask.
    Prevents blacked-out lungs or morphological boundary artifacts.
    """
    h, w = gray.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    return gray, mask


# ============================================================
# FINAL PREPROCESSING (Identical to 512 Training Pipeline)
# ============================================================
# Order: Grayscale -> CLAHE (clip 2.0, tile 8x8) -> Resize -> RGB (No Masking)


def process_image_array(
    image: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    image:
        RGB (H, W, 3) or grayscale (H, W) array.
    image_size:
        Output square size (512 standard / 224 legacy).

    Returns
    -------
    final:
        (image_size, image_size, 3) uint8 RGB image (unmasked, CLAHE enhanced).
    lung_mask:
        (H, W) uint8 full unmasked mask (all 255s) of original dimensions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = image

    # 1. CLAHE enhancement (clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = apply_clahe(gray)

    # 2. Resize directly without lung masking (identical to 512 training pipeline)
    resized = cv2.resize(
        clahe_enhanced,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    # 3. 3-channel RGB conversion
    final = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2RGB,
    )

    h, w = gray.shape[:2]
    full_mask = np.full((h, w), 255, dtype=np.uint8)

    return final, full_mask


def preprocess_pil(
    pil_image: Image.Image,
    image_size: int = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a PIL image into the model input and the lung mask.
    """
    image = np.asarray(
        pil_image.convert("RGB"),
        dtype=np.uint8,
    )

    return process_image_array(
        image,
        image_size=image_size,
    )