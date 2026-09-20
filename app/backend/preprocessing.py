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
# LUNG MASK
# ============================================================
# Reproduces create_lung_mask / apply_lung_mask from the
# fixed preprocessing notebook (stage 12).


def create_lung_mask(
    gray: np.ndarray,
) -> np.ndarray:
    img = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    blur = cv2.GaussianBlur(
        img,
        (5, 5),
        0,
    )

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    h, w = mask.shape

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    candidates = []

    for component_id in range(1, num_labels):

        x, y, cw, ch, area = stats[component_id]
        cx, cy = centroids[component_id]

        if area < 0.03 * h * w:
            continue

        if (
            0.10 * w <= cx <= 0.90 * w
            and
            0.10 * h <= cy <= 0.92 * h
        ):
            candidates.append(
                (area, component_id)
            )

    candidates.sort(reverse=True)

    lung_mask = np.zeros_like(mask)

    for _, component_id in candidates[:2]:
        lung_mask[
            labels == component_id
        ] = 255

    # --------------------------------------------------------
    # Fallback central chest region
    # --------------------------------------------------------

    if np.count_nonzero(lung_mask) < 0.05 * h * w:

        yy, xx = np.ogrid[:h, :w]

        cx = w / 2
        cy = h * 0.52

        rx = w * 0.38
        ry = h * 0.40

        ellipse = (
            ((xx - cx) ** 2 / rx ** 2)
            +
            ((yy - cy) ** 2 / ry ** 2)
            <= 1
        )

        lung_mask[ellipse] = 255

    # Smooth mask edges
    lung_mask = cv2.GaussianBlur(
        lung_mask,
        (7, 7),
        0,
    )

    return lung_mask


def apply_lung_mask(
    gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = create_lung_mask(gray)

    mask_float = (
        mask.astype(np.float32) / 255.0
    )

    masked = (
        gray.astype(np.float32)
        * mask_float
    )

    masked = np.clip(
        masked,
        0,
        255,
    ).astype(np.uint8)

    return masked, mask


# ============================================================
# FINAL PREPROCESSING (thumbnail processing chain)
# ============================================================
# Order: CLAHE -> Lung mask -> Resize 224 -> RGB


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
        Output square size (224 legacy / 512 high-resolution).

    Returns
    -------
    final:
        (image_size, image_size, 3) uint8 RGB image.
    lung_mask:
        (H, W) uint8 lung mask from the ORIGINAL dimensions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = image

    clahe_enhanced = apply_clahe(gray)

    masked, mask = apply_lung_mask(clahe_enhanced)

    resized = cv2.resize(
        masked,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    final = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2RGB,
    )

    return final, mask


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