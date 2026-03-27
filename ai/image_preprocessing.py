"""Shared image preprocessing helpers for training/inference consistency."""

from __future__ import annotations

import cv2
import numpy as np


def center_crop_to_square(image_rgb: np.ndarray) -> np.ndarray:
    """Crop the largest centered square region from an RGB image."""
    height, width = image_rgb.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image_rgb[y0:y0 + side, x0:x0 + side]


def preprocess_rgb_image_like_training(
    image_rgb: np.ndarray,
    img_size: tuple[int, int] = (224, 224),
    use_center_crop: bool = True,
) -> np.ndarray:
    """
    Preprocess RGB image for model inference, matching training pipeline.
    
    This mimics how Cloudinary c_fill works (center-crop + resize).
    For local files: applies center-crop to square, then resizes.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image array (height, width, 3)
    img_size : tuple[int, int]
        Target size (height, width)
    use_center_crop : bool
        If True, center-crop to square before resizing (matches training).
        If False, direct resize (assumes image is already roughly square).
    
    Returns
    -------
    np.ndarray
        Normalized float32 image [0, 1]
    """
    if use_center_crop:
        image_rgb = center_crop_to_square(image_rgb)
    
    resized = cv2.resize(image_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
