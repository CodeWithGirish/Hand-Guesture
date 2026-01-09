"""
Image Processing Utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range."""
    return image.astype(np.float32) / 255.0


def convert_color(image: np.ndarray, code: int = cv2.COLOR_BGR2RGB) -> np.ndarray:
    """Convert image color space."""
    return cv2.cvtColor(image, code)


def crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    """Crop region from image."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    return image[y1:y2, x1:x2]


def add_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """Add text to image."""
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return result


def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    """Save image to file."""
    try:
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(path, image)
        return True
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False
