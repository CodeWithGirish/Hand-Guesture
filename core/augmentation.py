"""
Data Augmentation for Gesture Images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Image augmentation for gesture datasets."""
    
    def __init__(self):
        self.transforms = [
            self.rotate,
            self.flip_horizontal,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_noise,
            self.blur,
            self.zoom,
            self.shift
        ]
    
    def augment_images(
        self,
        image_paths: List[str],
        variations: int = 3,
        output_dir: str = None
    ) -> List[str]:
        """
        Augment a list of images.
        
        Args:
            image_paths: List of image file paths
            variations: Number of variations per image
            output_dir: Output directory (default: same as input)
            
        Returns:
            List of new image paths
        """
        new_paths = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Determine output directory
            if output_dir:
                out_dir = Path(output_dir)
            else:
                out_dir = Path(img_path).parent
            
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate variations
            for i in range(variations):
                augmented = self._apply_random_transforms(img)
                
                # Save with unique name
                stem = Path(img_path).stem
                new_name = f"{stem}_aug_{i}.jpg"
                new_path = out_dir / new_name
                
                cv2.imwrite(str(new_path), augmented)
                new_paths.append(str(new_path))
        
        logger.info(f"Created {len(new_paths)} augmented images")
        return new_paths
    
    def _apply_random_transforms(
        self,
        image: np.ndarray,
        num_transforms: int = 3
    ) -> np.ndarray:
        """Apply random transforms to an image."""
        result = image.copy()
        
        # Select random transforms
        selected = random.sample(self.transforms, min(num_transforms, len(self.transforms)))
        
        for transform in selected:
            result = transform(result)
        
        return result
    
    def rotate(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by random angle."""
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust image brightness."""
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust image contrast."""
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise to image."""
        noise = np.random.normal(0, 10, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def blur(self, image: np.ndarray) -> np.ndarray:
        """Apply slight blur."""
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    def zoom(self, image: np.ndarray) -> np.ndarray:
        """Zoom in/out on image."""
        scale = random.uniform(0.9, 1.1)
        h, w = image.shape[:2]
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        if scale > 1:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            result = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            return result
    
    def shift(self, image: np.ndarray) -> np.ndarray:
        """Shift image randomly."""
        h, w = image.shape[:2]
        
        max_shift = 20
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return shifted
    
    def elastic_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation."""
        alpha = 20
        sigma = 5
        
        h, w = image.shape[:2]
        
        # Random displacement fields
        dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
        
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
