"""
Dataset Manager - Manage Gesture Datasets
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages gesture image datasets for training."""
    
    def __init__(self, settings):
        self.settings = settings
        self.base_path = Path(settings.dataset.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_gestures': 0,
            'total_images': 0,
            'gestures': {},
            'size_mb': 0,
            'avg_per_gesture': 0
        }
        
        if not self.base_path.exists():
            return stats
        
        total_size = 0
        
        for gesture_dir in self.base_path.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            images = list(gesture_dir.glob("*.jpg")) + list(gesture_dir.glob("*.png"))
            count = len(images)
            
            # Calculate size
            dir_size = sum(f.stat().st_size for f in images)
            total_size += dir_size
            
            stats['gestures'][gesture_dir.name] = {
                'count': count,
                'size_mb': dir_size / (1024 * 1024),
                'path': str(gesture_dir)
            }
            stats['total_images'] += count
        
        stats['total_gestures'] = len(stats['gestures'])
        stats['size_mb'] = total_size / (1024 * 1024)
        
        if stats['total_gestures'] > 0:
            stats['avg_per_gesture'] = stats['total_images'] / stats['total_gestures']
        
        return stats
    
    def get_gesture_images(self, gesture_name: str) -> List[str]:
        """Get all image paths for a gesture."""
        gesture_path = self.base_path / gesture_name
        
        if not gesture_path.exists():
            return []
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend([str(p) for p in gesture_path.glob(ext)])
        
        return sorted(images)
    
    def create_gesture(self, name: str) -> Path:
        """Create a new gesture directory."""
        gesture_path = self.base_path / name
        gesture_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created gesture directory: {gesture_path}")
        return gesture_path
    
    def delete_gesture(self, name: str):
        """Delete a gesture and all its images."""
        gesture_path = self.base_path / name
        
        if gesture_path.exists():
            shutil.rmtree(gesture_path)
            logger.info(f"Deleted gesture: {name}")
    
    def rename_gesture(self, old_name: str, new_name: str):
        """Rename a gesture."""
        old_path = self.base_path / old_name
        new_path = self.base_path / new_name
        
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            logger.info(f"Renamed gesture: {old_name} -> {new_name}")
    
    def prepare_training_data(
        self,
        input_size: Tuple[int, int] = (224, 224),
        test_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare dataset for training.
        
        Returns:
            Tuple of (X, y, class_names) where:
            - X: numpy array of images
            - y: numpy array of labels
            - class_names: list of gesture names
        """
        X = []
        y = []
        class_names = []
        
        # Get all gestures
        for i, gesture_dir in enumerate(sorted(self.base_path.iterdir())):
            if not gesture_dir.is_dir():
                continue
            
            class_names.append(gesture_dir.name)
            
            # Load images
            for img_path in gesture_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize
                img = cv2.resize(img, input_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                X.append(img)
                y.append(i)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Prepared dataset: {len(X)} images, {len(class_names)} classes")
        
        return X, y, class_names
    
    def import_from_folder(self, source_path: str):
        """Import dataset from external folder."""
        source = Path(source_path)
        
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")
        
        for gesture_dir in source.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            dest_dir = self.base_path / gesture_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_path in gesture_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_path, dest_dir)
        
        logger.info(f"Imported dataset from {source_path}")
    
    def export_to_folder(self, dest_path: str):
        """Export dataset to external folder."""
        dest = Path(dest_path)
        dest.mkdir(parents=True, exist_ok=True)
        
        for gesture_dir in self.base_path.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            dest_gesture = dest / gesture_dir.name
            dest_gesture.mkdir(parents=True, exist_ok=True)
            
            for img_path in gesture_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_path, dest_gesture)
        
        logger.info(f"Exported dataset to {dest_path}")
    
    def cleanup_duplicates(self):
        """Remove duplicate images from dataset."""
        import hashlib
        
        for gesture_dir in self.base_path.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            hashes = {}
            duplicates = []
            
            for img_path in gesture_dir.glob("*.jpg"):
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in hashes:
                    duplicates.append(img_path)
                else:
                    hashes[file_hash] = img_path
            
            # Remove duplicates
            for dup in duplicates:
                dup.unlink()
            
            if duplicates:
                logger.info(f"Removed {len(duplicates)} duplicates from {gesture_dir.name}")
