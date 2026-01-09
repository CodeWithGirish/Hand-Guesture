"""
Gesture Recognizer - Real-time Gesture Classification
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from core.hand_detector import HandDetector, HandLandmarks
from core.model_trainer import GestureCNN, LandmarkClassifier

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """Real-time gesture recognition using trained models."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_type = None
        self.class_names = []
        self.input_size = (224, 224)
        self.confidence_threshold = 0.85
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.class_names = metadata.get('class_names', [])
            self.input_size = tuple(metadata.get('input_shape', [224, 224])[:2])
            self.model_type = metadata.get('model_type', 'cnn')
        
        # Load model
        model_file = path / "model.keras"
        if model_file.exists():
            self.model = GestureCNN(
                input_shape=(*self.input_size, 3),
                num_classes=len(self.class_names)
            )
            self.model.load(model_path)
            self.model_type = 'cnn'
        else:
            # Try loading sklearn model
            import joblib
            sklearn_path = path / "model.joblib"
            if sklearn_path.exists():
                self.model = LandmarkClassifier()
                self.model.model = joblib.load(sklearn_path)
                self.model.class_names = self.class_names
                self.model_type = 'landmark'
        
        logger.info(f"Loaded model from {model_path} ({self.model_type})")
    
    def predict(
        self,
        frame: np.ndarray,
        hand: HandLandmarks
    ) -> Tuple[str, float]:
        """
        Predict gesture from frame and hand landmarks.
        
        Args:
            frame: BGR image from camera
            hand: Detected hand landmarks
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if self.model is None:
            return ("unknown", 0.0)
        
        if self.model_type == 'cnn':
            return self._predict_cnn(frame, hand)
        else:
            return self._predict_landmarks(hand)
    
    def _predict_cnn(
        self,
        frame: np.ndarray,
        hand: HandLandmarks
    ) -> Tuple[str, float]:
        """Predict using CNN model on hand image."""
        # Extract hand region
        x, y, w, h = hand.bbox
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        hand_region = frame[y:y+h, x:x+w]
        
        if hand_region.size == 0:
            return ("unknown", 0.0)
        
        # Resize and preprocess
        hand_image = cv2.resize(hand_region, self.input_size)
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        
        # Predict
        gesture_name, confidence = self.model.predict(hand_image)
        
        return (gesture_name, confidence)
    
    def _predict_landmarks(self, hand: HandLandmarks) -> Tuple[str, float]:
        """Predict using landmark classifier."""
        # Normalize landmarks
        landmarks = hand.landmarks.copy()
        
        # Center on wrist
        landmarks -= landmarks[0]
        
        # Scale to unit size
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist
        
        # Flatten for prediction
        features = landmarks.flatten()
        
        # Predict
        gesture_name, confidence = self.model.predict(features)
        
        return (gesture_name, confidence)
    
    def predict_batch(
        self,
        frames: List[np.ndarray],
        hands: List[HandLandmarks]
    ) -> List[Tuple[str, float]]:
        """Predict gestures for multiple frames."""
        return [
            self.predict(frame, hand)
            for frame, hand in zip(frames, hands)
        ]
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def get_class_names(self) -> List[str]:
        """Get list of gesture class names."""
        return self.class_names.copy()
