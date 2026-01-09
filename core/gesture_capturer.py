"""
Gesture Capturer - Image Capture for Training
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass
import logging

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from core.hand_detector import HandLandmarks

logger = logging.getLogger(__name__)


@dataclass
class CaptureSession:
    """Information about a capture session."""
    gesture_name: str
    target_count: int
    captured_count: int
    save_path: Path
    start_time: datetime


class GestureCapturer(QObject):
    """Manages gesture image capture for dataset creation."""
    
    capture_progress = pyqtSignal(int, int)  # current, total
    capture_complete = pyqtSignal(str, int)  # gesture_name, count
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.session: Optional[CaptureSession] = None
        self.capture_timer: Optional[QTimer] = None
        self.frame_buffer: List[np.ndarray] = []
        self.is_capturing = False
    
    def start_capture(
        self,
        gesture_name: str,
        target_count: int = 100,
        delay_ms: int = 100
    ):
        """Start a capture session."""
        # Create save directory
        save_path = Path(self.settings.dataset.base_path) / gesture_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get existing count
        existing = len(list(save_path.glob("*.jpg")))
        
        self.session = CaptureSession(
            gesture_name=gesture_name,
            target_count=target_count,
            captured_count=0,
            save_path=save_path,
            start_time=datetime.now()
        )
        
        self.is_capturing = True
        self.frame_buffer.clear()
        
        logger.info(f"Started capture session for '{gesture_name}' - target: {target_count}")
    
    def add_frame(self, frame: np.ndarray, hand: HandLandmarks):
        """Add a frame to the capture buffer."""
        if not self.is_capturing or not self.session:
            return
        
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
            return
        
        # Resize to standard size
        hand_region = cv2.resize(hand_region, (224, 224))
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.session.gesture_name}_{timestamp}.jpg"
        filepath = self.session.save_path / filename
        
        cv2.imwrite(str(filepath), hand_region)
        
        self.session.captured_count += 1
        self.capture_progress.emit(
            self.session.captured_count,
            self.session.target_count
        )
        
        # Check if complete
        if self.session.captured_count >= self.session.target_count:
            self.stop_capture()
    
    def stop_capture(self):
        """Stop the capture session."""
        if not self.session:
            return
        
        self.is_capturing = False
        
        gesture_name = self.session.gesture_name
        count = self.session.captured_count
        
        logger.info(f"Capture complete: {count} images for '{gesture_name}'")
        
        self.capture_complete.emit(gesture_name, count)
        self.session = None
    
    def clear(self):
        """Clear the current session without saving."""
        self.session = None
        self.is_capturing = False
        self.frame_buffer.clear()
    
    def get_progress(self) -> tuple:
        """Get current capture progress."""
        if not self.session:
            return (0, 0)
        return (self.session.captured_count, self.session.target_count)
