"""
Camera Widget - Live Camera Feed Display
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CameraThread(QThread):
    """Background thread for camera capture."""
    
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, device_index: int = 0, resolution: tuple = (640, 480)):
        super().__init__()
        self.device_index = device_index
        self.resolution = resolution
        self.running = False
        self.paused = False
        self.cap = None
    
    def run(self):
        """Main capture loop."""
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            self.error.emit(f"Failed to open camera {self.device_index}")
            return
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.running = True
        
        while self.running:
            if self.paused:
                self.msleep(50)
                continue
            
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.error.emit("Failed to read frame")
                break
            
            self.msleep(33)  # ~30 FPS
        
        self.cap.release()
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
        self.wait()
    
    def pause(self):
        """Pause capturing."""
        self.paused = True
    
    def resume(self):
        """Resume capturing."""
        self.paused = False


class CameraWidget(QWidget):
    """Widget for displaying live camera feed."""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.camera_thread = None
        self.is_running = False
        self.current_frame = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #34495e;
                border-radius: 8px;
            }
        """)
        self.display_label.setText("Camera Off")
        
        layout.addWidget(self.display_label)
    
    def start(self, device_index: int = None):
        """Start camera capture."""
        if self.is_running:
            self.stop()
        
        if device_index is None:
            device_index = self.settings.camera.device_index
        
        resolution = self.settings.camera.resolution
        
        self.camera_thread = CameraThread(device_index, resolution)
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.error.connect(self._on_error)
        self.camera_thread.start()
        
        self.is_running = True
        logger.info(f"Camera started: device {device_index}")
    
    def stop(self):
        """Stop camera capture."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.is_running = False
        self.display_label.setText("Camera Off")
        logger.info("Camera stopped")
    
    def pause(self):
        """Pause camera capture."""
        if self.camera_thread:
            self.camera_thread.pause()
    
    def resume(self):
        """Resume camera capture."""
        if self.camera_thread:
            self.camera_thread.resume()
    
    def _on_frame(self, frame: np.ndarray):
        """Handle new frame from camera."""
        self.current_frame = frame.copy()
        self.display_frame(frame)
        self.frame_ready.emit(frame)
    
    def _on_error(self, error: str):
        """Handle camera error."""
        logger.error(f"Camera error: {error}")
        self.display_label.setText(f"Error: {error}")
        self.is_running = False
    
    def display_frame(self, frame: np.ndarray):
        """Display a frame on the widget."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Scale to fit label
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.display_label.setPixmap(scaled_pixmap)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        return self.current_frame
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture current frame as image."""
        return self.current_frame.copy() if self.current_frame is not None else None
