"""
Capture Tab - Gesture Image Capture Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QLineEdit,
    QGroupBox, QProgressBar, QListWidget, QListWidgetItem,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np
from datetime import datetime
import os

from gui.widgets.camera_widget import CameraWidget
from core.hand_detector import HandDetector
from core.gesture_capturer import GestureCapturer


class CaptureTab(QWidget):
    """Tab for capturing gesture images."""
    
    capture_complete = pyqtSignal(str, int)  # gesture_name, count
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.hand_detector = HandDetector()
        self.capturer = GestureCapturer(settings)
        self.is_capturing = False
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        
        # Left side - Camera
        camera_group = QGroupBox("Camera Preview")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_widget = CameraWidget(self.settings)
        self.camera_widget.frame_ready.connect(self._process_frame)
        camera_layout.addWidget(self.camera_widget)
        
        # Camera controls
        cam_controls = QHBoxLayout()
        self.cam_selector = QComboBox()
        self.cam_selector.addItems([f"Camera {i}" for i in range(3)])
        cam_controls.addWidget(QLabel("Camera:"))
        cam_controls.addWidget(self.cam_selector)
        
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.clicked.connect(self._toggle_camera)
        cam_controls.addWidget(self.start_cam_btn)
        
        camera_layout.addLayout(cam_controls)
        layout.addWidget(camera_group, stretch=2)
        
        # Right side - Controls
        controls_group = QGroupBox("Capture Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Gesture selection
        gesture_layout = QHBoxLayout()
        gesture_layout.addWidget(QLabel("Gesture Name:"))
        self.gesture_input = QLineEdit()
        self.gesture_input.setPlaceholderText("Enter gesture name...")
        gesture_layout.addWidget(self.gesture_input)
        controls_layout.addLayout(gesture_layout)
        
        # Existing gestures
        controls_layout.addWidget(QLabel("Existing Gestures:"))
        self.gesture_list = QListWidget()
        self.gesture_list.itemClicked.connect(self._select_gesture)
        controls_layout.addWidget(self.gesture_list)
        
        # Capture settings
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("Images to Capture:"), 0, 0)
        self.capture_count = QSpinBox()
        self.capture_count.setRange(10, 500)
        self.capture_count.setValue(100)
        settings_layout.addWidget(self.capture_count, 0, 1)
        
        settings_layout.addWidget(QLabel("Capture Delay (ms):"), 1, 0)
        self.capture_delay = QSpinBox()
        self.capture_delay.setRange(50, 1000)
        self.capture_delay.setValue(100)
        settings_layout.addWidget(self.capture_delay, 1, 1)
        
        controls_layout.addLayout(settings_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to capture")
        controls_layout.addWidget(self.status_label)
        
        # Capture buttons
        btn_layout = QHBoxLayout()
        
        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.setStyleSheet("background-color: #2ecc71; font-weight: bold;")
        self.capture_btn.clicked.connect(self._toggle_capture)
        btn_layout.addWidget(self.capture_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_captures)
        btn_layout.addWidget(self.clear_btn)
        
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group, stretch=1)
    
    def _connect_signals(self):
        """Connect signals."""
        self.capturer.capture_progress.connect(self._update_progress)
        self.capturer.capture_complete.connect(self._on_capture_complete)
    
    def _toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_widget.is_running:
            self.camera_widget.stop()
            self.start_cam_btn.setText("Start Camera")
        else:
            cam_index = self.cam_selector.currentIndex()
            self.camera_widget.start(cam_index)
            self.start_cam_btn.setText("Stop Camera")
    
    def _process_frame(self, frame):
        """Process camera frame for hand detection."""
        hands = self.hand_detector.detect(frame)
        
        if hands:
            annotated = self.hand_detector.draw_landmarks(frame, hands)
            self.camera_widget.display_frame(annotated)
            
            if self.is_capturing:
                self.capturer.add_frame(frame, hands[0])
    
    def _toggle_capture(self):
        """Start/stop capture session."""
        if not self.is_capturing:
            gesture_name = self.gesture_input.text().strip()
            if not gesture_name:
                QMessageBox.warning(self, "Error", "Please enter a gesture name")
                return
            
            self.is_capturing = True
            self.capture_btn.setText("Stop Capture")
            self.capture_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold;")
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(self.capture_count.value())
            self.progress_bar.setValue(0)
            
            self.capturer.start_capture(
                gesture_name,
                self.capture_count.value(),
                self.capture_delay.value()
            )
        else:
            self.is_capturing = False
            self.capturer.stop_capture()
            self.capture_btn.setText("Start Capture")
            self.capture_btn.setStyleSheet("background-color: #2ecc71; font-weight: bold;")
    
    def _update_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Captured: {current}/{total}")
    
    def _on_capture_complete(self, gesture_name, count):
        """Handle capture completion."""
        self.is_capturing = False
        self.capture_btn.setText("Start Capture")
        self.capture_btn.setStyleSheet("background-color: #2ecc71; font-weight: bold;")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Completed: {count} images for '{gesture_name}'")
        
        self.capture_complete.emit(gesture_name, count)
        self._refresh_gesture_list()
    
    def _select_gesture(self, item):
        """Select gesture from list."""
        self.gesture_input.setText(item.text().split(" (")[0])
    
    def _clear_captures(self):
        """Clear current captures."""
        self.capturer.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Cleared")
    
    def _refresh_gesture_list(self):
        """Refresh gesture list from dataset."""
        self.gesture_list.clear()
        # Load from dataset directory
        dataset_path = self.settings.dataset.base_path
        if os.path.exists(dataset_path):
            for name in os.listdir(dataset_path):
                path = os.path.join(dataset_path, name)
                if os.path.isdir(path):
                    count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                    self.gesture_list.addItem(f"{name} ({count} images)")
    
    def pause_camera(self):
        """Pause the camera."""
        self.camera_widget.pause()
    
    def cleanup(self):
        """Cleanup resources."""
        self.camera_widget.stop()
        self.hand_detector.cleanup()
