"""
Recognition Tab - Live Gesture Recognition Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QGroupBox, QFrame, QListWidget, QListWidgetItem,
    QCheckBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

import cv2
import numpy as np
from datetime import datetime
from collections import deque

from gui.widgets.camera_widget import CameraWidget
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.action_executor import ActionExecutor


class RecognitionTab(QWidget):
    """Tab for live gesture recognition and action execution."""
    
    gesture_recognized = pyqtSignal(str, float)  # gesture_name, confidence
    
    def __init__(self, settings, database, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.database = database
        
        self.hand_detector = HandDetector()
        self.recognizer = None
        self.action_executor = ActionExecutor()
        
        self.is_running = False
        self.recognition_history = deque(maxlen=50)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        
        # Left - Camera and recognition
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera
        camera_group = QGroupBox("Live Recognition")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_widget = CameraWidget(self.settings)
        self.camera_widget.frame_ready.connect(self._process_frame)
        camera_layout.addWidget(self.camera_widget)
        
        left_layout.addWidget(camera_group)
        
        # Current gesture display
        gesture_group = QGroupBox("Detected Gesture")
        gesture_layout = QVBoxLayout(gesture_group)
        
        self.gesture_label = QLabel("--")
        self.gesture_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet("color: #3498db;")
        gesture_layout.addWidget(self.gesture_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Confidence: %p%")
        gesture_layout.addWidget(self.confidence_bar)
        
        self.action_label = QLabel("Action: --")
        self.action_label.setAlignment(Qt.AlignCenter)
        gesture_layout.addWidget(self.action_label)
        
        left_layout.addWidget(gesture_group)
        
        layout.addWidget(left_widget, stretch=2)
        
        # Right - Controls and history
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._load_model)
        model_layout.addWidget(self.model_combo)
        
        self.reload_btn = QPushButton("🔄 Reload Models")
        self.reload_btn.clicked.connect(self._refresh_models)
        model_layout.addWidget(self.reload_btn)
        
        self.model_info = QLabel("No model loaded")
        model_layout.addWidget(self.model_info)
        
        right_layout.addWidget(model_group)
        
        # Settings
        settings_group = QGroupBox("Recognition Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(85)
        self.threshold_slider.valueChanged.connect(self._update_threshold)
        settings_layout.addWidget(self.threshold_slider, 0, 1)
        self.threshold_label = QLabel("85%")
        settings_layout.addWidget(self.threshold_label, 0, 2)
        
        settings_layout.addWidget(QLabel("Cooldown (ms):"), 1, 0)
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(100, 2000)
        self.cooldown_spin.setValue(500)
        settings_layout.addWidget(self.cooldown_spin, 1, 1, 1, 2)
        
        self.show_landmarks = QCheckBox("Show Landmarks")
        self.show_landmarks.setChecked(True)
        settings_layout.addWidget(self.show_landmarks, 2, 0, 1, 3)
        
        self.execute_actions = QCheckBox("Execute Actions")
        self.execute_actions.setChecked(True)
        settings_layout.addWidget(self.execute_actions, 3, 0, 1, 3)
        
        right_layout.addWidget(settings_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶️ Start Recognition")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 15px;")
        self.start_btn.clicked.connect(self._toggle_recognition)
        controls_layout.addWidget(self.start_btn)
        
        right_layout.addLayout(controls_layout)
        
        # History
        history_group = QGroupBox("Recognition History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.clicked.connect(self.history_list.clear)
        history_layout.addWidget(self.clear_history_btn)
        
        right_layout.addWidget(history_group)
        
        # Stats
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_recognitions = QLabel("Total: 0")
        self.avg_confidence = QLabel("Avg Confidence: --%")
        self.fps_label = QLabel("FPS: --")
        
        stats_layout.addWidget(self.total_recognitions, 0, 0)
        stats_layout.addWidget(self.avg_confidence, 0, 1)
        stats_layout.addWidget(self.fps_label, 1, 0, 1, 2)
        
        right_layout.addWidget(stats_group)
        
        layout.addWidget(right_widget, stretch=1)
        
        self._refresh_models()
    
    def _refresh_models(self):
        """Refresh available models."""
        self.model_combo.clear()
        
        import os
        models_dir = "models/saved_models"
        if os.path.exists(models_dir):
            for name in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, name)):
                    self.model_combo.addItem(name)
    
    def _load_model(self, model_name):
        """Load selected model."""
        if not model_name:
            return
        
        try:
            model_path = f"models/saved_models/{model_name}"
            self.recognizer = GestureRecognizer(model_path)
            self.model_info.setText(
                f"Loaded: {model_name}\n"
                f"Classes: {len(self.recognizer.class_names)}"
            )
        except Exception as e:
            self.model_info.setText(f"Error: {str(e)[:50]}")
    
    def _toggle_recognition(self):
        """Start/stop recognition."""
        if not self.is_running:
            if not self.recognizer:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Warning", "Please load a model first")
                return
            
            self.is_running = True
            self.start_btn.setText("⏹️ Stop Recognition")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 15px;")
            self.camera_widget.start()
        else:
            self.is_running = False
            self.start_btn.setText("▶️ Start Recognition")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 15px;")
            self.camera_widget.stop()
    
    def _process_frame(self, frame):
        """Process camera frame."""
        if not self.is_running:
            return
        
        hands = self.hand_detector.detect(frame)
        
        if hands and self.recognizer:
            # Draw landmarks if enabled
            if self.show_landmarks.isChecked():
                frame = self.hand_detector.draw_landmarks(frame, hands)
            
            # Get prediction
            gesture, confidence = self.recognizer.predict(frame, hands[0])
            
            threshold = self.threshold_slider.value() / 100.0
            
            if confidence >= threshold:
                self._on_gesture_detected(gesture, confidence)
        else:
            self.gesture_label.setText("--")
            self.confidence_bar.setValue(0)
            self.action_label.setText("Action: --")
        
        self.camera_widget.display_frame(frame)
    
    def _on_gesture_detected(self, gesture, confidence):
        """Handle detected gesture."""
        self.gesture_label.setText(gesture)
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update UI color based on confidence
        if confidence >= 0.95:
            color = "#2ecc71"  # Green
        elif confidence >= 0.85:
            color = "#3498db"  # Blue
        else:
            color = "#f39c12"  # Orange
        
        self.gesture_label.setStyleSheet(f"color: {color};")
        
        # Execute action if enabled
        if self.execute_actions.isChecked():
            result = self.action_executor.execute_for_gesture(gesture)
            if result:
                self.action_label.setText(f"Action: {result.message}")
                
                # Log to database
                self.database.log_recognition(
                    gesture, confidence,
                    result.action_name, result.success
                )
        
        # Add to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.insertItem(
            0, f"[{timestamp}] {gesture} ({confidence:.0%})"
        )
        
        self.recognition_history.append((gesture, confidence))
        self._update_stats()
        
        self.gesture_recognized.emit(gesture, confidence)
    
    def _update_threshold(self, value):
        """Update threshold display."""
        self.threshold_label.setText(f"{value}%")
    
    def _update_stats(self):
        """Update session statistics."""
        self.total_recognitions.setText(f"Total: {len(self.recognition_history)}")
        
        if self.recognition_history:
            avg = sum(c for _, c in self.recognition_history) / len(self.recognition_history)
            self.avg_confidence.setText(f"Avg Confidence: {avg:.0%}")
    
    def start_recognition(self):
        """Start recognition (called externally)."""
        if not self.is_running:
            self._toggle_recognition()
    
    def pause_camera(self):
        """Pause camera."""
        self.camera_widget.pause()
    
    def cleanup(self):
        """Cleanup resources."""
        self.camera_widget.stop()
        self.hand_detector.cleanup()
