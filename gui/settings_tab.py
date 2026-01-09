"""
Settings Tab - Application Settings Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QLineEdit, QSlider, QTabWidget,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal

class SettingsTab(QWidget):
    """Tab for configuring application settings."""

    settings_changed = pyqtSignal()
    theme_changed = pyqtSignal(str)  # Signal for theme switching

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # --- 1. Camera Settings ---
        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)

        camera_group = QGroupBox("Camera Configuration")
        camera_grid = QGridLayout(camera_group)

        camera_grid.addWidget(QLabel("Camera Device:"), 0, 0)
        self.camera_device = QSpinBox()
        self.camera_device.setRange(0, 10)
        camera_grid.addWidget(self.camera_device, 0, 1)

        camera_grid.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution = QComboBox()
        self.resolution.addItems([
            "640x480", "800x600", "1280x720", "1920x1080"
        ])
        camera_grid.addWidget(self.resolution, 1, 1)

        camera_grid.addWidget(QLabel("Frame Rate:"), 2, 0)
        self.fps = QSpinBox()
        self.fps.setRange(15, 60)
        self.fps.setValue(30)
        camera_grid.addWidget(self.fps, 2, 1)

        camera_layout.addWidget(camera_group)
        camera_layout.addStretch()
        tabs.addTab(camera_tab, "📷 Camera")

        # --- 2. Recognition Settings ---
        recognition_tab = QWidget()
        recognition_layout = QVBoxLayout(recognition_tab)

        recognition_group = QGroupBox("Recognition Parameters")
        recognition_grid = QGridLayout(recognition_group)

        recognition_grid.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.5, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setValue(0.85)
        recognition_grid.addWidget(self.confidence_threshold, 0, 1)

        recognition_grid.addWidget(QLabel("Detection Sensitivity:"), 1, 0)
        self.detection_sensitivity = QSlider(Qt.Horizontal)
        self.detection_sensitivity.setRange(1, 100)
        self.detection_sensitivity.setValue(70)
        recognition_grid.addWidget(self.detection_sensitivity, 1, 1)

        recognition_grid.addWidget(QLabel("Cooldown (ms):"), 2, 0)
        self.cooldown = QSpinBox()
        self.cooldown.setRange(100, 2000)
        self.cooldown.setValue(500)
        recognition_grid.addWidget(self.cooldown, 2, 1)

        recognition_grid.addWidget(QLabel("Gesture Hold Time (ms):"), 3, 0)
        self.hold_time = QSpinBox()
        self.hold_time.setRange(100, 1000)
        self.hold_time.setValue(300)
        recognition_grid.addWidget(self.hold_time, 3, 1)

        recognition_layout.addWidget(recognition_group)
        recognition_layout.addStretch()
        tabs.addTab(recognition_tab, "👁 Recognition")

        # --- 3. Model Settings ---
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        model_group = QGroupBox("Model Settings")
        model_grid = QGridLayout(model_group)

        model_grid.addWidget(QLabel("Default Model:"), 0, 0)
        self.default_model = QComboBox()
        self.default_model.addItems(["gesture_cnn", "landmark_svm", "landmark_rf"])
        model_grid.addWidget(self.default_model, 0, 1)

        self.auto_load_model = QCheckBox("Auto-load last model on startup")
        self.auto_load_model.setChecked(True)
        model_grid.addWidget(self.auto_load_model, 1, 0, 1, 2)

        self.use_gpu = QCheckBox("Use GPU acceleration (if available)")
        self.use_gpu.setChecked(True)
        model_grid.addWidget(self.use_gpu, 2, 0, 1, 2)

        model_layout.addWidget(model_group)
        model_layout.addStretch()
        tabs.addTab(model_tab, "🧠 Model")

        # --- 4. Dataset Settings ---
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)

        dataset_group = QGroupBox("Dataset Configuration")
        dataset_grid = QGridLayout(dataset_group)

        dataset_grid.addWidget(QLabel("Dataset Path:"), 0, 0)
        self.dataset_path = QLineEdit()
        self.dataset_path.setText("datasets")
        dataset_grid.addWidget(self.dataset_path, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_grid.addWidget(browse_btn, 0, 2)

        dataset_grid.addWidget(QLabel("Min Images per Gesture:"), 1, 0)
        self.min_images = QSpinBox()
        self.min_images.setRange(10, 500)
        self.min_images.setValue(50)
        dataset_grid.addWidget(self.min_images, 1, 1)

        self.auto_backup = QCheckBox("Auto-backup dataset")
        self.auto_backup.setChecked(True)
        dataset_grid.addWidget(self.auto_backup, 2, 0, 1, 2)

        dataset_layout.addWidget(dataset_group)
        dataset_layout.addStretch()
        tabs.addTab(dataset_tab, "📁 Dataset")

        # --- 5. User Interface ---
        ui_tab = QWidget()
        ui_layout = QVBoxLayout(ui_tab)

        ui_group = QGroupBox("User Interface")
        ui_grid = QGridLayout(ui_group)

        ui_grid.addWidget(QLabel("Theme:"), 0, 0)
        # Renamed from 'self.theme' to 'self.theme_combo' to match MainWindow expectations
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        ui_grid.addWidget(self.theme_combo, 0, 1)

        ui_grid.addWidget(QLabel("Language:"), 1, 0)
        self.language = QComboBox()
        self.language.addItems(["English", "Spanish", "French", "German", "Chinese"])
        ui_grid.addWidget(self.language, 1, 1)

        self.show_fps = QCheckBox("Show FPS counter")
        self.show_fps.setChecked(True)
        ui_grid.addWidget(self.show_fps, 2, 0, 1, 2)

        self.show_landmarks = QCheckBox("Show hand landmarks")
        self.show_landmarks.setChecked(True)
        ui_grid.addWidget(self.show_landmarks, 3, 0, 1, 2)

        self.show_notifications = QCheckBox("Show system notifications")
        self.show_notifications.setChecked(True)
        ui_grid.addWidget(self.show_notifications, 4, 0, 1, 2)

        ui_layout.addWidget(ui_group)
        ui_layout.addStretch()
        tabs.addTab(ui_tab, "⚙️ Interface")

        layout.addWidget(tabs)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_settings)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)
        btn_layout.addWidget(self.apply_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("ActionBtn")
        self.save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def _load_settings(self):
        """Load current settings to UI."""
        if hasattr(self.settings, 'camera'):
            self.camera_device.setValue(self.settings.camera.device_index)
            res = self.settings.camera.resolution
            self.resolution.setCurrentText(f"{res[0]}x{res[1]}")
            self.fps.setValue(self.settings.camera.fps)

        if hasattr(self.settings, 'recognition'):
            self.confidence_threshold.setValue(self.settings.recognition.confidence_threshold)
            self.detection_sensitivity.setValue(int(self.settings.recognition.detection_sensitivity * 100))
            self.cooldown.setValue(self.settings.recognition.cooldown_ms)
            self.hold_time.setValue(self.settings.recognition.gesture_hold_time_ms)

        if hasattr(self.settings, 'model'):
            self.auto_load_model.setChecked(self.settings.model.auto_load_last)
            self.use_gpu.setChecked(self.settings.model.use_gpu)

        if hasattr(self.settings, 'dataset'):
            self.dataset_path.setText(self.settings.dataset.base_path)
            self.min_images.setValue(self.settings.dataset.min_images_per_gesture)
            self.auto_backup.setChecked(self.settings.dataset.auto_backup)

        if hasattr(self.settings, 'ui'):
            # Correctly reference theme_combo
            self.theme_combo.setCurrentText(self.settings.ui.theme.capitalize())
            self.show_fps.setChecked(self.settings.ui.show_fps)
            self.show_landmarks.setChecked(self.settings.ui.show_landmarks)

    def _browse_dataset(self):
        """Browse for dataset folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path.setText(path)

    def _on_theme_changed(self, text):
        """Handle theme change immediately."""
        theme = text.lower()
        if hasattr(self.settings, 'ui'):
            self.settings.ui.theme = theme
        self.theme_changed.emit(theme)

    def _apply_settings(self):
        """Apply settings without saving."""
        # Camera
        self.settings.camera.device_index = self.camera_device.value()
        res_str = self.resolution.currentText()
        w, h = map(int, res_str.split('x'))
        self.settings.camera.resolution = (w, h)
        self.settings.camera.fps = self.fps.value()

        # Recognition
        self.settings.recognition.confidence_threshold = self.confidence_threshold.value()
        self.settings.recognition.detection_sensitivity = self.detection_sensitivity.value() / 100.0
        self.settings.recognition.cooldown_ms = self.cooldown.value()
        self.settings.recognition.gesture_hold_time_ms = self.hold_time.value()

        # Model
        self.settings.model.auto_load_last = self.auto_load_model.isChecked()
        self.settings.model.use_gpu = self.use_gpu.isChecked()

        # Dataset
        self.settings.dataset.base_path = self.dataset_path.text()
        self.settings.dataset.min_images_per_gesture = self.min_images.value()
        self.settings.dataset.auto_backup = self.auto_backup.isChecked()

        # UI
        new_theme = self.theme_combo.currentText().lower()
        if self.settings.ui.theme != new_theme:
            self.settings.ui.theme = new_theme
            self.theme_changed.emit(new_theme)

        self.settings.ui.show_fps = self.show_fps.isChecked()
        self.settings.ui.show_landmarks = self.show_landmarks.isChecked()

        self.settings_changed.emit()
        QMessageBox.information(self, "Settings", "Settings applied successfully")

    def _save_settings(self):
        """Save settings to file."""
        self._apply_settings()
        if self.settings.save():
            QMessageBox.information(self, "Settings", "Settings saved successfully")
        else:
            QMessageBox.warning(self, "Settings", "Failed to save settings")

    def _reset_settings(self):
        """Reset to default settings."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.settings.reset_to_defaults()
            self._load_settings()
            # Also reset theme signal
            current_theme = self.theme_combo.currentText().lower()
            self.theme_changed.emit(current_theme)
            QMessageBox.information(self, "Settings", "Settings reset to defaults")