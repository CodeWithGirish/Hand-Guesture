"""
Mapping Tab - Gesture to Action Mapping Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox,
    QDoubleSpinBox, QGroupBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QInputDialog, QDialog, QDialogButtonBox,
    QFormLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon

from core.action_executor import (
    ActionExecutor, VolumeAction, MediaAction, KeyboardAction,
    MouseAction, ApplicationAction, ScreenshotAction, ActionType
)


class ActionConfigDialog(QDialog):
    """Dialog for configuring an action."""
    
    def __init__(self, action_type, parent=None):
        super().__init__(parent)
        self.action_type = action_type
        self.result_action = None
        
        self.setWindowTitle(f"Configure {action_type} Action")
        self.setMinimumWidth(400)
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        if self.action_type == "Volume":
            self.direction = QComboBox()
            self.direction.addItems(["up", "down", "mute"])
            form.addRow("Direction:", self.direction)
            
            self.amount = QSpinBox()
            self.amount.setRange(1, 20)
            self.amount.setValue(5)
            form.addRow("Amount:", self.amount)
            
        elif self.action_type == "Media":
            self.action = QComboBox()
            self.action.addItems(["play_pause", "next", "previous", "stop"])
            form.addRow("Action:", self.action)
            
        elif self.action_type == "Keyboard":
            self.keys = QLineEdit()
            self.keys.setPlaceholderText("e.g., ctrl+c or hello")
            form.addRow("Keys:", self.keys)
            
            self.is_hotkey = QCheckBox("Is Hotkey")
            form.addRow("", self.is_hotkey)
            
        elif self.action_type == "Mouse":
            self.mouse_action = QComboBox()
            self.mouse_action.addItems([
                "click", "double_click", "right_click",
                "scroll_up", "scroll_down"
            ])
            form.addRow("Action:", self.mouse_action)
            
        elif self.action_type == "Application":
            self.app_path = QLineEdit()
            form.addRow("Path:", self.app_path)
            
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(self._browse_app)
            form.addRow("", browse_btn)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._create_action)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _browse_app(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Application")
        if path:
            self.app_path.setText(path)
    
    def _create_action(self):
        """Create action from dialog inputs."""
        try:
            if self.action_type == "Volume":
                self.result_action = VolumeAction(
                    self.direction.currentText(),
                    self.amount.value()
                )
            elif self.action_type == "Media":
                self.result_action = MediaAction(self.action.currentText())
            elif self.action_type == "Keyboard":
                self.result_action = KeyboardAction(
                    self.keys.text(),
                    self.is_hotkey.isChecked()
                )
            elif self.action_type == "Mouse":
                self.result_action = MouseAction(self.mouse_action.currentText())
            elif self.action_type == "Application":
                self.result_action = ApplicationAction(self.app_path.text())
            
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


class MappingTab(QWidget):
    """Tab for mapping gestures to actions."""
    
    mapping_changed = pyqtSignal()
    
    def __init__(self, settings, database, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.database = database
        self.action_executor = ActionExecutor()
        
        self._setup_ui()
        self._load_mappings()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        
        # Left - Gestures
        gestures_group = QGroupBox("Available Gestures")
        gestures_layout = QVBoxLayout(gestures_group)
        
        self.gesture_list = QListWidget()
        self.gesture_list.itemClicked.connect(self._on_gesture_selected)
        gestures_layout.addWidget(self.gesture_list)
        
        layout.addWidget(gestures_group)
        
        # Center - Mapping controls
        mapping_group = QGroupBox("Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        
        self.selected_gesture = QLabel("Select a gesture")
        self.selected_gesture.setStyleSheet("font-size: 16px; font-weight: bold;")
        mapping_layout.addWidget(self.selected_gesture, alignment=Qt.AlignCenter)
        
        arrow_label = QLabel("⬇️")
        arrow_label.setStyleSheet("font-size: 24px;")
        mapping_layout.addWidget(arrow_label, alignment=Qt.AlignCenter)
        
        # Action type selection
        action_type_layout = QHBoxLayout()
        action_type_layout.addWidget(QLabel("Action Type:"))
        self.action_type_combo = QComboBox()
        self.action_type_combo.addItems([
            "Volume", "Media", "Keyboard", "Mouse", "Application", "Screenshot"
        ])
        action_type_layout.addWidget(self.action_type_combo)
        mapping_layout.addLayout(action_type_layout)
        
        # Create action button
        self.create_action_btn = QPushButton("Configure Action...")
        self.create_action_btn.clicked.connect(self._configure_action)
        mapping_layout.addWidget(self.create_action_btn)
        
        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 1.0)
        self.threshold_spin.setValue(0.85)
        self.threshold_spin.setSingleStep(0.05)
        threshold_layout.addWidget(self.threshold_spin)
        mapping_layout.addLayout(threshold_layout)
        
        # Map button
        self.map_btn = QPushButton("🔗 Create Mapping")
        self.map_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.map_btn.clicked.connect(self._create_mapping)
        mapping_layout.addWidget(self.map_btn)
        
        mapping_layout.addStretch()
        layout.addWidget(mapping_group)
        
        # Right - Existing mappings
        mappings_group = QGroupBox("Active Mappings")
        mappings_layout = QVBoxLayout(mappings_group)
        
        self.mappings_table = QTableWidget()
        self.mappings_table.setColumnCount(4)
        self.mappings_table.setHorizontalHeaderLabels([
            "Gesture", "Action", "Threshold", "Enabled"
        ])
        self.mappings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mappings_layout.addWidget(self.mappings_table)
        
        # Mapping controls
        mapping_controls = QHBoxLayout()
        
        self.test_btn = QPushButton("🧪 Test")
        self.test_btn.clicked.connect(self._test_mapping)
        mapping_controls.addWidget(self.test_btn)
        
        self.delete_mapping_btn = QPushButton("🗑 Delete")
        self.delete_mapping_btn.clicked.connect(self._delete_mapping)
        mapping_controls.addWidget(self.delete_mapping_btn)
        
        mapping_controls.addStretch()
        
        self.save_btn = QPushButton("💾 Save Mappings")
        self.save_btn.clicked.connect(self._save_mappings)
        mapping_controls.addWidget(self.save_btn)
        
        mappings_layout.addLayout(mapping_controls)
        layout.addWidget(mappings_group)
    
    def _load_mappings(self):
        """Load existing mappings."""
        # Load gestures
        self.gesture_list.clear()
        gestures = self.database.get_all_gestures()
        for gesture in gestures:
            self.gesture_list.addItem(gesture.name)
        
        # Load mappings
        self._refresh_mappings_table()
    
    def _refresh_mappings_table(self):
        """Refresh mappings table."""
        mappings = list(self.action_executor.gesture_mappings.items())
        self.mappings_table.setRowCount(len(mappings))
        
        for i, (gesture, action_name) in enumerate(mappings):
            self.mappings_table.setItem(i, 0, QTableWidgetItem(gesture))
            self.mappings_table.setItem(i, 1, QTableWidgetItem(action_name))
            self.mappings_table.setItem(i, 2, QTableWidgetItem("0.85"))
            
            enabled_check = QCheckBox()
            enabled_check.setChecked(True)
            self.mappings_table.setCellWidget(i, 3, enabled_check)
    
    def _on_gesture_selected(self, item):
        """Handle gesture selection."""
        self.selected_gesture.setText(item.text())
    
    def _configure_action(self):
        """Open action configuration dialog."""
        action_type = self.action_type_combo.currentText()
        
        if action_type == "Screenshot":
            self.current_action = ScreenshotAction()
            QMessageBox.information(self, "Action Created", "Screenshot action created")
        else:
            dialog = ActionConfigDialog(action_type, self)
            if dialog.exec_() == QDialog.Accepted:
                self.current_action = dialog.result_action
                QMessageBox.information(
                    self, "Action Created",
                    f"Action created: {self.current_action.name}"
                )
    
    def _create_mapping(self):
        """Create gesture-action mapping."""
        gesture_name = self.selected_gesture.text()
        if gesture_name == "Select a gesture":
            QMessageBox.warning(self, "Warning", "Please select a gesture")
            return
        
        if not hasattr(self, 'current_action') or not self.current_action:
            QMessageBox.warning(self, "Warning", "Please configure an action first")
            return
        
        # Register action and create mapping
        self.action_executor.register_action(self.current_action)
        self.action_executor.map_gesture(gesture_name, self.current_action.name)
        
        self._refresh_mappings_table()
        self.mapping_changed.emit()
        
        QMessageBox.information(
            self, "Success",
            f"Mapped '{gesture_name}' to '{self.current_action.name}'"
        )
    
    def _test_mapping(self):
        """Test selected mapping."""
        row = self.mappings_table.currentRow()
        if row < 0:
            return
        
        gesture = self.mappings_table.item(row, 0).text()
        result = self.action_executor.execute_for_gesture(gesture)
        
        if result and result.success:
            QMessageBox.information(self, "Test Result", f"Success: {result.message}")
        else:
            QMessageBox.warning(self, "Test Result", "Action failed or not found")
    
    def _delete_mapping(self):
        """Delete selected mapping."""
        row = self.mappings_table.currentRow()
        if row < 0:
            return
        
        gesture = self.mappings_table.item(row, 0).text()
        if gesture in self.action_executor.gesture_mappings:
            del self.action_executor.gesture_mappings[gesture]
        
        self._refresh_mappings_table()
    
    def _save_mappings(self):
        """Save mappings to file."""
        self.action_executor.save_mappings("config/gesture_mappings.json")
        QMessageBox.information(self, "Success", "Mappings saved")
