"""
Main Application Window - Modern Layout with Theme Support
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QAction, QMessageBox
)
from PyQt5.QtCore import Qt
from gui.capture_tab import CaptureTab
from gui.dataset_tab import DatasetTab
from gui.training_tab import TrainingTab
from gui.mapping_tab import MappingTab
from gui.recognition_tab import RecognitionTab
from gui.settings_tab import SettingsTab
from gui.dashboard_tab import DashboardTab
from gui.styles import StyleSheet

class MainWindow(QMainWindow):
    """Main application window with modern tabbed interface."""

    def __init__(self, settings, database):
        super().__init__()
        self.settings = settings
        self.database = database

        self.setWindowTitle("GestureControl Pro")
        self.setMinimumSize(1280, 850)

        # Load initial theme
        current_theme = 'dark'
        if hasattr(self.settings, 'ui') and hasattr(self.settings.ui, 'theme'):
            current_theme = self.settings.ui.theme

        self.update_theme(current_theme)

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        # Create tabs
        self.dashboard_tab = DashboardTab(self.settings, self.database)
        self.capture_tab = CaptureTab(self.settings)
        self.dataset_tab = DatasetTab(self.settings, self.database)
        self.training_tab = TrainingTab(self.settings, self.database)
        self.mapping_tab = MappingTab(self.settings, self.database)
        self.recognition_tab = RecognitionTab(self.settings, self.database)
        self.settings_tab = SettingsTab(self.settings)

        # Add tabs
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.capture_tab, "Capture")
        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.mapping_tab, "Mapping")
        self.tabs.addTab(self.recognition_tab, "Recognition")
        self.tabs.addTab(self.settings_tab, "Settings")

        layout.addWidget(self.tabs)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Connect theme switch signals
        self.dashboard_tab.theme_toggled.connect(self.update_theme)
        self.settings_tab.theme_changed.connect(self.update_theme)

    def update_theme(self, theme_name):
        """Update application theme dynamically."""
        self.setStyleSheet(StyleSheet.get_stylesheet(theme_name))

        # Sync dashboard switch if changed from settings
        if hasattr(self, 'dashboard_tab'):
            self.dashboard_tab.mode_switch.blockSignals(True)
            self.dashboard_tab.mode_switch.setChecked(theme_name == 'dark')
            self.dashboard_tab.mode_switch.setText("Dark Mode" if theme_name == 'dark' else "Light Mode")
            self.dashboard_tab.mode_switch.blockSignals(False)

        # Sync settings combo if changed from dashboard
        if hasattr(self, 'settings_tab'):
            self.settings_tab.theme_combo.blockSignals(True)
            self.settings_tab.theme_combo.setCurrentText(theme_name.capitalize())
            self.settings_tab.theme_combo.blockSignals(False)

    def _on_tab_changed(self, index):
        """Handle tab change (pause/resume camera)."""
        if index != 1:
            self.capture_tab.pause_camera()
        if index != 5:
            self.recognition_tab.pause_camera()

    def _setup_menu(self):
        """Setup application menu bar."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        QMessageBox.about(self, "About", "GestureControl Pro v1.0.0")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Exit", "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.capture_tab.cleanup()
            self.recognition_tab.cleanup()
            self.settings.save()
            event.accept()
        else:
            event.ignore()