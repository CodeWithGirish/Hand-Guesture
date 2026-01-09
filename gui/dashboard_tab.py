"""
Dashboard Tab - Application Overview and Quick Actions
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QFrame, QCheckBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QCursor
from datetime import datetime
from gui.widgets.chart_widget import StatisticsChartWidget
from gui.styles import StyleSheet

class StatCard(QFrame):
    """Card displaying a statistic with professional styling."""

    def __init__(self, title, value, icon="", color_key="primary", parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard") # Link to StyleSheet
        self.setCursor(QCursor(Qt.PointingHandCursor))

        accent_color = StyleSheet.get_color(color_key)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QHBoxLayout()
        if icon:
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet("font-size: 24px; background: transparent;")
            header.addWidget(icon_lbl)
        header.addStretch()

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {accent_color}; background: transparent;")
        header.addWidget(self.val_lbl)
        layout.addLayout(header)

        # Title
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("font-size: 11px; font-weight: bold; opacity: 0.7; background: transparent;")
        layout.addWidget(title_lbl)

    def update_value(self, value):
        self.val_lbl.setText(str(value))

class DashboardTab(QWidget):
    """Dashboard tab with Mode Switch."""

    # Signal to update the main window theme
    theme_toggled = pyqtSignal(str)

    def __init__(self, settings, database, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.database = database

        self._setup_ui()
        self._load_data()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._load_data)
        self.refresh_timer.start(30000)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # --- Header ---
        header = QHBoxLayout()

        welcome_box = QVBoxLayout()
        lbl_welcome = QLabel("Dashboard")
        lbl_welcome.setStyleSheet("font-size: 28px; font-weight: bold;")
        lbl_sub = QLabel("System Overview")
        lbl_sub.setStyleSheet("font-size: 14px; opacity: 0.8;")
        welcome_box.addWidget(lbl_welcome)
        welcome_box.addWidget(lbl_sub)
        header.addLayout(welcome_box)

        header.addStretch()

        # --- MODE SWITCH BUTTON ---
        self.mode_switch = QCheckBox("Dark Mode")
        self.mode_switch.setCursor(Qt.PointingHandCursor)
        # Load initial state safely
        is_dark = True
        if hasattr(self.settings, 'ui') and hasattr(self.settings.ui, 'theme'):
            is_dark = self.settings.ui.theme == 'dark'

        self.mode_switch.setChecked(is_dark)
        self.mode_switch.setText("Dark Mode" if is_dark else "Light Mode")
        self.mode_switch.toggled.connect(self._on_mode_switch)
        header.addWidget(self.mode_switch)

        # Time
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-weight: bold; padding-left: 15px;")
        header.addWidget(self.time_label)

        layout.addLayout(header)

        # --- Stats ---
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        self.card_gestures = StatCard("Gestures", "0", "✋", "primary")
        stats_layout.addWidget(self.card_gestures)
        self.card_imgs = StatCard("Images", "0", "🖼️", "secondary")
        stats_layout.addWidget(self.card_imgs)
        self.card_recogs = StatCard("Recognitions", "0", "👁️", "accent")
        stats_layout.addWidget(self.card_recogs)
        self.card_acc = StatCard("Accuracy", "--", "🎯", "success")
        stats_layout.addWidget(self.card_acc)
        layout.addLayout(stats_layout)

        # --- Content ---
        content = QHBoxLayout()

        # Actions
        left_col = QVBoxLayout()
        grp_actions = QGroupBox("Quick Actions")
        act_layout = QVBoxLayout(grp_actions)
        act_layout.setSpacing(10)

        actions = [
            ("▶ Start Recognition", self._nav_recog, "success"),
            ("📷 Capture Gestures", self._nav_capture, "primary"),
            ("🧠 Train Model", self._nav_train, "accent"),
            ("🔗 Configure", self._nav_map, "warning")
        ]

        for text, func, color_key in actions:
            btn = QPushButton(text)
            btn.setObjectName("ActionBtn")
            btn.clicked.connect(func)
            # Add colored border via inline style
            color = StyleSheet.get_color(color_key)
            btn.setStyleSheet(f"border-left: 4px solid {color};")
            act_layout.addWidget(btn)

        left_col.addWidget(grp_actions)
        left_col.addStretch()
        content.addLayout(left_col, 1)

        # Chart
        right_col = QVBoxLayout()
        grp_chart = QGroupBox("Analytics")
        grp_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout = QVBoxLayout(grp_chart)
        self.chart = StatisticsChartWidget()
        chart_layout.addWidget(self.chart)
        right_col.addWidget(grp_chart)
        content.addLayout(right_col, 2)

        layout.addLayout(content)

        # --- Footer Status ---
        footer = QHBoxLayout()
        self.status_cam = QLabel("📷 Disconnected")
        self.status_cam.setStyleSheet(f"color: {StyleSheet.get_color('danger')}; font-weight: bold;")
        footer.addWidget(self.status_cam)

        self.status_model = QLabel("🧠 No Model")
        self.status_model.setStyleSheet(f"color: {StyleSheet.get_color('warning')}; font-weight: bold;")
        footer.addWidget(self.status_model)

        footer.addStretch()
        footer.addWidget(QLabel("v1.0.0 Pro"))

        frame_footer = QFrame()
        frame_footer.setObjectName("StatCard")
        frame_footer.setLayout(footer)
        frame_footer.setFixedHeight(50)
        layout.addWidget(frame_footer)

        self._update_time()
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)

    def _on_mode_switch(self, checked):
        """Handle theme toggle."""
        theme = 'dark' if checked else 'light'
        self.mode_switch.setText("Dark Mode" if checked else "Light Mode")
        self.settings.ui.theme = theme
        self.settings.save()
        self.theme_toggled.emit(theme)

    def _update_time(self):
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))

    def _load_data(self):
        try:
            gestures = self.database.get_all_gestures()
            self.card_gestures.update_value(len(gestures))
            self.card_imgs.update_value(sum(g.image_count for g in gestures))
            stats = self.database.get_recognition_stats()
            self.card_recogs.update_value(stats.get('total_recognitions', 0))
            if stats.get('success_rate', 0):
                self.card_acc.update_value(f"{stats['success_rate']:.0%}")
            self.chart.update_data(stats.get('top_gestures', []))
        except: pass

    # Navigation
    def _nav_recog(self): self._nav(5)
    def _nav_capture(self): self._nav(1)
    def _nav_train(self): self._nav(3)
    def _nav_map(self): self._nav(4)
    def _nav(self, i):
        p = self.parent()
        while p:
            if hasattr(p, 'tabs'): p.tabs.setCurrentIndex(i); break
            p = p.parent()