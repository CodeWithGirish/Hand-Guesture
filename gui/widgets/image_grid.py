"""
Image Grid Widget - Display and Select Multiple Images
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QFrame,
    QSizePolicy, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

import cv2
import os
from pathlib import Path
from typing import List, Optional


class ImageThumbnail(QFrame):
    """Individual image thumbnail with selection support."""
    
    clicked = pyqtSignal(str)
    double_clicked = pyqtSignal(str)
    
    def __init__(self, image_path: str, size: int = 100, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.thumb_size = size
        self.selected = False
        
        self.setFixedSize(size + 10, size + 30)
        self.setCursor(Qt.PointingHandCursor)
        
        self._setup_ui()
        self._load_image()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.thumb_size, self.thumb_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 1px solid #34495e;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.image_label)
        
        self.name_label = QLabel(Path(self.image_path).stem[:12])
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size: 9px; color: #bdc3c7;")
        layout.addWidget(self.name_label)
        
        self._update_style()
    
    def _load_image(self):
        """Load and display the image."""
        if not os.path.exists(self.image_path):
            self.image_label.setText("?")
            return
        
        # Load image
        img = cv2.imread(self.image_path)
        if img is None:
            self.image_label.setText("!")
            return
        
        # Convert and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Scale to fit
        scale = min(self.thumb_size / w, self.thumb_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Create QPixmap
        qimg = QImage(img.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.image_label.setPixmap(pixmap)
    
    def _update_style(self):
        """Update frame style based on selection."""
        if self.selected:
            self.setStyleSheet("""
                QFrame {
                    background-color: #2980b9;
                    border: 2px solid #3498db;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #2c3e50;
                    border: 1px solid #34495e;
                    border-radius: 6px;
                }
                QFrame:hover {
                    border: 1px solid #3498db;
                }
            """)
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self._update_style()
    
    def toggle_selection(self):
        """Toggle selection state."""
        self.selected = not self.selected
        self._update_style()
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click."""
        self.double_clicked.emit(self.image_path)


class ImageGridWidget(QWidget):
    """Grid widget for displaying multiple images with selection."""
    
    image_selected = pyqtSignal(str)
    selection_changed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnails: List[ImageThumbnail] = []
        self.multi_select = True
        self.thumbnail_size = 100
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        toolbar.addWidget(self.select_all_btn)
        
        self.deselect_btn = QPushButton("Deselect All")
        self.deselect_btn.clicked.connect(self.deselect_all)
        toolbar.addWidget(self.deselect_btn)
        
        toolbar.addStretch()
        
        self.count_label = QLabel("0 images")
        toolbar.addWidget(self.count_label)
        
        layout.addLayout(toolbar)
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        
        self.scroll_area.setWidget(self.grid_widget)
        layout.addWidget(self.scroll_area)
    
    def load_images(self, image_paths: List[str]):
        """Load images into the grid."""
        # Clear existing
        self.clear()
        
        # Calculate columns based on width
        cols = max(1, self.scroll_area.width() // (self.thumbnail_size + 20))
        
        for i, path in enumerate(image_paths):
            thumb = ImageThumbnail(path, self.thumbnail_size)
            thumb.clicked.connect(self._on_thumbnail_clicked)
            thumb.double_clicked.connect(self._on_thumbnail_double_clicked)
            
            row = i // cols
            col = i % cols
            
            self.grid_layout.addWidget(thumb, row, col)
            self.thumbnails.append(thumb)
        
        self.count_label.setText(f"{len(image_paths)} images")
    
    def clear(self):
        """Clear all thumbnails."""
        for thumb in self.thumbnails:
            thumb.deleteLater()
        self.thumbnails.clear()
        
        # Clear layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.count_label.setText("0 images")
    
    def _on_thumbnail_clicked(self, path: str):
        """Handle thumbnail click."""
        # Find thumbnail
        thumb = next((t for t in self.thumbnails if t.image_path == path), None)
        if not thumb:
            return
        
        if self.multi_select:
            thumb.toggle_selection()
        else:
            # Single select mode
            for t in self.thumbnails:
                t.set_selected(t.image_path == path)
        
        self.image_selected.emit(path)
        self.selection_changed.emit(self.get_selected_images())
    
    def _on_thumbnail_double_clicked(self, path: str):
        """Handle thumbnail double click."""
        # Could open image in larger view
        pass
    
    def get_selected_images(self) -> List[str]:
        """Get list of selected image paths."""
        return [t.image_path for t in self.thumbnails if t.selected]
    
    def select_all(self):
        """Select all images."""
        for thumb in self.thumbnails:
            thumb.set_selected(True)
        self.selection_changed.emit(self.get_selected_images())
    
    def deselect_all(self):
        """Deselect all images."""
        for thumb in self.thumbnails:
            thumb.set_selected(False)
        self.selection_changed.emit([])
    
    def set_thumbnail_size(self, size: int):
        """Set thumbnail size."""
        self.thumbnail_size = size
        # Reload if there are images
        paths = [t.image_path for t in self.thumbnails]
        if paths:
            self.load_images(paths)
    
    def resizeEvent(self, event):
        """Handle resize to reorganize grid."""
        super().resizeEvent(event)
        # Could reorganize grid here
