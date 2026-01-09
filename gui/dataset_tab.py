"""
Dataset Tab - Dataset Management Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QProgressBar, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QFileDialog, QSplitter, QMenu,
    QAction, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon

import os
import shutil
from pathlib import Path

from gui.widgets.image_grid import ImageGridWidget
from core.dataset_manager import DatasetManager
from core.augmentation import DataAugmenter


class DatasetTab(QWidget):
    """Tab for managing gesture datasets."""
    
    dataset_changed = pyqtSignal()
    
    def __init__(self, settings, database, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.database = database
        self.dataset_manager = DatasetManager(settings)
        self.augmenter = DataAugmenter()
        
        self._setup_ui()
        self._load_dataset()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.import_btn = QPushButton("📁 Import Dataset")
        self.import_btn.clicked.connect(self._import_dataset)
        toolbar.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("💾 Export Dataset")
        self.export_btn.clicked.connect(self._export_dataset)
        toolbar.addWidget(self.export_btn)
        
        toolbar.addStretch()
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self._load_dataset)
        toolbar.addWidget(self.refresh_btn)
        
        layout.addLayout(toolbar)
        
        # Main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left - Gesture tree
        tree_group = QGroupBox("Gestures")
        tree_layout = QVBoxLayout(tree_group)
        
        self.gesture_tree = QTreeWidget()
        self.gesture_tree.setHeaderLabels(["Name", "Count", "Status"])
        self.gesture_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gesture_tree.customContextMenuRequested.connect(self._show_context_menu)
        self.gesture_tree.itemClicked.connect(self._on_gesture_selected)
        tree_layout.addWidget(self.gesture_tree)
        
        # Tree controls
        tree_controls = QHBoxLayout()
        self.add_gesture_btn = QPushButton("+ Add Gesture")
        self.add_gesture_btn.clicked.connect(self._add_gesture)
        tree_controls.addWidget(self.add_gesture_btn)
        
        self.delete_gesture_btn = QPushButton("🗑 Delete")
        self.delete_gesture_btn.clicked.connect(self._delete_gesture)
        tree_controls.addWidget(self.delete_gesture_btn)
        tree_layout.addLayout(tree_controls)
        
        splitter.addWidget(tree_group)
        
        # Right - Image preview
        preview_group = QGroupBox("Images")
        preview_layout = QVBoxLayout(preview_group)
        
        self.image_grid = ImageGridWidget()
        self.image_grid.image_selected.connect(self._on_image_selected)
        preview_layout.addWidget(self.image_grid)
        
        # Image controls
        img_controls = QHBoxLayout()
        
        self.augment_btn = QPushButton("🔀 Augment")
        self.augment_btn.clicked.connect(self._augment_selected)
        img_controls.addWidget(self.augment_btn)
        
        self.delete_img_btn = QPushButton("🗑 Delete Selected")
        self.delete_img_btn.clicked.connect(self._delete_selected_images)
        img_controls.addWidget(self.delete_img_btn)
        
        img_controls.addStretch()
        
        self.img_count_label = QLabel("0 images")
        img_controls.addWidget(self.img_count_label)
        
        preview_layout.addLayout(img_controls)
        
        splitter.addWidget(preview_group)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        
        # Statistics
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_gestures_label = QLabel("Total Gestures: 0")
        self.total_images_label = QLabel("Total Images: 0")
        self.avg_images_label = QLabel("Avg per Gesture: 0")
        self.dataset_size_label = QLabel("Dataset Size: 0 MB")
        
        stats_layout.addWidget(self.total_gestures_label, 0, 0)
        stats_layout.addWidget(self.total_images_label, 0, 1)
        stats_layout.addWidget(self.avg_images_label, 0, 2)
        stats_layout.addWidget(self.dataset_size_label, 0, 3)
        
        layout.addWidget(stats_group)
    
    def _load_dataset(self):
        """Load dataset information."""
        self.gesture_tree.clear()
        
        stats = self.dataset_manager.get_statistics()
        
        for gesture_name, info in stats['gestures'].items():
            item = QTreeWidgetItem([
                gesture_name,
                str(info['count']),
                "✓" if info['count'] >= self.settings.dataset.min_images_per_gesture else "⚠"
            ])
            self.gesture_tree.addTopLevelItem(item)
        
        # Update statistics
        self.total_gestures_label.setText(f"Total Gestures: {stats['total_gestures']}")
        self.total_images_label.setText(f"Total Images: {stats['total_images']}")
        self.avg_images_label.setText(f"Avg per Gesture: {stats['avg_per_gesture']:.1f}")
        self.dataset_size_label.setText(f"Dataset Size: {stats['size_mb']:.2f} MB")
    
    def _on_gesture_selected(self, item):
        """Handle gesture selection."""
        gesture_name = item.text(0)
        images = self.dataset_manager.get_gesture_images(gesture_name)
        self.image_grid.load_images(images)
        self.img_count_label.setText(f"{len(images)} images")
    
    def _on_image_selected(self, image_path):
        """Handle image selection."""
        pass  # Could show larger preview
    
    def _show_context_menu(self, position):
        """Show context menu for gesture tree."""
        item = self.gesture_tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu()
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_gesture(item))
        menu.addAction(rename_action)
        
        augment_action = QAction("Augment All", self)
        augment_action.triggered.connect(lambda: self._augment_gesture(item.text(0)))
        menu.addAction(augment_action)
        
        menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_gesture_item(item))
        menu.addAction(delete_action)
        
        menu.exec_(self.gesture_tree.mapToGlobal(position))
    
    def _add_gesture(self):
        """Add a new gesture."""
        name, ok = QInputDialog.getText(self, "Add Gesture", "Gesture name:")
        if ok and name:
            self.dataset_manager.create_gesture(name)
            self._load_dataset()
    
    def _delete_gesture(self):
        """Delete selected gesture."""
        item = self.gesture_tree.currentItem()
        if item:
            self._delete_gesture_item(item)
    
    def _delete_gesture_item(self, item):
        """Delete gesture item."""
        name = item.text(0)
        reply = QMessageBox.question(
            self, "Delete Gesture",
            f"Delete gesture '{name}' and all its images?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.dataset_manager.delete_gesture(name)
            self._load_dataset()
            self.dataset_changed.emit()
    
    def _rename_gesture(self, item):
        """Rename gesture."""
        old_name = item.text(0)
        new_name, ok = QInputDialog.getText(
            self, "Rename Gesture",
            "New name:", text=old_name
        )
        if ok and new_name and new_name != old_name:
            self.dataset_manager.rename_gesture(old_name, new_name)
            self._load_dataset()
    
    def _augment_selected(self):
        """Augment selected images."""
        selected = self.image_grid.get_selected_images()
        if not selected:
            QMessageBox.warning(self, "Warning", "No images selected")
            return
        
        self.augmenter.augment_images(selected, variations=3)
        self._load_dataset()
    
    def _augment_gesture(self, gesture_name):
        """Augment all images for a gesture."""
        images = self.dataset_manager.get_gesture_images(gesture_name)
        self.augmenter.augment_images(images, variations=2)
        self._load_dataset()
    
    def _delete_selected_images(self):
        """Delete selected images."""
        selected = self.image_grid.get_selected_images()
        if not selected:
            return
        
        reply = QMessageBox.question(
            self, "Delete Images",
            f"Delete {len(selected)} selected images?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for path in selected:
                os.remove(path)
            self._load_dataset()
    
    def _import_dataset(self):
        """Import dataset from folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_manager.import_from_folder(path)
            self._load_dataset()
    
    def _export_dataset(self):
        """Export dataset to folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if path:
            self.dataset_manager.export_to_folder(path)
            QMessageBox.information(self, "Success", "Dataset exported successfully")
