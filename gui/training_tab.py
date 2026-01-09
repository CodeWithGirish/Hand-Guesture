"""
Training Tab - Model Training Interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QTabWidget,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import numpy as np
from datetime import datetime
import json

from core.model_trainer import GestureCNN, LandmarkClassifier
from core.dataset_manager import DatasetManager
from gui.widgets.chart_widget import TrainingChartWidget
from gui.styles import StyleSheet

class TrainingWorker(QThread):
    """Background worker for model training."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model, train_data, val_data, config):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

    def run(self):
        try:
            history = self.model.train(
                self.train_data,
                self.val_data,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                progress_callback=self._emit_progress
            )
            self.finished.emit(history)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, data):
        self.progress.emit(data)


class TrainingTab(QWidget):
    """Tab for training gesture recognition models."""

    training_complete = pyqtSignal(str, float)  # model_path, accuracy

    def __init__(self, settings, database, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.database = database
        self.dataset_manager = DatasetManager(settings)
        self.model = None
        self.training_worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)

        # --- Left Side: Configuration ---
        config_group = QGroupBox("Training Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(15)

        # Model type
        model_layout = QGridLayout()
        model_layout.setVerticalSpacing(10)

        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type = QComboBox()
        self.model_type.addItems(["CNN (Image-based)", "Landmark Classifier (SVM)", "Landmark Classifier (Random Forest)"])
        model_layout.addWidget(self.model_type, 0, 1)

        model_layout.addWidget(QLabel("Input Size:"), 1, 0)
        self.input_size = QComboBox()
        self.input_size.addItems(["128x128", "224x224", "256x256"])
        self.input_size.setCurrentText("224x224")
        model_layout.addWidget(self.input_size, 1, 1)

        config_layout.addLayout(model_layout)

        # Training parameters
        params_group = QGroupBox("Hyperparameters")
        params_layout = QGridLayout(params_group)
        params_layout.setVerticalSpacing(10)

        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 500)
        # Try to load from settings safely, else default
        self.epochs_spin.setValue(self.settings.training.epochs if hasattr(self.settings, 'training') else 50)
        params_layout.addWidget(self.epochs_spin, 0, 1)

        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(self.settings.training.batch_size if hasattr(self.settings, 'training') else 32)
        params_layout.addWidget(self.batch_spin, 1, 1)

        params_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(self.settings.training.learning_rate if hasattr(self.settings, 'training') else 0.001)
        params_layout.addWidget(self.lr_spin, 2, 1)

        params_layout.addWidget(QLabel("Validation Split:"), 3, 0)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.1, 0.4)
        self.val_split.setSingleStep(0.05)
        self.val_split.setValue(self.settings.training.val_split if hasattr(self.settings, 'training') else 0.2)
        params_layout.addWidget(self.val_split, 3, 1)

        config_layout.addWidget(params_group)

        # Options
        options_layout = QVBoxLayout()
        self.use_augmentation = QCheckBox("Use Data Augmentation")
        self.use_augmentation.setChecked(True)
        options_layout.addWidget(self.use_augmentation)

        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        options_layout.addWidget(self.use_gpu)

        self.early_stopping = QCheckBox("Early Stopping")
        self.early_stopping.setChecked(True)
        options_layout.addWidget(self.early_stopping)
        config_layout.addLayout(options_layout)

        # Dataset info
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        self.dataset_info = QLabel("No dataset loaded")
        self.dataset_info.setStyleSheet(f"color: {StyleSheet.get_color('text_dim')};")
        dataset_layout.addWidget(self.dataset_info)

        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        dataset_layout.addWidget(self.load_dataset_btn)

        config_layout.addWidget(dataset_group)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background-color: {StyleSheet.get_color('danger')};")
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)

        config_layout.addLayout(btn_layout)
        config_layout.addStretch()

        layout.addWidget(config_group, 1)

        # --- Right Side: Progress and Results ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.epoch_label = QLabel("Epoch: 0/0")
        self.epoch_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        progress_layout.addWidget(self.epoch_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)

        metrics_layout = QGridLayout()
        self.loss_label = QLabel("Loss: --")
        self.acc_label = QLabel("Accuracy: --")
        self.val_loss_label = QLabel("Val Loss: --")
        self.val_acc_label = QLabel("Val Accuracy: --")

        metrics_layout.addWidget(self.loss_label, 0, 0)
        metrics_layout.addWidget(self.acc_label, 0, 1)
        metrics_layout.addWidget(self.val_loss_label, 1, 0)
        metrics_layout.addWidget(self.val_acc_label, 1, 1)

        progress_layout.addLayout(metrics_layout)
        right_layout.addWidget(progress_group)

        # Charts
        charts_tabs = QTabWidget()
        self.loss_chart = TrainingChartWidget("Loss")
        charts_tabs.addTab(self.loss_chart, "Loss")

        self.acc_chart = TrainingChartWidget("Accuracy")
        charts_tabs.addTab(self.acc_chart, "Accuracy")
        right_layout.addWidget(charts_tabs, 2)

        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        # Results
        results_group = QGroupBox("Training History")
        results_layout = QVBoxLayout(results_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Model", "Accuracy", "Loss", "Epochs"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        results_layout.addWidget(self.history_table)

        right_layout.addWidget(results_group, 1)

        layout.addWidget(right_widget, 2)

        self._load_history()

    def _load_dataset(self):
        """Load dataset for training."""
        try:
            stats = self.dataset_manager.get_statistics()
            if stats['total_gestures'] == 0:
                self._log("No gestures found in dataset")
                return

            self.dataset_info.setText(
                f"Gestures: {stats['total_gestures']}\n"
                f"Images: {stats['total_images']}\n"
                f"Ready for training"
            )
            self._log(f"Dataset loaded: {stats['total_gestures']} gestures, {stats['total_images']} images")
        except Exception as e:
            self._log(f"Error loading dataset: {e}")

    def _start_training(self):
        """Start model training."""
        stats = self.dataset_manager.get_statistics()
        if stats['total_gestures'] < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 gestures to train")
            return

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        try:
            # Prepare data
            self._log("Preparing training data...")
            X, y, class_names = self.dataset_manager.prepare_training_data(
                input_size=self._get_input_size()
            )

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_split.value(), random_state=42
            )

            # Create model
            model_type = self.model_type.currentText()
            self._log(f"Initializing {model_type}...")

            if "CNN" in model_type:
                self.model = GestureCNN(
                    input_shape=(*self._get_input_size(), 3),
                    num_classes=len(class_names)
                )
                self.model.build()
                self.model.compile(learning_rate=self.lr_spin.value())
            else:
                self.model = LandmarkClassifier(num_classes=len(class_names))
                self.model.build("svm" if "SVM" in model_type else "rf")

            self.model.class_names = class_names

            # Start training
            self.progress_bar.setMaximum(self.epochs_spin.value())
            self.progress_bar.setValue(0)

            # Save settings
            if hasattr(self.settings, 'training'):
                self.settings.training.epochs = self.epochs_spin.value()
                self.settings.training.batch_size = self.batch_spin.value()
                self.settings.training.learning_rate = self.lr_spin.value()
                self.settings.training.val_split = self.val_split.value()
                self.settings.save()

            config = {
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_spin.value()
            }

            self.training_worker = TrainingWorker(
                self.model,
                (X_train, y_train),
                (X_val, y_val),
                config
            )
            self.training_worker.progress.connect(self._on_training_progress)
            self.training_worker.finished.connect(self._on_training_complete)
            self.training_worker.error.connect(self._on_training_error)
            self.training_worker.start()

            self._log("Training started...")

        except Exception as e:
            self._on_training_error(str(e))

    def _stop_training(self):
        """Stop training."""
        if self.training_worker:
            self.training_worker.terminate()
            self.training_worker.wait()
            self._log("Training stopped by user")
            self._reset_ui()

    def _on_training_progress(self, data):
        """Handle training progress update."""
        epoch = data.get('epoch', 0)
        self.epoch_label.setText(f"Epoch: {epoch}/{self.epochs_spin.value()}")
        self.progress_bar.setValue(epoch)

        loss = data.get('loss', 0)
        acc = data.get('accuracy', 0)
        val_loss = data.get('val_loss', 0)
        val_acc = data.get('val_accuracy', 0)

        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.acc_label.setText(f"Accuracy: {acc:.2%}")
        self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        self.val_acc_label.setText(f"Val Accuracy: {val_acc:.2%}")

        # Update charts
        self.loss_chart.add_point(epoch, loss, val_loss)
        self.acc_chart.add_point(epoch, acc, val_acc)

        self._log(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")

    def _on_training_complete(self, history):
        """Handle training completion."""
        self._log("Training completed successfully!")
        self.progress_bar.setValue(self.epochs_spin.value())

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/saved_models/gesture_model_{timestamp}"

        try:
            self.model.save(model_path)

            # Record in database
            final_acc = history['val_accuracy'][-1] if 'val_accuracy' in history and history['val_accuracy'] else 0
            final_loss = history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else 0

            self.database.add_training_record(
                model_name=self.model_type.currentText(),
                accuracy=final_acc,
                loss=final_loss,
                epochs=len(history.get('accuracy', [])),
                training_time=0,
                model_path=model_path
            )

            self._load_history()
            self._reset_ui()

            QMessageBox.information(
                self, "Training Complete",
                f"Model saved to {model_path}\nAccuracy: {final_acc:.2%}"
            )

            self.training_complete.emit(model_path, final_acc)

        except Exception as e:
            self._log(f"Error saving model: {e}")

    def _on_training_error(self, error):
        """Handle training error."""
        self._log(f"Error: {error}")
        QMessageBox.critical(self, "Training Error", str(error))
        self._reset_ui()

    def _reset_ui(self):
        """Reset UI after training."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _get_input_size(self):
        """Get input size tuple."""
        size_str = self.input_size.currentText()
        size = int(size_str.split("x")[0])
        return (size, size)

    def _log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _load_history(self):
        """Load training history."""
        try:
            records = self.database.get_training_history(limit=10)
            self.history_table.setRowCount(len(records))

            for i, record in enumerate(records):
                self.history_table.setItem(i, 0, QTableWidgetItem(str(record.created_at)[:16]))
                self.history_table.setItem(i, 1, QTableWidgetItem(record.model_name))
                self.history_table.setItem(i, 2, QTableWidgetItem(f"{record.accuracy:.1%}"))
                self.history_table.setItem(i, 3, QTableWidgetItem(f"{record.loss:.4f}"))
                self.history_table.setItem(i, 4, QTableWidgetItem(str(record.epochs)))
        except Exception as e:
            self._log(f"Could not load history: {e}")