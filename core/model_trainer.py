"""
Model Training Module
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, Callback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class TrainingProgressCallback(Callback):
    """Callback to report training progress."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        super().__init__()
        self.progress_callback = progress_callback
    
    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback:
            self.progress_callback({
                'epoch': epoch + 1,
                'loss': logs.get('loss', 0),
                'accuracy': logs.get('accuracy', 0),
                'val_loss': logs.get('val_loss', 0),
                'val_accuracy': logs.get('val_accuracy', 0)
            })


class GestureCNN:
    """CNN model for gesture classification from images."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = []
    
    def build(self) -> Model:
        """Build CNN architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation layers
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Rescaling
        x = layers.Rescaling(1./255)(x)
        
        # Convolutional blocks
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name='gesture_cnn')
        
        logger.info(f"Built GestureCNN with {self.model.count_params():,} parameters")
        
        return self.model
    
    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001
    ):
        """Compile the model."""
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List[Callback] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Train the model."""
        X_train, y_train = train_data
        
        # Default callbacks
        default_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            TrainingProgressCallback(progress_callback)
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            verbose=1
        )
        
        return {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history.get('val_loss', []),
            'val_accuracy': self.history.history.get('val_accuracy', [])
        }
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Evaluate model on test data."""
        X_test, y_test = test_data
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        # Metrics
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
    
    def save(self, path: str):
        """Save model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path / 'model.keras')
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat()
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and metadata."""
        path = Path(path)
        
        self.model = keras.models.load_model(path / 'model.keras')
        
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.input_shape = tuple(metadata['input_shape'])
        self.num_classes = metadata['num_classes']
        self.class_names = metadata['class_names']
        
        logger.info(f"Model loaded from {path}")
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict gesture from image."""
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        
        predictions = self.model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        gesture_name = self.class_names[class_idx] if self.class_names else str(class_idx)
        
        return gesture_name, float(confidence)


class LandmarkClassifier:
    """Classifier using MediaPipe landmarks with sklearn."""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.model = None
        self.class_names = []
    
    def build(self, model_type: str = 'svm'):
        """Build classifier."""
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train classifier."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict gesture from landmarks."""
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        proba = self.model.predict_proba(landmarks)
        class_idx = np.argmax(proba[0])
        confidence = proba[0][class_idx]
        
        gesture_name = self.class_names[class_idx] if self.class_names else str(class_idx)
        
        return gesture_name, float(confidence)
