"""
Model Architectures for Gesture Recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List
import numpy as np


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10
) -> Model:
    """Create a simple CNN for gesture classification."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Rescaling(1./255)(inputs)
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='simple_cnn')


def create_deep_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10
) -> Model:
    """Create a deeper CNN with batch normalization."""
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='deep_cnn')


def create_mobilenet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    freeze_base: bool = True
) -> Model:
    """Create a MobileNetV2-based model for transfer learning."""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    if freeze_base:
        base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for MobileNet
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='mobilenet_gesture')


def create_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    freeze_base: bool = True
) -> Model:
    """Create an EfficientNetB0-based model for transfer learning."""
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    if freeze_base:
        base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='efficientnet_gesture')


def create_landmark_mlp(
    input_dim: int = 63,  # 21 landmarks * 3 coordinates
    num_classes: int = 10
) -> Model:
    """Create an MLP for landmark-based classification."""
    inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='landmark_mlp')


def create_lstm_model(
    sequence_length: int = 30,
    num_landmarks: int = 21,
    num_classes: int = 10
) -> Model:
    """Create an LSTM model for temporal gesture sequences."""
    inputs = keras.Input(shape=(sequence_length, num_landmarks * 3))
    
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='gesture_lstm')


class ModelFactory:
    """Factory for creating gesture recognition models."""
    
    ARCHITECTURES = {
        'simple_cnn': create_simple_cnn,
        'deep_cnn': create_deep_cnn,
        'mobilenet': create_mobilenet_model,
        'efficientnet': create_efficientnet_model,
        'landmark_mlp': create_landmark_mlp,
        'lstm': create_lstm_model
    }
    
    @classmethod
    def create(
        cls,
        architecture: str,
        num_classes: int,
        input_shape: Tuple = None,
        **kwargs
    ) -> Model:
        """Create a model of the specified architecture."""
        if architecture not in cls.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        factory_func = cls.ARCHITECTURES[architecture]
        
        if input_shape:
            return factory_func(input_shape=input_shape, num_classes=num_classes, **kwargs)
        else:
            return factory_func(num_classes=num_classes, **kwargs)
    
    @classmethod
    def list_architectures(cls) -> List[str]:
        """List available architectures."""
        return list(cls.ARCHITECTURES.keys())
