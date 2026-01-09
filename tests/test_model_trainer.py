"""
Tests for Model Trainer
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
from pathlib import Path
from core.model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_path = Path(cls.temp_dir) / "test_dataset"
        cls.model_path = Path(cls.temp_dir) / "models"
        cls.dataset_path.mkdir(parents=True)
        cls.model_path.mkdir(parents=True)
        
        # Create mock dataset structure
        for gesture in ['thumbs_up', 'thumbs_down', 'peace']:
            gesture_dir = cls.dataset_path / gesture
            gesture_dir.mkdir()
            # Create mock images (10 per gesture for testing)
            for i in range(10):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(gesture_dir / f"img_{i}.jpg"), img)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        self.trainer = ModelTrainer(
            dataset_path=str(self.dataset_path),
            model_save_path=str(self.model_path)
        )
    
    def test_initialization(self):
        """Test trainer initializes correctly."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.dataset_path, str(self.dataset_path))
    
    def test_load_dataset(self):
        """Test dataset loading."""
        X, y, classes = self.trainer.load_dataset()
        self.assertEqual(len(classes), 3)
        self.assertIn('thumbs_up', classes)
        self.assertIn('thumbs_down', classes)
        self.assertIn('peace', classes)
    
    def test_create_model(self):
        """Test model creation."""
        model = self.trainer.create_model(num_classes=3)
        self.assertIsNotNone(model)
        # Check output shape
        self.assertEqual(model.output_shape[-1], 3)
    
    def test_prepare_data(self):
        """Test data preparation and splitting."""
        X, y, _ = self.trainer.load_dataset()
        X_train, X_val, y_train, y_val = self.trainer.prepare_data(X, y)
        
        # Check shapes
        self.assertEqual(len(X_train.shape), 4)  # (N, H, W, C)
        self.assertEqual(X_train.shape[1:], (224, 224, 3))
        
        # Check validation split
        total = len(X_train) + len(X_val)
        self.assertAlmostEqual(len(X_val) / total, 0.2, delta=0.1)
    
    def test_train_small_epochs(self):
        """Test training with minimal epochs."""
        history = self.trainer.train(epochs=1, batch_size=8)
        
        self.assertIsNotNone(history)
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train briefly
        self.trainer.train(epochs=1, batch_size=8)
        
        # Save model
        model_file = self.model_path / "test_model.keras"
        self.trainer.save_model(str(model_file))
        self.assertTrue(model_file.exists())
        
        # Load model
        loaded_trainer = ModelTrainer(
            dataset_path=str(self.dataset_path),
            model_save_path=str(self.model_path)
        )
        loaded_trainer.load_model(str(model_file))
        self.assertIsNotNone(loaded_trainer.model)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        self.trainer.train(epochs=1, batch_size=8)
        metrics = self.trainer.evaluate()
        
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_predict(self):
        """Test prediction on single image."""
        self.trainer.train(epochs=1, batch_size=8)
        
        # Create test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        prediction, confidence = self.trainer.predict(test_img)
        
        self.assertIsInstance(prediction, str)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_get_training_config(self):
        """Test training configuration retrieval."""
        config = self.trainer.get_training_config()
        
        self.assertIn('epochs', config)
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
        self.assertIn('model_type', config)


class TestModelTrainerEdgeCases(unittest.TestCase):
    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises appropriate error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(dataset_path=temp_dir)
            with self.assertRaises(ValueError):
                trainer.load_dataset()
    
    def test_invalid_model_path(self):
        """Test loading non-existent model."""
        trainer = ModelTrainer(dataset_path=".")
        with self.assertRaises(FileNotFoundError):
            trainer.load_model("/nonexistent/path/model.keras")


if __name__ == '__main__':
    unittest.main()
