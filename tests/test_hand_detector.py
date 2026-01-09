"""
Tests for Hand Detector
"""

import unittest
import numpy as np
from core.hand_detector import HandDetector


class TestHandDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HandDetector()
    
    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.max_hands, 2)
    
    def test_detect_empty_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.detector.detect(frame)
        self.assertEqual(len(hands), 0)
    
    def tearDown(self):
        self.detector.cleanup()


if __name__ == '__main__':
    unittest.main()
