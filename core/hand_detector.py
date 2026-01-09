"""
Hand Detection using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: np.ndarray  # Shape: (21, 3) - 21 points with x, y, z
    handedness: str  # 'Left' or 'Right'
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

class HandDetector:
    """MediaPipe-based hand detection and landmark extraction."""
    
    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=model_complexity
        )
        
        logger.info("HandDetector initialized with MediaPipe")
    
    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands and extract landmarks from frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of HandLandmarks objects for each detected hand
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        detected_hands = []
        
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Extract landmarks as numpy array
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ])
                
                # Calculate bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                
                padding = 20
                bbox = (
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    min(w, x_max + padding) - max(0, x_min - padding),
                    min(h, y_max + padding) - max(0, y_min - padding)
                )
                
                hand_data = HandLandmarks(
                    landmarks=landmarks,
                    handedness=handedness.classification[0].label,
                    confidence=handedness.classification[0].score,
                    bbox=bbox
                )
                
                detected_hands.append(hand_data)
        
        return detected_hands
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        hands: List[HandLandmarks],
        draw_bbox: bool = True,
        draw_connections: bool = True
    ) -> np.ndarray:
        """Draw hand landmarks and connections on frame."""
        output = frame.copy()
        h, w, _ = output.shape
        
        for hand in hands:
            # Convert normalized landmarks to pixel coordinates
            points = (hand.landmarks[:, :2] * [w, h]).astype(int)
            
            # Draw connections
            if draw_connections:
                connections = self.mp_hands.HAND_CONNECTIONS
                for start_idx, end_idx in connections:
                    cv2.line(
                        output,
                        tuple(points[start_idx]),
                        tuple(points[end_idx]),
                        (0, 255, 200),
                        2
                    )
            
            # Draw landmarks
            for i, point in enumerate(points):
                color = (255, 100, 100) if i in self.FINGER_TIPS else (100, 255, 100)
                cv2.circle(output, tuple(point), 5, color, -1)
                cv2.circle(output, tuple(point), 7, (255, 255, 255), 1)
            
            # Draw bounding box
            if draw_bbox:
                x, y, bw, bh = hand.bbox
                cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                
                # Label
                label = f"{hand.handedness} ({hand.confidence:.0%})"
                cv2.putText(
                    output, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        
        return output
    
    def get_normalized_landmarks(self, hand: HandLandmarks) -> np.ndarray:
        """
        Get landmarks normalized relative to hand bounding box.
        Useful for gesture classification.
        """
        landmarks = hand.landmarks.copy()
        
        # Center on wrist
        landmarks -= landmarks[self.WRIST]
        
        # Scale to unit size
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist
        
        return landmarks.flatten()
    
    def cleanup(self):
        """Release resources."""
        self.hands.close()
