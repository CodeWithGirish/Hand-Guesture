"""
Camera Manager - Camera Device Management
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import platform
import logging

logger = logging.getLogger(__name__)


class CameraInfo:
    """Information about a camera device."""
    
    def __init__(
        self,
        index: int,
        name: str = "",
        resolutions: List[Tuple[int, int]] = None
    ):
        self.index = index
        self.name = name or f"Camera {index}"
        self.resolutions = resolutions or [(640, 480)]
        self.available = False


class CameraManager:
    """Manages camera devices and capture settings."""
    
    COMMON_RESOLUTIONS = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080)
    ]
    
    def __init__(self):
        self.cameras: List[CameraInfo] = []
        self.active_camera: Optional[cv2.VideoCapture] = None
        self.active_index: int = -1
    
    def detect_cameras(self, max_cameras: int = 5) -> List[CameraInfo]:
        """Detect available camera devices."""
        self.cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Get camera info
                info = CameraInfo(index=i)
                info.available = True
                
                # Test resolutions
                info.resolutions = self._test_resolutions(cap)
                
                # Try to get camera name (platform specific)
                info.name = self._get_camera_name(i)
                
                self.cameras.append(info)
                cap.release()
            else:
                break
        
        logger.info(f"Detected {len(self.cameras)} cameras")
        return self.cameras
    
    def _test_resolutions(self, cap: cv2.VideoCapture) -> List[Tuple[int, int]]:
        """Test which resolutions are supported."""
        supported = []
        
        for w, h in self.COMMON_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_w, actual_h) == (w, h):
                supported.append((w, h))
        
        return supported if supported else [(640, 480)]
    
    def _get_camera_name(self, index: int) -> str:
        """Get camera name (platform specific)."""
        system = platform.system()
        
        if system == "Windows":
            try:
                # Could use DirectShow to get name
                pass
            except:
                pass
        elif system == "Linux":
            try:
                import subprocess
                result = subprocess.run(
                    ['v4l2-ctl', '--list-devices'],
                    capture_output=True, text=True
                )
                # Parse output
            except:
                pass
        elif system == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.run(
                    ['system_profiler', 'SPCameraDataType'],
                    capture_output=True, text=True
                )
                # Parse output
            except:
                pass
        
        return f"Camera {index}"
    
    def open_camera(
        self,
        index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ) -> bool:
        """Open a camera for capture."""
        # Close existing camera
        self.close_camera()
        
        self.active_camera = cv2.VideoCapture(index)
        
        if not self.active_camera.isOpened():
            logger.error(f"Failed to open camera {index}")
            return False
        
        # Set properties
        self.active_camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.active_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.active_camera.set(cv2.CAP_PROP_FPS, fps)
        
        # Set buffer size to 1 for lower latency
        self.active_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.active_index = index
        
        logger.info(f"Opened camera {index} at {resolution} @ {fps}fps")
        return True
    
    def close_camera(self):
        """Close the active camera."""
        if self.active_camera:
            self.active_camera.release()
            self.active_camera = None
            self.active_index = -1
            logger.info("Camera closed")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the active camera."""
        if not self.active_camera:
            return None
        
        ret, frame = self.active_camera.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            return None
        
        return frame
    
    def get_camera_properties(self) -> Dict:
        """Get current camera properties."""
        if not self.active_camera:
            return {}
        
        return {
            'width': int(self.active_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.active_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.active_camera.get(cv2.CAP_PROP_FPS),
            'brightness': self.active_camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.active_camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.active_camera.get(cv2.CAP_PROP_SATURATION),
            'exposure': self.active_camera.get(cv2.CAP_PROP_EXPOSURE)
        }
    
    def set_property(self, prop: str, value: float):
        """Set a camera property."""
        if not self.active_camera:
            return
        
        prop_map = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN
        }
        
        if prop in prop_map:
            self.active_camera.set(prop_map[prop], value)
    
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self.active_camera is not None and self.active_camera.isOpened()
