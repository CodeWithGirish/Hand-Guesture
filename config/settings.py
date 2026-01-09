"""
Configuration and Settings Management
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class CameraSettings:
    device_index: int = 0
    resolution: tuple = (640, 480)
    fps: int = 30

@dataclass
class RecognitionSettings:
    confidence_threshold: float = 0.85
    detection_sensitivity: float = 0.7
    cooldown_ms: int = 500
    gesture_hold_time_ms: int = 300

@dataclass
class ModelSettings:
    default_model: str = "gesture_cnn"
    auto_load_last: bool = True
    use_gpu: bool = True

@dataclass
class DatasetSettings:
    base_path: str = "datasets"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    min_images_per_gesture: int = 50

@dataclass
class TrainingSettings:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    val_split: float = 0.2

@dataclass
class UISettings:
    theme: str = "dark"
    language: str = "en"
    show_fps: bool = True
    show_landmarks: bool = True

class Settings:
    """Application settings manager with persistence."""

    CONFIG_FILE = "config/app_settings.json"

    def __init__(self):
        self.camera = CameraSettings()
        self.recognition = RecognitionSettings()
        self.model = ModelSettings()
        self.dataset = DatasetSettings()
        self.training = TrainingSettings()
        self.ui = UISettings()
        self._config_path = Path(self.CONFIG_FILE)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            if '.' in key:
                section, name = key.split('.', 1)
                if hasattr(self, section):
                    obj = getattr(self, section)
                    if hasattr(obj, name):
                        return getattr(obj, name)
            return default
        except: return default

    def set(self, key: str, value: Any):
        try:
            if '.' in key:
                section, name = key.split('.', 1)
                if hasattr(self, section):
                    obj = getattr(self, section)
                    if hasattr(obj, name):
                        setattr(obj, name, value)
        except: pass

    def load(self) -> bool:
        if not self._config_path.exists():
            self.save()
            return False
        try:
            with open(self._config_path, 'r') as f:
                data = json.load(f)
            self._apply_dict(data)
            return True
        except: return False

    def save(self) -> bool:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                'camera': asdict(self.camera),
                'recognition': asdict(self.recognition),
                'model': asdict(self.model),
                'dataset': asdict(self.dataset),
                'training': asdict(self.training),
                'ui': asdict(self.ui)
            }
            with open(self._config_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except: return False

    def _apply_dict(self, data: Dict[str, Any]):
        sections = {
            'camera': self.camera, 'recognition': self.recognition,
            'model': self.model, 'dataset': self.dataset,
            'training': self.training, 'ui': self.ui
        }
        for name, obj in sections.items():
            if name in data:
                for k, v in data[name].items():
                    if hasattr(obj, k): setattr(obj, k, v)

    def reset_to_defaults(self):
        self.__init__()
        self.save()