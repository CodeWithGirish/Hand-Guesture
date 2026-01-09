"""
Database Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class GestureModel:
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    image_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ActionModel:
    id: Optional[int] = None
    name: str = ""
    action_type: str = ""
    parameters: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass  
class MappingModel:
    id: Optional[int] = None
    gesture_id: int = 0
    action_id: int = 0
    confidence_threshold: float = 0.85
    enabled: bool = True


@dataclass
class TrainingModel:
    id: Optional[int] = None
    model_name: str = ""
    accuracy: float = 0.0
    loss: float = 0.0
    epochs: int = 0
    training_time: int = 0
    model_path: str = ""
    config: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProfileModel:
    id: Optional[int] = None
    name: str = ""
    settings: Dict = field(default_factory=dict)
    active: bool = False
    created_at: datetime = field(default_factory=datetime.now)
