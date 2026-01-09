"""
Input Validation Utilities
"""

import re
from pathlib import Path
from typing import Any, List, Optional


def validate_gesture_name(name: str) -> tuple:
    """Validate gesture name."""
    if not name or len(name.strip()) == 0:
        return False, "Name cannot be empty"
    if len(name) > 50:
        return False, "Name too long (max 50 chars)"
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', name):
        return False, "Invalid characters in name"
    return True, ""


def validate_path(path: str, must_exist: bool = False) -> tuple:
    """Validate file path."""
    if not path:
        return False, "Path cannot be empty"
    p = Path(path)
    if must_exist and not p.exists():
        return False, "Path does not exist"
    return True, ""


def validate_threshold(value: float, min_val: float = 0.0, max_val: float = 1.0) -> tuple:
    """Validate threshold value."""
    if not isinstance(value, (int, float)):
        return False, "Must be a number"
    if value < min_val or value > max_val:
        return False, f"Must be between {min_val} and {max_val}"
    return True, ""


def validate_resolution(width: int, height: int) -> tuple:
    """Validate resolution."""
    if width < 320 or height < 240:
        return False, "Resolution too small"
    if width > 3840 or height > 2160:
        return False, "Resolution too large"
    return True, ""
