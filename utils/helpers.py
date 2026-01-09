"""
General Helper Functions
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Any, Callable, Optional
import functools


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def md5_hash(data: bytes) -> str:
    """Calculate MD5 hash."""
    return hashlib.md5(data).hexdigest()


def file_hash(filepath: str) -> str:
    """Calculate file MD5 hash."""
    with open(filepath, 'rb') as f:
        return md5_hash(f.read())


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
