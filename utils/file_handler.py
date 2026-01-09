"""
File Handling Utilities
"""

import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_delete(path: str) -> bool:
    """Safely delete file or directory."""
    try:
        p = Path(path)
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
        return True
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False


def load_json(path: str) -> Optional[Dict]:
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON load failed: {e}")
        return None


def save_json(path: str, data: Any, indent: int = 2) -> bool:
    """Save data to JSON file."""
    try:
        ensure_dir(str(Path(path).parent))
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"JSON save failed: {e}")
        return False


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    return Path(path).stat().st_size if Path(path).exists() else 0


def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    """List files in directory."""
    p = Path(directory)
    if not p.exists():
        return []
    
    files = []
    for f in p.iterdir():
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                files.append(str(f))
    return sorted(files)
