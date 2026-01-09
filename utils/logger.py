"""
Logging Configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "logs/gesture_control.log",
    max_size: int = 5 * 1024 * 1024,
    backup_count: int = 3
):
    """Configure application logging."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info("Logging initialized")
