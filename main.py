#!/usr/bin/env python3
"""
GestureControl Pro - Main Application Entry Point
A professional hand gesture recognition system for automating tasks.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow
from config.settings import Settings
from utils.logger import setup_logging
from database.db_manager import DatabaseManager

def main():
    """Initialize and run the GestureControl Pro application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting GestureControl Pro...")

    # Initialize settings
    settings = Settings()
    settings.load()

    # Initialize database
    db = DatabaseManager()
    db.initialize()

    # Enable high DPI scaling - Must be set BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("GestureControl Pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("GestureControl")

    # Create and show main window
    window = MainWindow(settings, db)
    window.show()

    logger.info("Application started successfully")

    # Run event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()