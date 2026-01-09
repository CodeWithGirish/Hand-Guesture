"""
Setup script for GestureControl Pro
"""

from setuptools import setup, find_packages

setup(
    name="gesture-control-pro",
    version="1.0.0",
    description="Hand gesture recognition for system automation",
    author="GestureControl Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "mediapipe>=0.9.0",
        "opencv-python>=4.7.0",
        "PyQt5>=5.15.0",
        "pyautogui>=0.9.53",
        "pynput>=1.7.6",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
    ],
    entry_points={
        "console_scripts": [
            "gesture-control=main:main",
        ],
    },
)
