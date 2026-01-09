"""
Chart Widgets for Training and Statistics Visualization
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont

from typing import List, Dict, Tuple
import numpy as np


class TrainingChartWidget(QWidget):
    """Widget for displaying training progress charts."""
    
    def __init__(self, title: str = "Chart", parent=None):
        super().__init__(parent)
        self.title = title
        self.train_data = []
        self.val_data = []
        self.epochs = []
        
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Colors
        self.train_color = QColor("#3498db")
        self.val_color = QColor("#e74c3c")
        self.grid_color = QColor("#34495e")
        self.text_color = QColor("#ecf0f1")
        self.bg_color = QColor("#2c3e50")
    
    def add_point(self, epoch: int, train_value: float, val_value: float = None):
        """Add a data point to the chart."""
        self.epochs.append(epoch)
        self.train_data.append(train_value)
        if val_value is not None:
            self.val_data.append(val_value)
        self.update()
    
    def clear(self):
        """Clear all data."""
        self.train_data = []
        self.val_data = []
        self.epochs = []
        self.update()
    
    def paintEvent(self, event):
        """Paint the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Margins
        margin = 50
        chart_width = self.width() - 2 * margin
        chart_height = self.height() - 2 * margin
        
        if chart_width < 50 or chart_height < 50:
            return
        
        # Draw title
        painter.setPen(QPen(self.text_color))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(margin, 25, self.title)
        
        # Draw axes
        painter.setPen(QPen(self.grid_color, 1))
        
        # X axis
        painter.drawLine(margin, self.height() - margin, 
                        self.width() - margin, self.height() - margin)
        
        # Y axis
        painter.drawLine(margin, margin, margin, self.height() - margin)
        
        # Draw grid lines
        painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))
        for i in range(1, 5):
            y = margin + (chart_height * i) // 4
            painter.drawLine(margin, y, self.width() - margin, y)
        
        if not self.train_data:
            # No data message
            painter.setPen(QPen(self.text_color))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No data yet"
            )
            return
        
        # Calculate scales
        all_values = self.train_data + self.val_data
        min_val = min(all_values) * 0.9
        max_val = max(all_values) * 1.1
        
        if max_val == min_val:
            max_val = min_val + 1
        
        def scale_x(i):
            if len(self.epochs) <= 1:
                return margin + chart_width // 2
            return margin + (chart_width * i) // (len(self.epochs) - 1)
        
        def scale_y(val):
            return self.height() - margin - int(
                chart_height * (val - min_val) / (max_val - min_val)
            )
        
        # Draw training line
        painter.setPen(QPen(self.train_color, 2))
        for i in range(1, len(self.train_data)):
            x1, y1 = scale_x(i - 1), scale_y(self.train_data[i - 1])
            x2, y2 = scale_x(i), scale_y(self.train_data[i])
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw validation line
        if self.val_data:
            painter.setPen(QPen(self.val_color, 2))
            for i in range(1, len(self.val_data)):
                x1, y1 = scale_x(i - 1), scale_y(self.val_data[i - 1])
                x2, y2 = scale_x(i), scale_y(self.val_data[i])
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw points
        painter.setPen(QPen(Qt.white, 1))
        
        painter.setBrush(QBrush(self.train_color))
        for i, val in enumerate(self.train_data):
            x, y = scale_x(i), scale_y(val)
            painter.drawEllipse(x - 3, y - 3, 6, 6)
        
        if self.val_data:
            painter.setBrush(QBrush(self.val_color))
            for i, val in enumerate(self.val_data):
                x, y = scale_x(i), scale_y(val)
                painter.drawEllipse(x - 3, y - 3, 6, 6)
        
        # Draw legend
        legend_y = 15
        legend_x = self.width() - 150
        
        painter.setPen(QPen(self.train_color, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(self.text_color))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Training")
        
        if self.val_data:
            legend_y += 15
            painter.setPen(QPen(self.val_color, 2))
            painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
            painter.setPen(QPen(self.text_color))
            painter.drawText(legend_x + 25, legend_y + 4, "Validation")
        
        # Draw axis labels
        painter.setPen(QPen(self.text_color))
        painter.setFont(QFont("Arial", 8))
        
        # Y axis labels
        for i in range(5):
            val = min_val + (max_val - min_val) * (4 - i) / 4
            y = margin + (chart_height * i) // 4
            painter.drawText(5, y + 4, f"{val:.3f}")
        
        # X axis labels (epochs)
        if self.epochs:
            step = max(1, len(self.epochs) // 5)
            for i in range(0, len(self.epochs), step):
                x = scale_x(i)
                painter.drawText(x - 10, self.height() - margin + 15, str(self.epochs[i]))


class StatisticsChartWidget(QWidget):
    """Widget for displaying bar chart statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []  # List of {'name': str, 'count': int}
        
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Colors
        self.bar_colors = [
            QColor("#3498db"), QColor("#2ecc71"), QColor("#e74c3c"),
            QColor("#9b59b6"), QColor("#f39c12"), QColor("#1abc9c"),
            QColor("#e67e22"), QColor("#16a085"), QColor("#2980b9"),
            QColor("#8e44ad")
        ]
        self.text_color = QColor("#ecf0f1")
        self.bg_color = QColor("#2c3e50")
        self.grid_color = QColor("#34495e")
    
    def update_data(self, data: List[Dict]):
        """Update chart data."""
        self.data = data
        self.update()
    
    def paintEvent(self, event):
        """Paint the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        if not self.data:
            painter.setPen(QPen(self.text_color))
            painter.drawText(self.rect(), Qt.AlignCenter, "No statistics available")
            return
        
        margin = 60
        chart_width = self.width() - 2 * margin
        chart_height = self.height() - 2 * margin
        
        if chart_width < 50 or chart_height < 50:
            return
        
        # Calculate max value
        max_count = max(item['count'] for item in self.data)
        if max_count == 0:
            max_count = 1
        
        # Calculate bar dimensions
        num_bars = len(self.data)
        bar_width = min(40, chart_width // (num_bars * 2))
        spacing = (chart_width - bar_width * num_bars) // (num_bars + 1)
        
        # Draw grid
        painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))
        for i in range(1, 5):
            y = self.height() - margin - (chart_height * i) // 4
            painter.drawLine(margin, y, self.width() - margin, y)
        
        # Draw bars
        for i, item in enumerate(self.data):
            x = margin + spacing + i * (bar_width + spacing)
            bar_height = int(chart_height * item['count'] / max_count)
            y = self.height() - margin - bar_height
            
            # Draw bar
            color = self.bar_colors[i % len(self.bar_colors)]
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            painter.drawRect(x, y, bar_width, bar_height)
            
            # Draw label
            painter.setPen(QPen(self.text_color))
            painter.setFont(QFont("Arial", 8))
            
            # Truncate name if too long
            name = item['name'][:8] + ".." if len(item['name']) > 10 else item['name']
            
            painter.save()
            painter.translate(x + bar_width // 2, self.height() - margin + 5)
            painter.rotate(45)
            painter.drawText(0, 10, name)
            painter.restore()
            
            # Draw count on top
            painter.drawText(
                x, y - 5,
                bar_width, 15,
                Qt.AlignCenter,
                str(item['count'])
            )
        
        # Draw axes
        painter.setPen(QPen(self.grid_color, 2))
        painter.drawLine(margin, margin, margin, self.height() - margin)
        painter.drawLine(margin, self.height() - margin, 
                        self.width() - margin, self.height() - margin)
