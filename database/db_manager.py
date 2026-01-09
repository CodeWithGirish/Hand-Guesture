"""
Database Management with SQLite
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Gesture:
    id: Optional[int]
    name: str
    description: str
    image_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class Action:
    id: Optional[int]
    name: str
    type: str
    parameters: Dict
    created_at: datetime


@dataclass
class GestureMapping:
    id: Optional[int]
    gesture_id: int
    action_id: int
    confidence_threshold: float
    enabled: bool


@dataclass
class TrainingRecord:
    id: Optional[int]
    model_name: str
    accuracy: float
    loss: float
    epochs: int
    training_time: int
    model_path: str
    created_at: datetime


class DatabaseManager:
    """SQLite database manager for GestureControl Pro."""
    
    DB_PATH = "database/gesture_control.db"
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or self.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def initialize(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Gestures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    image_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Actions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Gesture-Action mappings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gesture_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_id INTEGER NOT NULL,
                    action_id INTEGER NOT NULL,
                    confidence_threshold REAL DEFAULT 0.85,
                    enabled INTEGER DEFAULT 1,
                    FOREIGN KEY (gesture_id) REFERENCES gestures(id) ON DELETE CASCADE,
                    FOREIGN KEY (action_id) REFERENCES actions(id) ON DELETE CASCADE,
                    UNIQUE(gesture_id, action_id)
                )
            """)
            
            # Training history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    epochs INTEGER,
                    training_time INTEGER,
                    model_path TEXT,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    settings TEXT,
                    active INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Recognition logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT,
                    confidence REAL,
                    action_executed TEXT,
                    success INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Database initialized successfully")
    
    # Gesture operations
    def add_gesture(self, name: str, description: str = "") -> int:
        """Add a new gesture."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO gestures (name, description) VALUES (?, ?)",
                (name, description)
            )
            return cursor.lastrowid
    
    def get_gesture(self, gesture_id: int) -> Optional[Gesture]:
        """Get gesture by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures WHERE id = ?", (gesture_id,))
            row = cursor.fetchone()
            if row:
                return Gesture(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    image_count=row['image_count'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
        return None
    
    def get_all_gestures(self) -> List[Gesture]:
        """Get all gestures."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures ORDER BY name")
            return [
                Gesture(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    image_count=row['image_count'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in cursor.fetchall()
            ]
    
    def update_gesture_count(self, gesture_id: int, count: int):
        """Update gesture image count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE gestures SET image_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (count, gesture_id)
            )
    
    def delete_gesture(self, gesture_id: int):
        """Delete a gesture."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM gestures WHERE id = ?", (gesture_id,))
    
    # Action operations
    def add_action(self, name: str, action_type: str, parameters: Dict) -> int:
        """Add a new action."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO actions (name, type, parameters) VALUES (?, ?, ?)",
                (name, action_type, json.dumps(parameters))
            )
            return cursor.lastrowid
    
    def get_all_actions(self) -> List[Action]:
        """Get all actions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM actions ORDER BY type, name")
            return [
                Action(
                    id=row['id'],
                    name=row['name'],
                    type=row['type'],
                    parameters=json.loads(row['parameters'] or '{}'),
                    created_at=row['created_at']
                )
                for row in cursor.fetchall()
            ]
    
    # Mapping operations
    def map_gesture_to_action(
        self,
        gesture_id: int,
        action_id: int,
        threshold: float = 0.85
    ) -> int:
        """Create gesture-action mapping."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO gesture_actions 
                   (gesture_id, action_id, confidence_threshold, enabled)
                   VALUES (?, ?, ?, 1)""",
                (gesture_id, action_id, threshold)
            )
            return cursor.lastrowid
    
    def get_action_for_gesture(self, gesture_name: str) -> Optional[Dict]:
        """Get mapped action for a gesture."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.*, ga.confidence_threshold
                FROM actions a
                JOIN gesture_actions ga ON a.id = ga.action_id
                JOIN gestures g ON g.id = ga.gesture_id
                WHERE g.name = ? AND ga.enabled = 1
            """, (gesture_name,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'parameters': json.loads(row['parameters'] or '{}'),
                    'threshold': row['confidence_threshold']
                }
        return None
    
    # Training history
    def add_training_record(
        self,
        model_name: str,
        accuracy: float,
        loss: float,
        epochs: int,
        training_time: int,
        model_path: str,
        config: Dict = None
    ) -> int:
        """Record training session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO training_history 
                   (model_name, accuracy, loss, epochs, training_time, model_path, config)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model_name, accuracy, loss, epochs, training_time, model_path, json.dumps(config or {}))
            )
            return cursor.lastrowid
    
    def get_training_history(self, limit: int = 20) -> List[TrainingRecord]:
        """Get recent training history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM training_history ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [
                TrainingRecord(
                    id=row['id'],
                    model_name=row['model_name'],
                    accuracy=row['accuracy'],
                    loss=row['loss'],
                    epochs=row['epochs'],
                    training_time=row['training_time'],
                    model_path=row['model_path'],
                    created_at=row['created_at']
                )
                for row in cursor.fetchall()
            ]
    
    # Recognition logging
    def log_recognition(
        self,
        gesture_name: str,
        confidence: float,
        action_executed: str,
        success: bool
    ):
        """Log a recognition event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO recognition_logs 
                   (gesture_name, confidence, action_executed, success)
                   VALUES (?, ?, ?, ?)""",
                (gesture_name, confidence, action_executed, int(success))
            )
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total recognitions
            cursor.execute("SELECT COUNT(*) FROM recognition_logs")
            total = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("SELECT AVG(success) FROM recognition_logs")
            success_rate = cursor.fetchone()[0] or 0
            
            # Most used gestures
            cursor.execute("""
                SELECT gesture_name, COUNT(*) as count
                FROM recognition_logs
                GROUP BY gesture_name
                ORDER BY count DESC
                LIMIT 10
            """)
            top_gestures = [
                {'name': row['gesture_name'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
            
            return {
                'total_recognitions': total,
                'success_rate': success_rate,
                'top_gestures': top_gestures
            }
