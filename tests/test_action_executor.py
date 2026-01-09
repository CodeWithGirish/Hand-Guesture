"""
Tests for Action Executor
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
from core.action_executor import ActionExecutor
from actions.system_actions import (
    BrightnessAction, VolumeAction, LockScreenAction,
    NotificationAction
)
from actions.app_actions import BrowserAction, MediaPlayerAction
from actions.custom_actions import ScriptAction, WebhookAction


class TestActionExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = ActionExecutor()
    
    def tearDown(self):
        self.executor.cleanup()
    
    def test_initialization(self):
        """Test executor initializes correctly."""
        self.assertIsNotNone(self.executor)
        self.assertIsInstance(self.executor.action_history, list)
    
    def test_register_action(self):
        """Test action registration."""
        mock_action = Mock()
        mock_action.name = "test_action"
        
        self.executor.register_action("test_gesture", mock_action)
        self.assertIn("test_gesture", self.executor.gesture_mappings)
    
    def test_unregister_action(self):
        """Test action unregistration."""
        mock_action = Mock()
        mock_action.name = "test_action"
        
        self.executor.register_action("test_gesture", mock_action)
        self.executor.unregister_action("test_gesture")
        
        self.assertNotIn("test_gesture", self.executor.gesture_mappings)
    
    def test_get_registered_actions(self):
        """Test retrieving registered actions."""
        mock_action1 = Mock()
        mock_action1.name = "action1"
        mock_action2 = Mock()
        mock_action2.name = "action2"
        
        self.executor.register_action("gesture1", mock_action1)
        self.executor.register_action("gesture2", mock_action2)
        
        actions = self.executor.get_registered_actions()
        self.assertEqual(len(actions), 2)


class TestActionExecution(unittest.TestCase):
    def setUp(self):
        self.executor = ActionExecutor()
    
    def tearDown(self):
        self.executor.cleanup()
    
    @patch('pyautogui.press')
    def test_execute_keyboard_action(self, mock_press):
        """Test keyboard action execution."""
        from actions.app_actions import KeyboardAction
        action = KeyboardAction(key='space')
        
        self.executor.register_action("thumbs_up", action)
        result = self.executor.execute("thumbs_up", confidence=0.9)
        
        self.assertTrue(result)
        mock_press.assert_called_once_with('space')
    
    @patch('subprocess.run')
    def test_execute_script_action(self, mock_run):
        """Test script action execution."""
        mock_run.return_value = Mock(returncode=0)
        action = ScriptAction(script="echo 'test'")
        
        self.executor.register_action("peace", action)
        result = self.executor.execute("peace", confidence=0.85)
        
        self.assertTrue(result)
    
    def test_execute_with_low_confidence(self):
        """Test that low confidence prevents execution."""
        mock_action = Mock()
        mock_action.execute = Mock(return_value=True)
        
        self.executor.register_action("gesture", mock_action)
        self.executor.confidence_threshold = 0.8
        
        result = self.executor.execute("gesture", confidence=0.5)
        
        self.assertFalse(result)
        mock_action.execute.assert_not_called()
    
    def test_execute_unknown_gesture(self):
        """Test execution of unregistered gesture."""
        result = self.executor.execute("unknown_gesture", confidence=0.9)
        self.assertFalse(result)
    
    def test_cooldown_prevention(self):
        """Test cooldown prevents rapid re-execution."""
        mock_action = Mock()
        mock_action.execute = Mock(return_value=True)
        
        self.executor.register_action("gesture", mock_action)
        self.executor.cooldown_ms = 1000
        
        # First execution should succeed
        result1 = self.executor.execute("gesture", confidence=0.9)
        self.assertTrue(result1)
        
        # Immediate second execution should be blocked
        result2 = self.executor.execute("gesture", confidence=0.9)
        self.assertFalse(result2)
    
    def test_action_history_tracking(self):
        """Test that action history is recorded."""
        mock_action = Mock()
        mock_action.name = "test_action"
        mock_action.execute = Mock(return_value=True)
        
        self.executor.register_action("gesture", mock_action)
        self.executor.execute("gesture", confidence=0.9)
        
        self.assertEqual(len(self.executor.action_history), 1)
        self.assertEqual(self.executor.action_history[0]['gesture'], "gesture")
        self.assertEqual(self.executor.action_history[0]['action'], "test_action")


class TestActionExecutorPersistence(unittest.TestCase):
    def test_save_mappings(self):
        """Test saving gesture-action mappings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mappings.json"
            executor = ActionExecutor(config_path=str(config_path))
            
            mock_action = Mock()
            mock_action.name = "volume_up"
            mock_action.to_dict = Mock(return_value={'type': 'volume', 'direction': 'up'})
            
            executor.register_action("thumbs_up", mock_action)
            executor.save_mappings()
            
            self.assertTrue(config_path.exists())
            
            with open(config_path) as f:
                data = json.load(f)
            self.assertIn("thumbs_up", data)
    
    def test_load_mappings(self):
        """Test loading gesture-action mappings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mappings.json"
            
            # Create mock config file
            mappings = {
                "thumbs_up": {"type": "volume", "direction": "up"},
                "thumbs_down": {"type": "volume", "direction": "down"}
            }
            with open(config_path, 'w') as f:
                json.dump(mappings, f)
            
            executor = ActionExecutor(config_path=str(config_path))
            executor.load_mappings()
            
            self.assertEqual(len(executor.gesture_mappings), 2)


class TestSystemActions(unittest.TestCase):
    @patch('subprocess.run')
    def test_brightness_action(self, mock_run):
        """Test brightness adjustment action."""
        mock_run.return_value = Mock(returncode=0)
        action = BrightnessAction(level=80)
        
        result = action.execute()
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_notification_action(self, mock_run):
        """Test notification action."""
        mock_run.return_value = Mock(returncode=0)
        action = NotificationAction(
            title="Test",
            message="Test notification"
        )
        
        result = action.execute()
        self.assertTrue(result)


class TestWebhookAction(unittest.TestCase):
    @patch('requests.post')
    def test_webhook_success(self, mock_post):
        """Test successful webhook execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        action = WebhookAction(
            url="https://example.com/webhook",
            payload={"gesture": "thumbs_up"}
        )
        
        result = action.execute()
        self.assertTrue(result)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_webhook_failure(self, mock_post):
        """Test webhook failure handling."""
        mock_post.side_effect = Exception("Connection error")
        
        action = WebhookAction(url="https://example.com/webhook")
        result = action.execute()
        
        self.assertFalse(result)


class TestActionExecutorIntegration(unittest.TestCase):
    def test_full_workflow(self):
        """Test complete workflow: register, execute, history."""
        executor = ActionExecutor()
        
        # Create mock actions
        with patch('pyautogui.hotkey') as mock_hotkey:
            from actions.app_actions import HotkeyAction
            action = HotkeyAction(keys=['ctrl', 'c'])
            
            # Register
            executor.register_action("fist", action)
            
            # Execute
            result = executor.execute("fist", confidence=0.95)
            self.assertTrue(result)
            
            # Verify history
            self.assertEqual(len(executor.action_history), 1)
            
            # Clear history
            executor.clear_history()
            self.assertEqual(len(executor.action_history), 0)


if __name__ == '__main__':
    unittest.main()
