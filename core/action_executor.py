"""
Action Execution System
"""

import os
import sys
import time
import json
import subprocess
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from abc import ABC, abstractmethod

import pyautogui
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

logger = logging.getLogger(__name__)

# Safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class ActionType(Enum):
    """Action categories."""
    SYSTEM = "system"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    APPLICATION = "application"
    MEDIA = "media"
    CUSTOM = "custom"


@dataclass
class ActionResult:
    """Result of action execution."""
    success: bool
    action_name: str
    message: str
    execution_time: float


class BaseAction(ABC):
    """Abstract base class for all actions."""
    
    def __init__(self, name: str, action_type: ActionType, params: Dict = None):
        self.name = name
        self.action_type = action_type
        self.params = params or {}
        self.enabled = True
        self.cooldown_ms = 500
        self.last_executed = 0
    
    @abstractmethod
    def execute(self) -> ActionResult:
        """Execute the action."""
        pass
    
    def can_execute(self) -> bool:
        """Check if action can be executed (cooldown)."""
        now = time.time() * 1000
        return (now - self.last_executed) >= self.cooldown_ms


class VolumeAction(BaseAction):
    """Volume control actions."""
    
    def __init__(self, direction: str = "up", amount: int = 5):
        super().__init__(
            f"Volume {direction.capitalize()}",
            ActionType.SYSTEM,
            {'direction': direction, 'amount': amount}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            direction = self.params['direction']
            amount = self.params.get('amount', 5)
            
            if sys.platform == 'win32':
                for _ in range(amount):
                    if direction == 'up':
                        pyautogui.press('volumeup')
                    elif direction == 'down':
                        pyautogui.press('volumedown')
                    elif direction == 'mute':
                        pyautogui.press('volumemute')
                        break
            elif sys.platform == 'darwin':
                # macOS
                script = f"set volume output volume (output volume of (get volume settings) {'+ ' if direction == 'up' else '- '}{amount})"
                subprocess.run(['osascript', '-e', script])
            else:
                # Linux
                cmd = f"amixer -D pulse sset Master {amount}%{'+' if direction == 'up' else '-'}"
                subprocess.run(cmd.split())
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Volume {direction}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class MediaAction(BaseAction):
    """Media playback control."""
    
    ACTIONS = ['play_pause', 'next', 'previous', 'stop']
    
    def __init__(self, action: str = "play_pause"):
        super().__init__(
            f"Media {action.replace('_', ' ').title()}",
            ActionType.MEDIA,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            action = self.params['action']
            
            key_map = {
                'play_pause': 'playpause',
                'next': 'nexttrack',
                'previous': 'prevtrack',
                'stop': 'stop'
            }
            
            pyautogui.press(key_map.get(action, 'playpause'))
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Media {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class KeyboardAction(BaseAction):
    """Keyboard input action."""
    
    def __init__(self, keys: str = "", hotkey: bool = False):
        super().__init__(
            f"Keyboard: {keys}",
            ActionType.KEYBOARD,
            {'keys': keys, 'hotkey': hotkey}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            keys = self.params['keys']
            is_hotkey = self.params.get('hotkey', False)
            
            if is_hotkey:
                pyautogui.hotkey(*keys.split('+'))
            else:
                pyautogui.typewrite(keys, interval=0.05)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Typed: {keys}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class MouseAction(BaseAction):
    """Mouse control action."""
    
    def __init__(
        self,
        action: str = "click",
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left"
    ):
        super().__init__(
            f"Mouse {action}",
            ActionType.MOUSE,
            {'action': action, 'x': x, 'y': y, 'button': button}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            action = self.params['action']
            x = self.params.get('x')
            y = self.params.get('y')
            button = self.params.get('button', 'left')
            
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            
            if action == 'click':
                pyautogui.click(button=button)
            elif action == 'double_click':
                pyautogui.doubleClick(button=button)
            elif action == 'right_click':
                pyautogui.rightClick()
            elif action == 'scroll_up':
                pyautogui.scroll(3)
            elif action == 'scroll_down':
                pyautogui.scroll(-3)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Mouse {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ApplicationAction(BaseAction):
    """Application launch/control action."""
    
    def __init__(self, path: str = "", action: str = "launch"):
        super().__init__(
            f"App: {os.path.basename(path)}",
            ActionType.APPLICATION,
            {'path': path, 'action': action}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            path = self.params['path']
            action = self.params.get('action', 'launch')
            
            if action == 'launch':
                if sys.platform == 'win32':
                    os.startfile(path)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen([path])
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Launched {path}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ScreenshotAction(BaseAction):
    """Take screenshot action."""
    
    def __init__(self, save_path: str = "screenshots"):
        super().__init__("Screenshot", ActionType.SYSTEM, {'save_path': save_path})
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            save_path = self.params['save_path']
            os.makedirs(save_path, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"screenshot_{timestamp}.png")
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Saved: {filename}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ActionExecutor:
    """Manages and executes gesture-mapped actions."""
    
    def __init__(self):
        self.actions: Dict[str, BaseAction] = {}
        self.gesture_mappings: Dict[str, str] = {}  # gesture_name -> action_name
        self.execution_log: List[ActionResult] = []
        self.enabled = True
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register built-in actions."""
        defaults = [
            VolumeAction("up"),
            VolumeAction("down"),
            VolumeAction("mute"),
            MediaAction("play_pause"),
            MediaAction("next"),
            MediaAction("previous"),
            ScreenshotAction(),
            MouseAction("scroll_up"),
            MouseAction("scroll_down"),
        ]
        
        for action in defaults:
            self.register_action(action)
    
    def register_action(self, action: BaseAction):
        """Register an action."""
        self.actions[action.name] = action
        logger.debug(f"Registered action: {action.name}")
    
    def map_gesture(self, gesture_name: str, action_name: str):
        """Map a gesture to an action."""
        if action_name in self.actions:
            self.gesture_mappings[gesture_name] = action_name
            logger.info(f"Mapped gesture '{gesture_name}' to action '{action_name}'")
    
    def execute_for_gesture(self, gesture_name: str) -> Optional[ActionResult]:
        """Execute action mapped to gesture."""
        if not self.enabled:
            return None
        
        action_name = self.gesture_mappings.get(gesture_name)
        if not action_name:
            return None
        
        action = self.actions.get(action_name)
        if not action or not action.enabled:
            return None
        
        if not action.can_execute():
            return None
        
        result = action.execute()
        self.execution_log.append(result)
        
        if result.success:
            logger.info(f"Executed: {result.action_name} - {result.message}")
        else:
            logger.error(f"Failed: {result.action_name} - {result.message}")
        
        return result
    
    def get_available_actions(self) -> List[Dict]:
        """Get list of available actions."""
        return [
            {
                'name': a.name,
                'type': a.action_type.value,
                'enabled': a.enabled,
                'params': a.params
            }
            for a in self.actions.values()
        ]
    
    def save_mappings(self, filepath: str):
        """Save gesture-action mappings to file."""
        with open(filepath, 'w') as f:
            json.dump(self.gesture_mappings, f, indent=2)
    
    def load_mappings(self, filepath: str):
        """Load gesture-action mappings from file."""
        with open(filepath, 'r') as f:
            self.gesture_mappings = json.load(f)
