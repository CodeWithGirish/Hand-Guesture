"""
Application-specific Actions
"""

import os
import sys
import subprocess
import platform
import logging
from typing import Optional

from core.action_executor import BaseAction, ActionType, ActionResult
import time

logger = logging.getLogger(__name__)


class BrowserAction(BaseAction):
    """Browser control actions."""
    
    ACTIONS = {
        'new_tab': {'win': 'ctrl+t', 'mac': 'command+t'},
        'close_tab': {'win': 'ctrl+w', 'mac': 'command+w'},
        'refresh': {'win': 'ctrl+r', 'mac': 'command+r'},
        'back': {'win': 'alt+left', 'mac': 'command+left'},
        'forward': {'win': 'alt+right', 'mac': 'command+right'},
        'zoom_in': {'win': 'ctrl+plus', 'mac': 'command+plus'},
        'zoom_out': {'win': 'ctrl+minus', 'mac': 'command+minus'},
        'find': {'win': 'ctrl+f', 'mac': 'command+f'},
    }
    
    def __init__(self, action: str = "new_tab"):
        super().__init__(
            f"Browser: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        import pyautogui
        start = time.time()
        try:
            action = self.params['action']
            
            if action not in self.ACTIONS:
                return ActionResult(False, self.name, f"Unknown action: {action}", time.time() - start)
            
            keys = self.ACTIONS[action]
            system = platform.system()
            
            key_combo = keys['mac'] if system == "Darwin" else keys['win']
            pyautogui.hotkey(*key_combo.split('+'))
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Browser {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class TextEditorAction(BaseAction):
    """Text editor actions."""
    
    ACTIONS = {
        'save': {'win': 'ctrl+s', 'mac': 'command+s'},
        'undo': {'win': 'ctrl+z', 'mac': 'command+z'},
        'redo': {'win': 'ctrl+y', 'mac': 'command+shift+z'},
        'copy': {'win': 'ctrl+c', 'mac': 'command+c'},
        'paste': {'win': 'ctrl+v', 'mac': 'command+v'},
        'cut': {'win': 'ctrl+x', 'mac': 'command+x'},
        'select_all': {'win': 'ctrl+a', 'mac': 'command+a'},
        'find': {'win': 'ctrl+f', 'mac': 'command+f'},
        'replace': {'win': 'ctrl+h', 'mac': 'command+h'},
    }
    
    def __init__(self, action: str = "save"):
        super().__init__(
            f"Editor: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        import pyautogui
        start = time.time()
        try:
            action = self.params['action']
            
            if action not in self.ACTIONS:
                return ActionResult(False, self.name, f"Unknown action: {action}", time.time() - start)
            
            keys = self.ACTIONS[action]
            system = platform.system()
            
            key_combo = keys['mac'] if system == "Darwin" else keys['win']
            pyautogui.hotkey(*key_combo.split('+'))
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Editor {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class PresentationAction(BaseAction):
    """Presentation control (PowerPoint, Keynote, etc.)."""
    
    def __init__(self, action: str = "next_slide"):
        super().__init__(
            f"Presentation: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        import pyautogui
        start = time.time()
        try:
            action = self.params['action']
            
            if action == 'next_slide':
                pyautogui.press('right')
            elif action == 'prev_slide':
                pyautogui.press('left')
            elif action == 'start':
                pyautogui.press('f5')
            elif action == 'end':
                pyautogui.press('escape')
            elif action == 'blank':
                pyautogui.press('b')
            elif action == 'pointer':
                pyautogui.hotkey('ctrl', 'p')
            else:
                return ActionResult(False, self.name, f"Unknown action: {action}", time.time() - start)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Presentation {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class VideoPlayerAction(BaseAction):
    """Video player control."""
    
    def __init__(self, action: str = "play_pause"):
        super().__init__(
            f"Video: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        import pyautogui
        start = time.time()
        try:
            action = self.params['action']
            
            if action == 'play_pause':
                pyautogui.press('space')
            elif action == 'fullscreen':
                pyautogui.press('f')
            elif action == 'mute':
                pyautogui.press('m')
            elif action == 'forward':
                pyautogui.press('right')
            elif action == 'backward':
                pyautogui.press('left')
            elif action == 'volume_up':
                pyautogui.press('up')
            elif action == 'volume_down':
                pyautogui.press('down')
            else:
                return ActionResult(False, self.name, f"Unknown action: {action}", time.time() - start)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Video {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class SpotifyAction(BaseAction):
    """Spotify-specific controls."""
    
    def __init__(self, action: str = "play_pause"):
        super().__init__(
            f"Spotify: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            action = self.params['action']
            system = platform.system()
            
            if system == "Darwin":
                script = {
                    'play_pause': 'tell application "Spotify" to playpause',
                    'next': 'tell application "Spotify" to next track',
                    'prev': 'tell application "Spotify" to previous track',
                }.get(action)
                
                if script:
                    subprocess.run(['osascript', '-e', script])
            else:
                # Use D-Bus on Linux or media keys elsewhere
                import pyautogui
                key_map = {
                    'play_pause': 'playpause',
                    'next': 'nexttrack',
                    'prev': 'prevtrack'
                }
                if action in key_map:
                    pyautogui.press(key_map[action])
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Spotify {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ZoomAction(BaseAction):
    """Zoom meeting controls."""
    
    def __init__(self, action: str = "mute"):
        super().__init__(
            f"Zoom: {action.replace('_', ' ').title()}",
            ActionType.APPLICATION,
            {'action': action}
        )
    
    def execute(self) -> ActionResult:
        import pyautogui
        start = time.time()
        try:
            action = self.params['action']
            system = platform.system()
            
            # Zoom shortcuts
            shortcuts = {
                'mute': 'alt+a' if system != "Darwin" else 'command+shift+a',
                'video': 'alt+v' if system != "Darwin" else 'command+shift+v',
                'share': 'alt+s' if system != "Darwin" else 'command+shift+s',
                'chat': 'alt+h' if system != "Darwin" else 'command+shift+h',
                'raise_hand': 'alt+y' if system != "Darwin" else 'option+y',
                'leave': 'alt+q' if system != "Darwin" else 'command+w',
            }
            
            if action in shortcuts:
                keys = shortcuts[action].split('+')
                pyautogui.hotkey(*keys)
            else:
                return ActionResult(False, self.name, f"Unknown action: {action}", time.time() - start)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Zoom {action}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)
