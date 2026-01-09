"""
Action Registry - Central Action Management
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.action_executor import (
    BaseAction, ActionType, ActionResult,
    VolumeAction, MediaAction, KeyboardAction,
    MouseAction, ApplicationAction, ScreenshotAction
)
from actions.system_actions import (
    BrightnessAction, LockScreenAction, SleepAction,
    ShowDesktopAction, SwitchWindowAction, TaskManagerAction,
    NotificationAction
)
from actions.app_actions import (
    BrowserAction, TextEditorAction, PresentationAction,
    VideoPlayerAction, SpotifyAction, ZoomAction
)
from actions.custom_actions import (
    ScriptAction, PythonAction, WebhookAction,
    IFTTTAction, SequenceAction
)

logger = logging.getLogger(__name__)


class ActionRegistry:
    """Central registry for all available actions."""
    
    # Built-in action classes
    BUILTIN_ACTIONS: Dict[str, Type[BaseAction]] = {
        # System
        'volume_up': lambda: VolumeAction('up'),
        'volume_down': lambda: VolumeAction('down'),
        'volume_mute': lambda: VolumeAction('mute'),
        'brightness_up': lambda: BrightnessAction('up'),
        'brightness_down': lambda: BrightnessAction('down'),
        'lock_screen': LockScreenAction,
        'sleep': SleepAction,
        'show_desktop': ShowDesktopAction,
        'switch_window': SwitchWindowAction,
        'task_manager': TaskManagerAction,
        'screenshot': ScreenshotAction,
        
        # Media
        'media_play_pause': lambda: MediaAction('play_pause'),
        'media_next': lambda: MediaAction('next'),
        'media_previous': lambda: MediaAction('previous'),
        'media_stop': lambda: MediaAction('stop'),
        
        # Mouse
        'mouse_click': lambda: MouseAction('click'),
        'mouse_double_click': lambda: MouseAction('double_click'),
        'mouse_right_click': lambda: MouseAction('right_click'),
        'scroll_up': lambda: MouseAction('scroll_up'),
        'scroll_down': lambda: MouseAction('scroll_down'),
        
        # Browser
        'browser_new_tab': lambda: BrowserAction('new_tab'),
        'browser_close_tab': lambda: BrowserAction('close_tab'),
        'browser_refresh': lambda: BrowserAction('refresh'),
        'browser_back': lambda: BrowserAction('back'),
        'browser_forward': lambda: BrowserAction('forward'),
        
        # Presentation
        'presentation_next': lambda: PresentationAction('next_slide'),
        'presentation_prev': lambda: PresentationAction('prev_slide'),
        'presentation_start': lambda: PresentationAction('start'),
        'presentation_end': lambda: PresentationAction('end'),
        
        # Video
        'video_play_pause': lambda: VideoPlayerAction('play_pause'),
        'video_fullscreen': lambda: VideoPlayerAction('fullscreen'),
        'video_mute': lambda: VideoPlayerAction('mute'),
        
        # Zoom
        'zoom_mute': lambda: ZoomAction('mute'),
        'zoom_video': lambda: ZoomAction('video'),
        'zoom_share': lambda: ZoomAction('share'),
        'zoom_raise_hand': lambda: ZoomAction('raise_hand'),
    }
    
    def __init__(self):
        self.actions: Dict[str, BaseAction] = {}
        self.custom_actions: Dict[str, dict] = {}
        self._initialize_builtin()
    
    def _initialize_builtin(self):
        """Initialize built-in actions."""
        for name, factory in self.BUILTIN_ACTIONS.items():
            try:
                if callable(factory):
                    action = factory()
                else:
                    action = factory()
                self.actions[name] = action
            except Exception as e:
                logger.warning(f"Failed to initialize action {name}: {e}")
    
    def register(self, action: BaseAction, name: str = None):
        """Register a new action."""
        action_name = name or action.name
        self.actions[action_name] = action
        logger.debug(f"Registered action: {action_name}")
    
    def unregister(self, name: str):
        """Unregister an action."""
        if name in self.actions:
            del self.actions[name]
            logger.debug(f"Unregistered action: {name}")
    
    def get(self, name: str) -> Optional[BaseAction]:
        """Get an action by name."""
        return self.actions.get(name)
    
    def list_actions(self) -> List[Dict]:
        """List all available actions."""
        return [
            {
                'name': name,
                'type': action.action_type.value,
                'enabled': action.enabled,
                'params': action.params
            }
            for name, action in self.actions.items()
        ]
    
    def list_by_type(self, action_type: ActionType) -> List[str]:
        """List actions of a specific type."""
        return [
            name for name, action in self.actions.items()
            if action.action_type == action_type
        ]
    
    def execute(self, name: str) -> Optional[ActionResult]:
        """Execute an action by name."""
        action = self.get(name)
        if action and action.enabled and action.can_execute():
            return action.execute()
        return None
    
    def save_custom_actions(self, filepath: str):
        """Save custom actions to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only save serializable custom actions
        serializable = {}
        for name, action in self.actions.items():
            if name not in self.BUILTIN_ACTIONS:
                if isinstance(action, ScriptAction):
                    serializable[name] = {
                        'type': 'script',
                        'params': action.params
                    }
                elif isinstance(action, WebhookAction):
                    serializable[name] = {
                        'type': 'webhook',
                        'params': action.params
                    }
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load_custom_actions(self, filepath: str):
        """Load custom actions from file."""
        path = Path(filepath)
        
        if not path.exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for name, config in data.items():
            action_type = config.get('type')
            params = config.get('params', {})
            
            if action_type == 'script':
                action = ScriptAction(
                    params.get('script', ''),
                    params.get('shell', True),
                    name
                )
            elif action_type == 'webhook':
                action = WebhookAction(
                    params.get('url', ''),
                    params.get('method', 'POST'),
                    params.get('data'),
                    params.get('headers'),
                    name
                )
            else:
                continue
            
            self.register(action, name)
    
    def create_keyboard_action(self, keys: str, hotkey: bool = False, name: str = None) -> BaseAction:
        """Create and register a keyboard action."""
        action = KeyboardAction(keys, hotkey)
        action_name = name or f"keyboard_{keys.replace('+', '_')}"
        self.register(action, action_name)
        return action
    
    def create_application_action(self, path: str, name: str = None) -> BaseAction:
        """Create and register an application launch action."""
        import os
        action = ApplicationAction(path)
        action_name = name or f"app_{os.path.basename(path)}"
        self.register(action, action_name)
        return action


# Global registry instance
_registry: Optional[ActionRegistry] = None


def get_registry() -> ActionRegistry:
    """Get the global action registry."""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
    return _registry
