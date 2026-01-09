"""
System-level Actions
"""

import os
import sys
import subprocess
import platform
import logging
from typing import Optional

import pyautogui
import psutil

from core.action_executor import BaseAction, ActionType, ActionResult
import time

logger = logging.getLogger(__name__)


class BrightnessAction(BaseAction):
    """Screen brightness control."""
    
    def __init__(self, direction: str = "up", amount: int = 10):
        super().__init__(
            f"Brightness {direction.capitalize()}",
            ActionType.SYSTEM,
            {'direction': direction, 'amount': amount}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            direction = self.params['direction']
            amount = self.params['amount']
            
            system = platform.system()
            
            if system == "Windows":
                try:
                    import screen_brightness_control as sbc
                    current = sbc.get_brightness()[0]
                    if direction == 'up':
                        new_val = min(100, current + amount)
                    else:
                        new_val = max(0, current - amount)
                    sbc.set_brightness(new_val)
                except ImportError:
                    # Fallback to WMI
                    pass
            elif system == "Darwin":  # macOS
                script = f'''
                    tell application "System Preferences"
                        activate
                        reveal anchor "displaysDisplayTab" of pane "com.apple.preference.displays"
                    end tell
                '''
                subprocess.run(['osascript', '-e', script])
            else:  # Linux
                cmd = f"xrandr --output $(xrandr | grep ' connected' | cut -d' ' -f1) --brightness {1.0 + (0.1 if direction == 'up' else -0.1)}"
                subprocess.run(cmd, shell=True)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Brightness {direction}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class LockScreenAction(BaseAction):
    """Lock the screen."""
    
    def __init__(self):
        super().__init__("Lock Screen", ActionType.SYSTEM)
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            system = platform.system()
            
            if system == "Windows":
                import ctypes
                ctypes.windll.user32.LockWorkStation()
            elif system == "Darwin":
                subprocess.run([
                    'osascript', '-e',
                    'tell application "System Events" to keystroke "q" using {command down, control down}'
                ])
            else:
                subprocess.run(['xdg-screensaver', 'lock'])
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, "Screen locked", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class SleepAction(BaseAction):
    """Put computer to sleep."""
    
    def __init__(self):
        super().__init__("Sleep", ActionType.SYSTEM)
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            system = platform.system()
            
            if system == "Windows":
                subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0,1,0'])
            elif system == "Darwin":
                subprocess.run(['pmset', 'sleepnow'])
            else:
                subprocess.run(['systemctl', 'suspend'])
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, "Going to sleep", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ShowDesktopAction(BaseAction):
    """Show/minimize all windows to desktop."""
    
    def __init__(self):
        super().__init__("Show Desktop", ActionType.SYSTEM)
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            system = platform.system()
            
            if system == "Windows":
                pyautogui.hotkey('win', 'd')
            elif system == "Darwin":
                pyautogui.hotkey('fn', 'f11')
            else:
                pyautogui.hotkey('super', 'd')
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, "Showing desktop", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class SwitchWindowAction(BaseAction):
    """Switch between windows."""
    
    def __init__(self, direction: str = "next"):
        super().__init__(
            f"Switch Window ({direction})",
            ActionType.SYSTEM,
            {'direction': direction}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            direction = self.params['direction']
            system = platform.system()
            
            if direction == 'next':
                if system == "Darwin":
                    pyautogui.hotkey('command', 'tab')
                else:
                    pyautogui.hotkey('alt', 'tab')
            else:
                if system == "Darwin":
                    pyautogui.hotkey('command', 'shift', 'tab')
                else:
                    pyautogui.hotkey('alt', 'shift', 'tab')
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, f"Switched window", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class TaskManagerAction(BaseAction):
    """Open task manager/activity monitor."""
    
    def __init__(self):
        super().__init__("Task Manager", ActionType.SYSTEM)
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            system = platform.system()
            
            if system == "Windows":
                subprocess.Popen(['taskmgr.exe'])
            elif system == "Darwin":
                subprocess.Popen(['open', '-a', 'Activity Monitor'])
            else:
                # Try common task managers
                for tm in ['gnome-system-monitor', 'ksysguard', 'xfce4-taskmanager']:
                    try:
                        subprocess.Popen([tm])
                        break
                    except FileNotFoundError:
                        continue
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, "Opened task manager", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class NotificationAction(BaseAction):
    """Show a system notification."""
    
    def __init__(self, title: str = "Gesture", message: str = "Action executed"):
        super().__init__(
            "Notification",
            ActionType.SYSTEM,
            {'title': title, 'message': message}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            title = self.params['title']
            message = self.params['message']
            system = platform.system()
            
            if system == "Windows":
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=3, threaded=True)
            elif system == "Darwin":
                subprocess.run([
                    'osascript', '-e',
                    f'display notification "{message}" with title "{title}"'
                ])
            else:
                subprocess.run(['notify-send', title, message])
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, "Notification sent", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)
