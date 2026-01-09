"""
Custom User-defined Actions
"""

import os
import subprocess
import logging
from typing import Callable, Dict, Any, Optional
import json

from core.action_executor import BaseAction, ActionType, ActionResult
import time

logger = logging.getLogger(__name__)


class ScriptAction(BaseAction):
    """Execute a custom script or command."""
    
    def __init__(self, script: str, shell: bool = True, name: str = "Custom Script"):
        super().__init__(
            name,
            ActionType.CUSTOM,
            {'script': script, 'shell': shell}
        )
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            script = self.params['script']
            shell = self.params.get('shell', True)
            
            result = subprocess.run(
                script,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.last_executed = time.time() * 1000
                return ActionResult(True, self.name, result.stdout[:100], time.time() - start)
            else:
                return ActionResult(False, self.name, result.stderr[:100], time.time() - start)
        except subprocess.TimeoutExpired:
            return ActionResult(False, self.name, "Script timed out", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class PythonAction(BaseAction):
    """Execute a Python function."""
    
    def __init__(self, func: Callable, name: str = "Python Function", args: tuple = (), kwargs: dict = None):
        super().__init__(
            name,
            ActionType.CUSTOM,
            {'func': func, 'args': args, 'kwargs': kwargs or {}}
        )
        self._func = func
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            args = self.params.get('args', ())
            kwargs = self.params.get('kwargs', {})
            
            result = self._func(*args, **kwargs)
            
            self.last_executed = time.time() * 1000
            return ActionResult(True, self.name, str(result)[:100], time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class WebhookAction(BaseAction):
    """Send a webhook request."""
    
    def __init__(
        self,
        url: str,
        method: str = "POST",
        data: Dict = None,
        headers: Dict = None,
        name: str = "Webhook"
    ):
        super().__init__(
            name,
            ActionType.CUSTOM,
            {
                'url': url,
                'method': method,
                'data': data or {},
                'headers': headers or {}
            }
        )
    
    def execute(self) -> ActionResult:
        import requests
        start = time.time()
        try:
            url = self.params['url']
            method = self.params['method']
            data = self.params.get('data', {})
            headers = self.params.get('headers', {})
            
            if method.upper() == 'GET':
                response = requests.get(url, params=data, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                response = requests.request(method, url, json=data, headers=headers, timeout=10)
            
            if response.ok:
                self.last_executed = time.time() * 1000
                return ActionResult(True, self.name, f"Status: {response.status_code}", time.time() - start)
            else:
                return ActionResult(False, self.name, f"HTTP {response.status_code}", time.time() - start)
        except requests.Timeout:
            return ActionResult(False, self.name, "Request timed out", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class IFTTTAction(BaseAction):
    """Trigger IFTTT webhook."""
    
    def __init__(self, event: str, key: str, value1: str = "", value2: str = "", value3: str = ""):
        super().__init__(
            f"IFTTT: {event}",
            ActionType.CUSTOM,
            {
                'event': event,
                'key': key,
                'value1': value1,
                'value2': value2,
                'value3': value3
            }
        )
    
    def execute(self) -> ActionResult:
        import requests
        start = time.time()
        try:
            event = self.params['event']
            key = self.params['key']
            
            url = f"https://maker.ifttt.com/trigger/{event}/with/key/{key}"
            data = {
                'value1': self.params.get('value1', ''),
                'value2': self.params.get('value2', ''),
                'value3': self.params.get('value3', '')
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.ok:
                self.last_executed = time.time() * 1000
                return ActionResult(True, self.name, "IFTTT triggered", time.time() - start)
            else:
                return ActionResult(False, self.name, f"HTTP {response.status_code}", time.time() - start)
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class SequenceAction(BaseAction):
    """Execute a sequence of actions."""
    
    def __init__(self, actions: list, name: str = "Action Sequence", delay_ms: int = 100):
        super().__init__(
            name,
            ActionType.CUSTOM,
            {'actions': actions, 'delay_ms': delay_ms}
        )
        self._actions = actions
    
    def execute(self) -> ActionResult:
        start = time.time()
        results = []
        delay = self.params.get('delay_ms', 100) / 1000
        
        try:
            for action in self._actions:
                result = action.execute()
                results.append(result)
                
                if not result.success:
                    return ActionResult(
                        False, self.name,
                        f"Failed at: {result.action_name}",
                        time.time() - start
                    )
                
                time.sleep(delay)
            
            self.last_executed = time.time() * 1000
            return ActionResult(
                True, self.name,
                f"Completed {len(results)} actions",
                time.time() - start
            )
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)


class ConditionalAction(BaseAction):
    """Execute action based on condition."""
    
    def __init__(
        self,
        condition: Callable[[], bool],
        true_action: BaseAction,
        false_action: Optional[BaseAction] = None,
        name: str = "Conditional Action"
    ):
        super().__init__(name, ActionType.CUSTOM)
        self._condition = condition
        self._true_action = true_action
        self._false_action = false_action
    
    def execute(self) -> ActionResult:
        start = time.time()
        try:
            if self._condition():
                result = self._true_action.execute()
            elif self._false_action:
                result = self._false_action.execute()
            else:
                return ActionResult(True, self.name, "Condition false, no action", time.time() - start)
            
            self.last_executed = time.time() * 1000
            return result
        except Exception as e:
            return ActionResult(False, self.name, str(e), time.time() - start)
