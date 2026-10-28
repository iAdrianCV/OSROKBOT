from Actions.action import Action
from window_handler import WindowHandler
import time


class PressKeyAction(Action):
    def __init__(self, key: str, delay=0, retard=0, times=1):
        self.key = key
        self.delay = delay
        self.retard = retard
        self.times = times
        self.window_handler = WindowHandler()

    def execute(self):
        time.sleep(self.delay)
        self.window_handler.hold_key(self.key, duration=1.0)
        return True  # Always return True since pressing a key will not fail
