import time
from Actions.action import Action
from window_handler import WindowHandler


class ManualClickAction(Action):
    def __init__(self, x=50, y=50, delay=0, remember_position=True, retard=0.0):
        self.window_handler = WindowHandler()
        self.delay = delay
        self.window_title = 'Rise of Kingdoms'
        self.x = x
        self.y = y
        self.remember_position = remember_position
        self.retard = retard

    def execute(self):
        time.sleep(0.1)
        win = self.window_handler.get_window(self.window_title)
        if not win:
            return False
        click_x = int(win.left + win.width * self.x / 100)
        click_y = int(win.top + win.height * self.y / 100)
        return self.window_handler.click_screen(click_x, click_y, self.window_title)
