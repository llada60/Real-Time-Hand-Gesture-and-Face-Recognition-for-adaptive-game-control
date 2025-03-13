import pyautogui

class GestureControl:
    def __init__(self, gesture):
        self.gesture = gesture
    def press_key(self):
        if self.gesture == 'C shape':
            pyautogui.press('space')
    