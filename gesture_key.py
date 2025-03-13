import pyautogui

class GestureControl:
    def __init__(self, gesture):
        self.gesture = gesture
    def press_key(self):
        if self.gesture == 'C shape':
            pyautogui.press('space')
        elif self.gesture == 'Up':
            pyautogui.press('up')
        elif self.gesture == 'Down':
            pyautogui.press('down')
        elif self.gesture == 'Left':
            pyautogui.press('left')
        elif self.gesture == 'Right':
            pyautogui.press('right')
    