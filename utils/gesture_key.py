import pyautogui

class GestureControl:
    def __init__(self, gesture):
        self.gesture = gesture
    def press_key(self, module_num):
        if module_num == 1 or module_num == 3:
            if self.gesture == 'C shape':
                pyautogui.press('space')
            elif self.gesture == 'Horizontal':
                pyautogui.press('down')
        elif module_num == 2 or module_num == 3:
            if self.gesture == 'Up':
                pyautogui.press('up')
            elif self.gesture == 'Down':
                pyautogui.press('down')
            elif self.gesture == 'Left':
                pyautogui.press('left')
            elif self.gesture == 'Right':
                pyautogui.press('right')
            

    