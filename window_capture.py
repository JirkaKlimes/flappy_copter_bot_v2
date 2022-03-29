import mss
import numpy as np
import win32gui
from threading import Thread, Lock
import time
import cv2 as cv

class WindowCapture:

    def __init__(self, window_name=None):
        self.window_name = window_name
        self.screenshot = None
        self.running = False
        self.lock = Lock()
        self.width = 1920
        self.height = 1080
        self.waiting_for_window = True

    def get_hwnd(self):
        if self.window_name is not None:
            windows = self.get_windows()
            for window in windows:
                title = window[1].lower()
                if self.window_name.lower() in title:
                    self.hwnd = window[0]
                    self.waiting_for_window = False
                    return
            self.waiting_for_window = True
            return
        self.hwnd = win32gui.GetDesktopWindow()
        self.waiting_for_window = False


    def get_rect(self):
        rect = win32gui.GetWindowRect(self.hwnd)
        y = rect[1]
        x = rect[0]
        self.width = rect[2] - x
        self.height = rect[3] - y
        return x, y, self.width, self.height
    
    def get_window_position(self):
        x, y, _, _ = win32gui.GetWindowRect(self.hwnd)
        return x, y
    
    def ofset_position(self, position):
        ox, oy = self.get_window_position()
        return ox + position[0], oy + position[1]

    def take_screenshot(self, save=False):
        with mss.mss() as sct:
            self.get_hwnd()
            if self.waiting_for_window: return None
            x, y, w, h = self.get_rect()
            monitor = {"top": y, "left": x, "width": w, "height": h}
            screenshot = np.array(sct.grab(monitor))[:,:,:3]
            screenshot = np.ascontiguousarray(screenshot, dtype=np.uint8)
            if save: cv.imwrite(f'{time.time()}.png', screenshot)
            return screenshot

    def _capture_loop(self):
        self.get_hwnd()
        while self.running:
            self.lock.acquire()
            self.screenshot = self.take_screenshot()
            self.lock.release()

    def start(self):
        self.running = True
        self.thread = Thread(target=self._capture_loop)
        self.thread.start()

    def stop(self):
        self.running = False

    @staticmethod
    def get_windows(out=False):
        windows = []
        def winEnumHandler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if not title: return
                windows.append((hwnd, title))
                if out: print(f'{hwnd} - {title}')
        win32gui.EnumWindows(winEnumHandler,None)
        return windows