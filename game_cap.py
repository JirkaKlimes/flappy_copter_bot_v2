from window_capture import WindowCapture
import cv2 as cv


# open "http://www.flappycopter.com/" in chrome
# set scale to 100%

win_cap = WindowCapture("chrome")
win_cap.start()

while True:
    if win_cap.screenshot is None: continue
    
    screenshot = win_cap.screenshot[140:-40,186:-186]
    screenshot = cv.resize(screenshot, (350, 500), interpolation=cv.INTER_AREA)
    
    cv.imshow("BOT", screenshot)
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
win_cap.stop()
