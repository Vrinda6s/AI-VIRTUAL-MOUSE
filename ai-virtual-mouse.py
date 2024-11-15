import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui
import pygetwindow as gw 

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(mode=False,maxHands=1, detectionCon=0.5, trackCon=0.5)
wScr, hScr = pyautogui.size()
# print(wScr, hScr)

# Variables for click detection
clicking = False
click_counter = 0
click_threshold = 10  # Number of consistent frames required for a click
click_delay = 0.3  # Delay in seconds between clicks

last_click_time = time.time()

# Load thumbs-up image
thumbs_up_img = cv2.imread('thumbs_up.png')
thumbs_up_img = cv2.resize(thumbs_up_img, (100, 100))


def isThumbsUp(lmList):
    # Ensure the thumb is up
    thumbIsUp = lmList[4][2] < lmList[3][2] and lmList[4][2] < lmList[2][2]

    # Ensure other fingers are down
    fingersAreDown = all(lmList[tip][2] > lmList[tip - 2][2] for tip in [8, 12, 16, 20])

    # Ensure the thumb is pointing up
    thumbIsPointingUp = lmList[4][1] < lmList[3][1] and lmList[4][2] < lmList[2][2]

    return thumbIsUp and fingersAreDown and thumbIsPointingUp

def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    alpha = overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]

cv2.namedWindow("Image")
cv2.moveWindow("Image", 0, 0)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
    (255, 0, 255), 2)
    
    if len(fingers) >= 3:
        if isThumbsUp(lmList):
            print("Thumbs Up detected!")
            h, w, _ = thumbs_up_img.shape
            img[10:10+h, 10:10+w] = thumbs_up_img  # Adjust the position as needed
            cv2.imshow("Image", img)
            cv2.waitKey(3000)  # Display the thumbs-up image for 3 seconds
            break  # End the program
        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
        
            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

            click_counter = 0
            clicking = False
        
        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. Click mouse if distance short
            click_counter += 1
            if click_counter > click_threshold:
                if not clicking and time.time() - last_click_time > click_delay:
                    clicking = True
                    last_click_time = time.time()
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    print("Click")
                click_counter = 0
        else:
                click_counter = 0
                clicking = False
    
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    try:
        window = gw.getWindowsWithTitle("Image")[0]
        window.activate()
        window.alwaysOnTop = True
    except IndexError:
        pass
    cv2.waitKey(1)