import math
import random
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# find function
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20,  25,  30,  35,  40,  45,  50,  55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)
A, B, C = coff

# game variables
cx, cy = 250, 250
color = (255, 0, 255)
counter = 0
score = 0
timeStart = time.time()
totalTime = 20

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if time.time() - timeStart < totalTime:
        hand, img = detector.findHands(img, draw=False)
        if hand:
            lmList = hand[0]['lmList']
            x, y, w, h = hand[0]['bbox']
            x1, y1, _ = lmList[5]
            x2, y2, __ = lmList[17]
            distance = int(math.sqrt((y2-y1)**2+(x2-x1)**2))
            distanceCM = A*(distance**2)+B*distance+C

            if distanceCM < 40:
                if x < cx < x + w and y < cy < y + h:
                    counter = 1

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))

        if counter:
            counter += 1
            color = (0, 255, 0)
            if counter == 3:
                cx = random.randint(100, 1100)
                cy = random.randint(100, 600)
                score += 1
                color = (255, 0, 255)
                counter = 0

        # draw button
        cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
        cv2.circle(img, (cx, cy), 20, (255, 255, 255), 3)
        cv2.circle(img, (cx, cy), 10, (0, 0, 0), 2)
        cv2.circle(img, (cx, cy), 30, (0, 0, 0), 2)
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

        # game head up display
        cvzone.putTextRect(img, f'Time : {int(totalTime+timeStart-time.time())}', (1000, 75), scale=3, offset=20)
        cvzone.putTextRect(img, f'Score : {str(score).zfill(2)}', (60, 75), scale=3, offset=20)

    else:
        cvzone.putTextRect(img, 'Game Over', (400, 300), scale=5, offset=30, thickness=7)
        cvzone.putTextRect(img, f'Score : {str(score).zfill(2)}', (485, 400), scale=3, offset=20, thickness=7)
        cvzone.putTextRect(img, 'Press R to restart', (465, 500), scale=2, offset=20, thickness=7)
        cvzone.putTextRect(img, 'Press Q to restart', (465, 600), scale=2, offset=20, thickness=7)

    cv2.imshow("webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        timeStart = time.time()
        score = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
