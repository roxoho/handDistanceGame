import math
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

#find function
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [ 20,  25,  30,  35,  40,  45,  50,  55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)
A, B, C = coff

while True:
    ret, img = cap.read()
    hand, img = detector.findHands(img, draw=False)

    if hand:
        lmList = hand[0]['lmList']
        x, y, w, h = hand[0]['bbox']
        x1, y1, _ = lmList[5]
        x2, y2, __ = lmList[17]
        distance = int(math.sqrt((y2-y1)**2+(x2-x1)**2))
        distanceCM = A*(distance**2)+B*distance+C
        #print(distanceCM)
        cv2.rectangle(img,(x, y), (x+w, y+h), (255, 0, 255), 4)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))

    cv2.imshow("webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
