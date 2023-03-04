import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

pTime = 0
cTime = 0

success, img = cap.read()

while True:
    success, img = cap.read()
    
    # Create mask to only select white
    maskR = cv2.inRange(img, np.array([0, 0, 150]), np.array([75, 75, 255]))
    maskBk = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))

    imgMats = 0

    # Change image to grey where we found white
    # img[maskR > 0] = (255, 255, 255)
    # img[maskBk > 0] = (150, 150, 150)

    img=cv2.cvtColor(img, cv2.COLORMAP_HOT)
    print(img[2])
 
    cv2.imshow('Reflectance Detection', img)
    cv2.waitKey(1)
