import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

success, scores = cap.read()
scores = cv2.cvtColor(scores, cv2.COLOR_BGR2GRAY)


while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create mask to only select white
    # maskR = cv2.inRange(img, np.array([0, 0, 150]), np.array([75, 75, 255]))
    # maskBk = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))

    # scoresSum = np.sum(scores)
    # scores = np.append(scores, np.zeros((1280-720, 1280)), axis=0) / scoresSum
    # img = np.append(img, np.zeros((1280-720, 1280)), axis=0) / 255
    # scores = np.matmul(scores, img)

    img2 = np.float32(np.zeros(img.size))
    img2 = cv2.accumulate(img, img2)
    img2 = cv2.accumulate(scores, img2)
    scores = scores/2

    cv2.imshow('Reflectance Detection', scores)
    print(scores)
    cv2.waitKey(1)
