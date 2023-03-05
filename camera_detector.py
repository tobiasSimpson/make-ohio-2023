import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Initialize video feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

# Initialize accumulation
scores = np.zeros((720,1280))
# Initialize frame count
frames = 0

# Update recommendation based on video feed light levels
while True:
    # Read a new frame
    success, img = cap.read()
    # Convert to gray scale to display light levels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add to the running total
    scores = cv2.accumulate(img, scores)

    # Increment frame count
    frames = frames + 1

    # Display the scaled accumlation
    cv2.imshow('Reflectance Detection', scores/(255*frames))

    # Wait
    cv2.waitKey(1)

    print(scores)
