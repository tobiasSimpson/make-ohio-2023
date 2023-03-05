import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

SLIDINGFRAMES = 2

# Returns a 25x25 pixel box with the given pixel in the center
def GetSurroundings(img, r, c):
    if r - 12 >= 0 and r + 12 < img.shape[0]:
        if c - 12 >= 0 and c + 12 < img.shape[1]:
            return img[r-12:r+12,c-12:c+12,:]
        elif c + 12 >= img.shape[1]:
            return img[r-12:r+12,c-12:img.shape[1]-1,:]
        elif c - 12 < 0:
            return img[r-12:r+12,0:c+12,:]
    elif r - 12 < 0:
        if c - 12 >= 0 and c + 12 < img.shape[1]:
            return img[0:r+12,c-12:c+12,:]
        elif c + 12 >= img.shape[1]:
            return img[0:r+12,c-12:img.shape[1]-1,:]
        elif c - 12 < 0:
            return img[0:r+12,0:c+12,:]
    elif r + 12 >= img.shape[0]:
        if c - 12 >= 0 and c + 12 < img.shape[1]:
            return img[r-12:img.shape[0] - 1,c-12:c+12,:]
        elif c + 12 >= img.shape[1]:
            return img[r-12:img.shape[0] - 1,c-12:img.shape[1]-1,:]
        elif c - 12 < 0:
            return img[r-12:img.shape[0] - 1,0:c+12,:]

def DetectMaterial(surroundings):
    return 0

def ReflectanceInitialization(img):
    # Initialize reflectance scores
    scores = np.zeros(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape)

    # Assign reflectance scores to every pixel based on material
    for r in range(0, scores.shape[0]):
        for c in range(0, scores.shape[1]):
            surroundings = GetSurroundings(img, r, c)
            scores[r][c] = DetectMaterial(surroundings)
    return scores

def main():
    # Initialize video feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    # Initialize accumulation
    success, img = cap.read()
    scores = ReflectanceInitialization(img)
    slidingScale = np.zeros((SLIDINGFRAMES, img.shape[0], img.shape[1], img.shape[2]))
    #slidingScores = np.zeros((SLIDINGFRAMES, img.shape[0], img.shape[1]))
    slidingIdx = 0
    # Initialize frame count
    frames = 1

    # Update recommendation based on video feed light levels
    while True:
        # Read a new frame
        success, img = cap.read()

        # Add image to sliding frame
        slidingScale[slidingIdx] = img
        #slidingScores[slidingIdx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert to gray scale to display light levels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Add to the running total
        scores = cv2.accumulate(img, scores)

        # Increment frame count
        frames = frames + 1

        # Increment sliding frame count
        if slidingIdx == (SLIDINGFRAMES - 1):
            slidingIdx = 0
        else:
            slidingIdx = slidingIdx + 1

        # Display the scaled accumlation
        #cv2.imshow('Solar Locator', scores/(255*frames))

        # Accumulated brightness bit mask
        brightMask = cv2.inRange(scores/frames, np.percentile(scores, 95)/frames, 255)

        # Calculate sliding average
        slidingScaleTotal = np.zeros(slidingScale[0].shape)
        #slidingScoresTotal = np.zeros(slidingScores[0].shape)
        for i in range(0, SLIDINGFRAMES):
            slidingScaleTotal = cv2.accumulate(slidingScale[i], slidingScaleTotal)
            #slidingScoresTotal = cv2.accumulate(slidingScores[i], slidingScoresTotal)

        # Mask bright areas with green
        slidingScaleTotal = slidingScaleTotal/(255*SLIDINGFRAMES)
        #slidingScoresTotal = slidingScoresTotal/(255*SLIDINGFRAMES)

        slidingScaleTotal32 = np.float32(slidingScaleTotal)
        slidingGray = cv2.cvtColor(slidingScaleTotal32, cv2.COLOR_BGR2GRAY)
        slidingBrightMask = cv2.inRange(slidingGray, np.percentile(slidingGray, 95), 255)

        slidingScaleTotal[brightMask > 0] = [0, 0, 0]
        slidingScaleTotal[slidingBrightMask > 0] = [0, 0, 0]
        slidingScaleTotal[brightMask > 0] += [0, 255, 0]
        slidingScaleTotal[slidingBrightMask > 0] += [255, 0, 0]

        # Display the sliding average
        cv2.imshow('Sliding Frame', slidingScaleTotal)

        # Wait
        cv2.waitKey(1)

if __name__ == "__main__":
    main()