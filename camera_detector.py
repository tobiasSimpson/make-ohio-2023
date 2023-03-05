import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

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

# Fill initial reflectance scores
def ReflectanceInitialization(img):
    # Initialize reflectance scores
    scores = np.zeros(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape)

    # Load knn model
    knn = pickle.load(open('knn_material_model', 'rb'))

    # Assign reflectance scores to every pixel based on material
    r = 0
    while r < scores.shape[0]:
        c = 0
        while c < scores.shape[1]:
            surroundings = GetSurroundings(img, r, c)
            material = knn.predict([[np.mean(surroundings[:,:,0]), np.mean(surroundings[:,:,1]), np.mean(surroundings[:,:,2])]])

            # Assign initial score based on material type
            if material == "Veneer":
                score = 208
            elif material == "Concrete":
                score = 255
            elif material == "Paper":
                score = 158
            elif material == "Drywall":
                score = 50
            elif material == "Cardboard":
                score = 132
            elif material == "Grass":
                score = 16
            elif material == "Dirt":
                score = 92
            scores[r:r+10][c:c+10] = score
            c = c+10
        r=r+10
    return scores

def main():
    # Initialize video feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    # Initialize accumulation
    success, img = cap.read()

    scores = ReflectanceInitialization(img)

    # Initialize frame count
    frames = 1

    # Update recommendation based on video feed light levels
    while True:
        # Read a new frame
        success, img = cap.read()

        # Convert to gray scale to display light levels
        imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Add to the running total
        scores = cv2.accumulate(imgG, scores)

        # Increment frame count
        frames = frames + 1

        # Display the scaled accumlation
        cv2.imshow('Solar Locator', scores/(255*frames))

        # Accumulated brightness bit mask
        brightMaskG = cv2.inRange(scores/frames, np.percentile(scores, 95)/frames, 255)
        brightMaskB = cv2.inRange(imgG, np.percentile(imgG, 95), 255)

        img[brightMaskG > 0] = [0, 0, 0]
        img[brightMaskB > 0] = [0, 0, 0]
        img[brightMaskG > 0] += np.array((0,255,0), dtype='uint8')
        img[brightMaskB > 0] += np.array((255,0,0), dtype='uint8')

        # Display the sliding average
        #cv2.imshow('Live Solar Update', img)

        # Wait
        cv2.waitKey(1)

if __name__ == "__main__":
    # Create the model if it does not exist
    path = Path('./knn_material_model')
    if not path.is_file():
        # Load Training Data
        materials = pd.read_csv("RGBdata.csv")
        X = materials.drop("Class", axis=1)
        y = materials["Class"]

        # Train Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Its important to use binary mode 
        knnPickle = open('knn_material_model', 'wb') 
            
        # source, destination 
        pickle.dump(knn, knnPickle)  

        # close the file
        knnPickle.close()
  
    # Execute main
    main()
