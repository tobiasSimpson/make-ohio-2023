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


def GetScores(material):
    # Assign initial score based on material type
    if material == "Veneer":
        return 208
    elif material == "Concrete":
        return 255
    elif material == "Paper":
        return 158
    elif material == "Drywall":
        return 50
    elif material == "Cardboard":
        return 132
    elif material == "Grass":
        return 16
    elif material == "Dirt":
        return 92

def GetMaterialScores(img, knn):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))

    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    attempts=1
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    label=label.reshape(img.shape[0:2])
    center = np.uint8(center)

    materials = knn.predict(center)
    scores = np.vectorize(GetScores)(materials)
    # breakpoint()

    # result= lambda label:scores[label]
    result=np.array(list(map(lambda l: scores[l],label)))
    return result


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

            score=GetScores(material)
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


    # Load knn model
    knn = pickle.load(open('knn_material_model', 'rb'))

    # Initialize frame count
    frames = 1

    # Update recommendation based on video feed light levels
    while True:
        # Read a new frame
        success, img = cap.read()

        # Convert to gray scale to display light levels
        imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # get matrix of scores for each pixel
        imgScores = GetMaterialScores(img, knn)

        # Add to the running total
        scores = cv2.accumulate(imgG, scores)

        # Increment frame count
        frames = frames + 1

        # Display the scaled accumlation
        # cv2.imshow('Solar Locator', scores/(255*frames))

        # Accumulated brightness bit mask
        brightMaskG = cv2.inRange(scores/frames, np.percentile(scores, 95)/frames, 255)
        brightMaskB = cv2.inRange(imgG, np.percentile(imgG, 95), 255)

        img[brightMaskG > 0] = [0, 0, 0]
        img[brightMaskB > 0] = [0, 0, 0]
        img[brightMaskG > 0] += np.array((0,255,0), dtype='uint8')
        img[brightMaskB > 0] += np.array((255,0,0), dtype='uint8')

        # Display the sliding average
        cv2.imshow('Live Solar Update', img)

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
