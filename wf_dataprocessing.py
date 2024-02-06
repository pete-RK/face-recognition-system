import os
import shutil
import cv2

# function for getting paths
def createDirectories():
    # storing paths for 
    pathDataset = "./data_original/celebrity"
    pathProcessedImages = "./data_processed/"

    # creating directories for storing data
    imgDir = []
    for entry in os.scandir(pathDataset):
        if entry.is_dir():
            imgDir.append(entry.path)

    if os.path.exists(pathProcessedImages):
        shutil.rmtree(pathProcessedImages)
    os.mkdir(pathProcessedImages)

    return imgDir, pathProcessedImages


#function for detecting face and eyes
def detect_FaceandEyes(image):
    #creating a grey scale of requested image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #face and eyes detection using haarcascades
    face_detector= cv2.CascadeClassifier('./data_original/haarcascades/haarcascade_frontalface_default.xml')
    eyes_detected= cv2.CascadeClassifier('./data_original/haarcascades/haarcascade_eye.xml')
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eyes_detected.detectMultiScale(roi_gray)
            if len(eyes) >= 2:      
                return roi_color
    else:
        face_detector= cv2.CascadeClassifier('./data_original/haarcascades/haarcascade_profileface.xml')
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eyes_detected.detectMultiScale(roi_gray)
            if len(eyes) >= 0:      
                return roi_color

# function to store cropped images into required directory       
def storeCroppedData(imgDir, pathProcessedImages):
    croppedDirectories = []
    sportsCelebDict = {}

    for img in imgDir:
        count = 1
        celebName = img.split('/')[-1]
        sportsCelebDict[celebName] = []

        for entry in os.scandir(img):
            if entry.path.lower().endswith(('.png', '.jpg', '.jpeg')):
                roi_color = detect_FaceandEyes(cv2.imread(entry.path))
                if roi_color is not None:
                    cropped_Folder = pathProcessedImages + celebName
                    if not os.path.exists(cropped_Folder):
                        os.makedirs(cropped_Folder)
                        croppedDirectories.append(cropped_Folder)
                        print("Generated: ", cropped_Folder)

                    croppedFileName = celebName + str(count) +".png"
                    croppedFilePath = cropped_Folder + "/" + croppedFileName

                    cv2.imwrite(croppedFilePath, roi_color)
                    sportsCelebDict[celebName].append(croppedFilePath)
                    count += 1
    return croppedDirectories, sportsCelebDict
            


