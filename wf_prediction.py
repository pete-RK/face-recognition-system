import numpy as np
import base64
import cv2
import os
import pickle
import wf_ml_training as tr
from matplotlib import pyplot as plt


def detect_FaceandEyes(image):
    #creating a grey scale of requested image
    if image is None:
        return None

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


def predictSample():
    cls_PickleModel = pickle.load(open("./models/RandomForest.pkl", "rb"))

    result= []

    path = './test_images'

    for entry in os.scandir(path):
        image  = detect_FaceandEyes(cv2.imread(entry.path))
        if image is None:
            continue
        img, imgHar = tr.creatingWavelets(image, 'db2', 3)
        scaledRawimage= cv2.resize(img, (32, 32))
        scaledHaarImage = cv2.resize(imgHar, (32, 32))
        combinedImage = np.vstack((scaledRawimage.reshape(32*32*3,1), scaledHaarImage.reshape(32*32,1)))

        len_image_array = 32*32*3 + 32*32

        final = combinedImage.reshape(1,len_image_array).astype(float)

        if cls_PickleModel.predict(final)[0] == 1:
            result.append('Cristiano')
        else:
            result.append('Messi')

    path2 = path = r'./evaluation/evaluation.txt'
    with open(path2,'w') as data:  
      data.write(str(result))
