import pywt
import cv2
import os
import shutil
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def creatingWavelets(img, mode = 'haar', lev = 1):
    featureCoeff= []

    #converting to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #converting to float values
    imArray = np.float32(imArray)
    imArray /= 255

    # Extracting Coefficients by wavelet decomposition
    coeffs = pywt.wavedec2(imArray, mode, level=lev)

    
    cA2 = coeffs[0] # Approximate Coefficients@level 2
    (cH1, cV1, cD1) = coeffs[-1] # Detailed Coefficients@level 1
    (cH2, cV2, cD2) = coeffs[-2] # Deatiled Coefficients@level 2

    #converting coefficeints into list
    coeffs_H = list(coeffs)
    
    # Processing Approximate Coefficient
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)


    return img, imArray_H


def createDataVals(sportsCelebDict, pathProcessedImages):
    classDict = {}
    count = 1

    for celebNames in sportsCelebDict.keys():
        classDict[celebNames] = count
        count += 1

    X, y = [], []
    count = 0 
    for celebFolder in os.scandir(pathProcessedImages):
        count += 1
        for entry in os.scandir(celebFolder):
            img, im_har = creatingWavelets(cv2.imread(entry.path), 'db2', 3)
            scaledRawimage= cv2.resize(img, (32, 32))
            scaledHaarImage = cv2.resize(im_har, (32, 32))
            combinedImage = np.vstack((scaledRawimage.reshape(32*32*3,1), scaledHaarImage.reshape(32*32,1)))
            X.append(combinedImage)
            y.append(count)

    X = np.array(X).reshape(len(X),4096).astype(float)
    return X, y, classDict

#training SVM model
def trainSVMModel(X_train, X_test, y_train, y_test):
    svmClf = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'poly', C = 1))])
    svmClf.fit(X_train, y_train)

    #dumping model into models folder
    with open('./models/SVM.pkl', 'wb') as f:
        pickle.dump(svmClf, f)
    
    return svmClf.score(X_test, y_test), classification_report(y_test, svmClf.predict(X_test))

#training randomforest model
def tarinRandomForestClassifierModel(X_train, X_test, y_train, y_test):
    rfClf = Pipeline([('scaler', StandardScaler()), ('randomforest', RandomForestClassifier(criterion = "gini", max_depth = 8, min_samples_split=10, random_state=5))])
    rfClf.fit(X_train, y_train)

    #dumping model into models folder
    pickle.dump(rfClf, open('./models/RandomForest.pkl', "wb"))
    
    return rfClf.score(X_test, y_test), classification_report(y_test, rfClf.predict(X_test))


#training randomforest model
def tarinLogisticRegressionModel(X_train, X_test, y_train, y_test):
    lrClf = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver = "liblinear", multi_class= 'auto', random_state=10))])
    lrClf.fit(X_train, y_train)

    #dumping model into models folder
    pickle.dump(lrClf, open('./models/LogisticRegression.pkl', "wb"))
    
    return lrClf.score(X_test, y_test), classification_report(y_test, lrClf.predict(X_test))


