import pywt
import cv2
import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Function for generation coefficients for each image to extract features
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

    #Get max mean for cA2 
    cA2_Mean = []
    cA2_Min = []
    cA2_Max = []
    for i in cA2:
        cA2_Mean.append(i.mean())
        cA2_Min.append(min(i))
        cA2_Max.append(max(i))
    cA2_MaxMean = max(cA2_Mean)
    featureCoeff.append(cA2_MaxMean)

    #Get max mean for cH2 
    cH2_Mean = []
    cH2_Min = []
    cH2_Max = []
    for i in cH2:
        cH2_Mean.append(i.mean())
        cH2_Min.append(min(i))
        cH2_Max.append(max(i))
    cH2_MaxMean = max(cH2_Mean)
    featureCoeff.append(cH2_MaxMean)

     #Get max mean for cV2 
    cV2_Mean = []
    cV2_Min = []
    cV2_Max = []
    for i in cV2:
        cV2_Mean.append(i.mean())
        cV2_Min.append(min(i))
        cV2_Max.append(max(i))
    cV2_MaxMean = max(cV2_Mean)
    featureCoeff.append(cV2_MaxMean)

    #Get max mean for cD2 
    cD2_Mean = []
    cD2_Min = []
    cD2_Max = []
    for i in cD2:
        cD2_Mean.append(i.mean())
        cD2_Min.append(min(i))
        cD2_Max.append(max(i))
    cD2_MaxMean = max(cD2_Mean)
    featureCoeff.append(cD2_MaxMean)


    #converting coefficeints into list
    coeffs_H = list(coeffs)
    
    # Processing Approximate Coefficient
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)


    return imArray_H, featureCoeff

#Function to extract required features
def extractingFeatures(cropped_Folder):
    featureCoeff_A2 = []
    featureCoeff_H2 = []
    featureCoeff_V2 = []
    featureCoeff_D2 = []
    for entry in os.scandir(cropped_Folder):
        im_har, featureCoeff = creatingWavelets(cv2.imread(entry.path), 'db2', 3)
        featureCoeff_A2.append(featureCoeff[0])
        featureCoeff_H2.append(featureCoeff[1])
        featureCoeff_V2.append(featureCoeff[2])
        featureCoeff_D2.append(featureCoeff[3])

    return featureCoeff_A2, featureCoeff_H2, featureCoeff_V2, featureCoeff_D2


# function to create matrix and graphs
def createMatrixandGraphs(featureCoeff_A2, featureCoeff_H2, featureCoeff_V2, featureCoeff_D2):
    dict = {'Approximate Coefficient': featureCoeff_A2, 'Horizontal Detailed Coefficient':featureCoeff_H2, 'Vertical Detailed Coefficient':featureCoeff_V2, 'Diagonal Detailed Coefficient':featureCoeff_D2}
    feature_Matrix = pd.DataFrame(dict)

    #Normalizing data
    for column in feature_Matrix.columns:
        feature_Matrix[column] = (feature_Matrix[column] - feature_Matrix[column].min()) / (feature_Matrix[column].max() - feature_Matrix[column].min())

    #creating a directory for saving correlation coefficicnets
    path = r'./data_processed/correlations.txt'

    #export DataFrame to text file
    with open(path, 'a') as f:
        df_string = feature_Matrix.corr().to_string(header=False, index=False)
        f.write(df_string)


    # saving figures in a directory
    dir_name = './visuals/'
    plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

    #plotting and saving graphs
    plt.scatter(feature_Matrix['Approximate Coefficient'], feature_Matrix['Horizontal Detailed Coefficient'], c='green')
    plt.ylabel('Approximate Coefficient')
    plt.xlabel('Horizontal Detailed Coefficient')
    plt.title("ScatterPlot 1")
    plt.savefig("ScatterPlot1.png")
    plt.clf()

    plt.scatter(feature_Matrix['Approximate Coefficient'], feature_Matrix['Vertical Detailed Coefficient'], c='blue')
    plt.ylabel('Approximate Coefficient')
    plt.xlabel('Vertical Detailed Coefficient')
    plt.title("ScatterPlot 2")
    plt.savefig("ScatterPlot2.png")
    plt.clf()

    plt.scatter(feature_Matrix['Approximate Coefficient'], feature_Matrix['Diagonal Detailed Coefficient'], c='red')
    plt.ylabel('Approximate Coefficient')
    plt.xlabel('Digital Detailed Coefficient')
    plt.title("ScatterPlot 3")
    plt.savefig("ScatterPlot3.png")
    plt.clf()

    plt.scatter(feature_Matrix['Vertical Detailed Coefficient'], feature_Matrix['Horizontal Detailed Coefficient'], c='pink')
    plt.ylabel('Vertical Detailed Coefficient')
    plt.xlabel('Horizontal Detailed Coefficient')
    plt.title("ScatterPlot 4")
    plt.savefig("ScatterPlot4.png")
    plt.clf()

    plt.scatter(feature_Matrix['Vertical Detailed Coefficient'], feature_Matrix['Diagonal Detailed Coefficient'], c='black')
    plt.ylabel('Vertical Detailed Coefficient')
    plt.xlabel('Digital Detailed Coefficient')
    plt.title("ScatterPlot 5")
    plt.savefig("ScatterPlot5.png")
    plt.clf()

    plt.scatter(feature_Matrix['Diagonal Detailed Coefficient'], feature_Matrix['Horizontal Detailed Coefficient'], c='purple')
    plt.ylabel('Diagonal Detailed Coefficient')
    plt.xlabel('Horizontal Detailed Coefficient')
    plt.title("ScatterPlot 6")
    plt.savefig("ScatterPlot6.png")
    plt.clf()




