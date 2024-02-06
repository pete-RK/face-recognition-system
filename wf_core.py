import wf_dataprocessing as dp
import wf_visualization as vs
import os


# Calling to create image directories
imgDir, pathProcessedImages = dp.createDirectories()

# Calling function to create cropped image folder with cropped images
croppedImageFolder, sportsCelebDict = dp.storeCroppedData(imgDir, pathProcessedImages)

# Calling function to extract features
for directory in croppedImageFolder:
    featureCoeff_A2, featureCoeff_H2, featureCoeff_V2, featureCoeff_D2 = vs.extractingFeatures(directory)
   
# Calling to create plots nad matrix
def runningVisuals():
    vs.createMatrixandGraphs(featureCoeff_A2, featureCoeff_H2, featureCoeff_V2, featureCoeff_D2)

