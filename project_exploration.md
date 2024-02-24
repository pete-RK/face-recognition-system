Exploratory Data Munging and Visualization
#### Title: Face Recognition System for Identifying required Subjects (Celebrities)
#### Pete Rohan Kanaparthi
#### 10/16/2023

## Basic Questions
**Dataset Author(s):** Google Images

**Dataset Construction Date:** 11/16/2023

**Dataset Record Count:** 210 Images (109 of Cristiano and 101 of Messi)

**Dataset Field Meanings:** Individual Images of a celebrity (Cristiano Ronaldo dos Santos Aveiro) and (Lionel Messi)

**Dataset File Hash(es):** Downloaded images of celebrities (Cristiano Ronaldo dos Santos Aveiro and Leo Messi)using fatkun Google Chrome extension for bulk image download, from these links:
Cristiano URL : https://www.google.com/search?q=cristiano+ronaldo&client=firefox-b-1-d&sca_esv=574621129&channel=fen&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjc16ebkYGCAxWxH0QIHdNvB64Q_AUoAnoECAIQBA&biw=1130&bih=749&dpr=2
Messi URL : https://www.google.com/search?q=leo+messi&client=firefox-b-1-d&sca_esv=584201750&tbm=isch&source=lnms&sa=X&ved=2ahUKEwj-zPLAsdSCAxWFOUQIHe7wDMkQ_AUoAnoECAIQBA&biw=1680&bih=965&dpr=2
MD5 Hash for the Link(Cristiano) : fceb7d458a6854a71a9a0ff73b075f15
MD5 Hash for the Link(Messi) : b8d2f46f01b4cc5a4402fa7aa05576a2


**Additional Files:** For image processing this project requires additional files that are available in: https://github.com/opencv/opencv/tree/master/data/haarcascades. The folder consists of haarcascade XML files that are used for face and eyes detection for an image using opencv-python package. You can access the file from './data_original/haarcascades'

## Interpretable Records
### Record 1
**Raw Data:**  The data are images of Cristiano Ronaldo and Leo Messi(test subjects) particularly of their face. This data is then used to run through algorithms to identify underlying features like eyes face shape and other facial features.

**Interpretation:** The underlying feature that play a huge role in determining a face are facial lines. These features can be obtained by running the data through haarcascade xml files (haarcascade_frontalface_default.xml, haarcascade_profileface.xml). These features then help us in identifying only faces from the images thereby reducing the amount of unwanted data processing.

### Record 2
**Raw Data:** The data are images of Cristiano Ronaldo and Leo Messi(test subjects) particularly of their face. This data is then used to run through algorithms to identify underlying features like eyes face shape and other facial features.

**Interpretation:** Another underlying feature that play a huge role in determining a face are the eyes. Similar to the previous step these features can be obtained by running the data through haarcascade xml file (haarcascade_eye.xml).


## Background Domain Knowledge
The project deals with identifying celebs(test subject) amoung other in a picture with multiple faces(including that of our test subject). The images are downloaded for the test subject are downloaded from google, using a google chrome extension for bulk image download called fatkun, and are stored in data_original folder. 

For this project additional files, called haarcascadeXML, are also required. These XML files help us in identifying regions of interest in an image. These regions of interest are face or faces and eyes for each face. Each image is run through an algorithm to identify regions of interest and return a cropped image containing a face and eyes, rest of the images with missing eyes or face are discarded. These cropped images are then stored in a folder called data_processed [1][3]. 

We now have cropped images with face and eyes, how can we identify this celebrity amoung others..? For this we need to understand wavelet transformations. Wavelet transformations are mathematical tools that help us in analyzing data where features vary over different scales. In simpler terms wavelets transform decompose the image features into frequencies which can then be used to extract hidden features of an image to better understand how normal features vary. The decomposed features can then be reconstructed to get the image in wavelet form [2].

For this project we will be extracting features from decomposed wavelet transform of each image. We will look into the approximate coefficient "cA2" and  detailed coefficients for horizontal, vertical and diagonal axis "cH2, cV2, cD2". These values are then used to extract the hidden features in an image. This way we can understand how the image features vary over minute levels. 

By following these steps and running a training model we can extract important features that can help us in identifying the test subject amoung others. 


Resources:
1. Face Recognition System: https://towardsdatascience.com/face-recognition-for-beginners-a7a9bd5eb5c2
2. Image Processing: https://neptune.ai/blog/image-processing-python
3. Hard Cascades(OpenCV): https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html



## Data Transformation
### Transformation 1
**Description:** Raw data used to extract features is an image that consists of a face with either one or two eyes. This is identified using haarcascade xmls, for frontal face: "haarcascade_frontalface_default.xml", for profile face: "haarcascade_profileface.xml" and "haarcascade_eye.xml" for eyes. The image is then sent into './data_processed/{celebrity_name}' folder.

**Soundness Justification:** To identify a face we have to look at the facial features and the eyeys of a person for this reason it's required to extract these regions of interest for better results.

### Transformation 2
**Description:** The data obtained from wavelet transformation for approximate coefficinet cA2 should be normalized to meet the range of other coefficients.

**Soundness Justification:** It is a standard procedure that helps in getting a better visual representation of the data


## Visualization
### Visual 1
**Analysis:** This graph is a scatter plot between the maximum mean for Approximate Coefficient and Horizontal Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are. 

### Visual 2
**Analysis:** This graph is a scatter plot between the maximum mean for Approximate Coefficient and Vertical Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are.

### Visual 3
**Analysis:** This graph is a scatter plot between the maximum mean for Approximate Coefficient and Digital Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are.

### Visual 4
**Analysis:** This graph is a scatter plot between the maximum mean for Vertical Detailed Coefficient and Horizontal Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are.

### Visual 5
**Analysis:** This graph is a scatter plot between the maximum mean for Vertical Detailed Coefficient and Digital Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are.

### Visual 6
**Analysis:** This graph is a scatter plot between the maximum mean for Digital Detailed Coefficient and Horizontal Detailed Coefficient obtained from image processing using pywavelets. Each plot shows how various coefficinets vary between each other. The more denser the plot is the more correlated the coefficinets are.
