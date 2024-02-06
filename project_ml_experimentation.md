#### SER594: Experimentation
#### Title: Face Recognition System for Identifying required Subjects (Celebrities)
#### Pete Rohan Kanaparthi
#### 11/16/2023


## Explainable Records
### Record 1
**Raw Data:** For this experiment raw data is given in the form of images of Cristiano ronaldo available in './test_images/

Prediction Explanation:** The model which provided the best results for this project(Random Forest) is used to predict the images of test subjects provided and the model produced good results. The model was able to predict the test subjects given in the folder, the output can be seen in evaluation.txt. Since this is a Face Recognition project the chances of getting close to 95% success rate is low. The model was able to give 9 correct labels out of 14 provided

### Record 2
**Raw Data:** For this experiment raw data is given in the form of images of Leo Messi available in './test_images/. 

Prediction Explanation:** The model which provided the best results for this project(Random Forest) is used to predict the images of test subjects provided and the model produced good results as seen in evaluation.txt. The model was able to predict the test subjects given in the folder, the output can be seen in evaluation.txt.Since this is a Face Recognition project the chances of getting close to 95% success rate is low. The model was able to give 9 correct labels out of 14 provided

## Interesting Features
### Feature A
**Feature:** Wavelet Transform of each Picture

**Justification:** Wavelet transforms and normal image are combined to create a single image with both normal and wavelet pixelings. The image values are then read into an array. This gives the required values of pixels for botht the images. These values can then be trained using an ML model whch can then be used for predictions.

### Feature B
**Feature:** Eyes and Face

**Justification:** The underlying feature that play a huge role in determining a face are facial lines. These features can be obtained by running the data through haarcascade xml files (haarcascade_frontalface_default.xml, haarcascade_profileface.xml). These features then help us in identifying only faces from the images thereby reducing the amount of unwanted data processing. Another underlying feature that play a huge role in determining a face are the eyes. Similar to the previous step these features can be obtained by running the data through haarcascade xml file (haarcascade_eye.xml).

## Experiments 
### Varying : Eyes
**Prediction Trend Seen:** This is the most important feature as there can be two eyes ot just one eye in the picture based on the subjects positioning. based on this a haarcascade xml file is called to check for facial features. Thsi can either make or break the model

### Varying : Face
**Prediction Trend Seen:** This is the most important feature in determinig the face strucure and the boundaries that might be required in maintianng while cropping the image. If unnecessary features are also added because of improper assessment of the image. The model will not be able to give better predictions. 

### Varying Eye and Face together
**Prediction Trend Seen:** Both the features are required together in determining the face of a person. The cropped image will have better results because of this


### Varying Eye and Face inversely
**Prediction Trend Seen:** Both are required or the face cannot be identified. 

