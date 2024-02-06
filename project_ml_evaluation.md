#### SER594: Machine Learning Evaluation
#### Title: Face Recognition System for Identifying required Subjects (Celebrities)
#### Pete Rohan Kanaparthi
#### 11/16/2023

## Evaluation Metrics: SVM Model
### Metric 1
**Name:** Precision

**Choice Justification:** As per the definition the higher the precision the better the model, therefore to justify the use of a model it is logical to use precision. In this case precision gives, how effectively the model produced the results in identifying face of a person in images.

### Metric 2
**Name:** Recall

**Choice Justification:** Recall gives you the rate of true positive, this implies that if the recall is having a good value, we will be getting very good results. In this case recall provides, how effectively the model produced the results in identifying face of a person in images and assigning the proper class.

## Alternative Models: RandomForestClassifier 
### Alternative N
**Construction:** This model took the same daat that was used for SVM but it differes in it's execution of classifying the images. Basically it creates a decision tree for evaluation y labels that are acquired from the model.predict fucntion and y_test.

**Evaluation:** This model provided a better score than SVM. SVM gave a score of 0.72 where as Random Forest gave a score of 0.90 but this doesn't imply that random forest will always work better than SVM for all cases. In this project since only two different classes of data were given, Random Forest was able to perform a little bit better.

(duplicate above three times; remove this line when done)


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


## Best Model

**Model:** Since SVM works on creating vectors for classifying data, the data is divided based on various tactics like Linear Seperation, Kernel Trick for Non-Linear, etc. Therefore for most of face recognition projects, SVM is considered. But in this project Random Forest gave better results compared to SVM so Random Forest is considered for running the classification for test images provided in './test_images'.