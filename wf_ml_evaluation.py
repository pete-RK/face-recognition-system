import wf_core as cr
import wf_ml_training as tr
import wf_prediction as pr
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# running wf_core to get the required variables
sportsCelebDict = cr.sportsCelebDict
pathProcessedImages = cr.pathProcessedImages

# creating X, y variables to train a model
X, y, classDict = tr.createDataVals(sportsCelebDict, pathProcessedImages)

# splitting data into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

svm_Score, svm_report = tr.trainSVMModel(X_train, X_test, y_train, y_test)
rf_score, rf_report = tr.tarinRandomForestClassifierModel(X_train, X_test, y_train, y_test)
lr_score, lr_report = tr.tarinLogisticRegressionModel(X_train, X_test, y_train, y_test)

#printing results
print(f"SVM Score: {svm_Score}, Random Forest Score: {rf_score}, Logistical Regression Score:{lr_score}")
print(f"SVM Report: {svm_report}, Random Forest Report: {rf_report}, Logistical Regression Report:{lr_report}")

# creating and saving in path
path = r'./evaluation/summary.txt'

modelScores = {'SVM Score': svm_Score, 'Random Forest Score' :rf_score, 'Logistical Regression Score':lr_score} 
modelReport = {'SVM Report': svm_report, 'Random Forest Report' :rf_report, 'Logistical Regression Report':lr_report}  
  
with open(path,'w') as data:  
      data.write(str(modelScores))
      data.write(str(modelReport))

# predicting sample
pr.predictSample()

#running visuals
cr.runningVisuals()



