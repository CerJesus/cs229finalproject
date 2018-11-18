from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
import numpy as np

def get_features_labels(file):

    training_data = pd.read_csv(file, index_col = [0])
    training_data = pd.DataFrame(training_data, columns = ["time", "temperature", "pressure", "press_change", "label"])

    #the last three pressure readings
    training_data["press_1"] = training_data["pressure"].shift(1)
    training_data["press_2"] = training_data["pressure"].shift(2)
    training_data["press_3"] = training_data["pressure"].shift(3)

    training_data = training_data.dropna(axis=0)
    x = training_data[["temperature", "pressure", "press_change", "press_1", "press_2", "press_3"]]
    y = training_data["label"]
    return(x,y)

def test_predict(model, features):
    test_pred = model.predict(features)
    features["predictions"] = test_pred
    #features.to_csv("ssi_pressure_test_pred.csv")
    print("Counts: ", features.groupby("predictions").count()["pressure"])

x,y=get_features_labels("ssi_pressure_labels.csv")

k = 10

#Logistic regression
lr = LogisticRegression()
lrscores = cross_val_score(lr, x, y, cv=k)
print("Logistic Regression Cross Validation Scores for k = ", k, ": ", lrscores)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print(test_x.columns)
#test
lr.fit(x,y)
print("Logistic Regression score on test data (altitude-based labels): ", lr.score(test_x, test_y))
test_predict(lr,test_x)

#SVM
clf = SVC()
#Cross Validation for SVM
scores = cross_val_score(clf, x, y, cv = k)
print("Cross Validation Scores for k = ", k, ": ", scores)
clf.fit(x,y)

#Prediction accuracy on train set
print("SVM score on train data: ", clf.score(x, y))
test_predict(clf,x)

#Predictions on test set (labeled based on altitude)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print(test_x.columns)
print("SVM score on test data (altitude-based labels): ", clf.score(test_x, test_y))
test_predict(clf,test_x)

#Predictions on test set (labeled based on pressure)
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
print(test_x.columns)
print("SVM score on test data (pressure-based labels): ", clf.score(test_x, test_y))
test_predict(clf,test_x)

#Random Forest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rfscores = cross_val_score(rf, x, y, cv = k)
print("Random Forest Cross Validation Scores for k = ", k, ": ", rfscores)
