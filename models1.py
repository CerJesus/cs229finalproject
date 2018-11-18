from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
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

def test_svm_predict(x):
    test_pred = clf.predict(test_x)
    test_x["predictions"] = test_pred
    test_x.to_csv("ssi_pressure_test_pred.csv")

x,y=get_features_labels("ssi_pressure_labels.csv")

k = 10

#Random Forest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rfscores = cross_val_score(rf, x, y, cv = k)
print("Random Forest Cross Validation Scores for k = ", k, ": ", rfscores)




clf = SVC()


#Cross Validation for SVM
scores = cross_val_score(clf, x, y, cv = k)
print("Cross Validation Scores for k = ", k, ": ", scores)
clf.fit(x,y)




#Run SVM model on a different time range
test_x,test_y = get_features_labels("ssi_pressure_test1.csv")
test_svm_predict(test_x)
