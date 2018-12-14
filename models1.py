from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)



def get_features_labels(file):

    training_data = pd.read_csv(file, index_col = [0])
    training_data = pd.DataFrame(training_data, columns = ["time", "temperature", "pressure", "press_change", "var", "label"])

    #the last three pressure readings
    training_data["press_change_1"] = training_data["press_change"].shift(1)
    training_data["press_change_2"] = training_data["press_change"].shift(2)
    training_data["press_change_3"] = training_data["press_change"].shift(3)
    training_data["press_change_4"] = training_data["press_change"].shift(4)
    training_data["press_change_5"] = training_data["press_change"].shift(5)

    training_data = training_data.dropna(axis=0)
    x = training_data[["temperature", "pressure", "press_change", "press_change_1", "press_change_2", "press_change_3", "press_change_4", "press_change_5", "var"]]
    y = training_data["label"]
    return(x,y)

def test_predict(model, features):
    test_pred = model.predict(features)
    features["predictions"] = test_pred
    #features.to_csv("ssi_pressure_test_pred.csv")
    print("Counts: ", features.groupby("predictions").count()["pressure"])
    return features["predictions"]



x,y=get_features_labels("ssi_pressure_labels.csv")

k = 10


#Logistic regression

lr = LogisticRegression()
lrscores = cross_val_score(lr, x, y, cv=k)
print("Logistic Regression Cross Validation Scores for k = ", k, ": ", lrscores)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
#print(test_x.columns)
#test
lr.fit(x,y)

#Prediction accuracy on train set
print("Logistic Regression score on train data: ", lr.score(x, y))
test_predict(lr,x)

#Predictions on test set (labeled based on altitude)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print("Logistic Regression score on val data (altitude-based labels): ", lr.score(test_x, test_y))
test_predict(lr,test_x)

#Predictions on test set (labeled based on pressure)
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
print("Logistic Regression score on val data (pressure-based labels): ", lr.score(test_x, test_y))
print(lr.coef_)
theta = np.zeros(len(lr.coef_[0]))
for ind in range(len(lr.coef_[0])):
    theta[ind] = lr.coef_[0][ind]
plot(test_x.values,test_y.values,theta,"logregdecisionboundary.png")
test_predict(lr,test_x)


x,y=get_features_labels("ssi_pressure_labels.csv")

maxCV = 0.17
"""
maxCV = 0.0
maxScore = 0.0
for cv in np.arange(0.01,1.01,0.01):
    logr = LogisticRegression(penalty="l1",C=cv)
    logr.fit(x,y)
    val_x,val_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
    curScore = logr.score(val_x,val_y)
    if curScore > maxScore:
        maxScore = curScore
        maxCV = cv
print("Best C value for Logistic Regression is: ", maxCV, "with a score of: ", maxScore)
"""

x,y=get_features_labels("ssi_pressure_labels.csv")
finlr = LogisticRegression(C=maxCV)
finlr.fit(x,y)
test_x, test_y = get_features_labels("ssi_pressure_testfin_badpressure.csv")
print("Logistic Regression score on test data (pressure-based labels): ", finlr.score(test_x, test_y))
lrPressurePreds = test_predict(finlr,test_x)


#SVM
clf = SVC()
#Cross Validation for SVM
x,y=get_features_labels("ssi_pressure_labels.csv")
scores = cross_val_score(clf, x, y, cv = k)
print("SVM Cross Validation Scores for k = ", k, ": ", scores)
clf.fit(x,y)
#Prediction accuracy on train set
print("SVM score on train data: ", clf.score(x, y))
test_predict(clf,x)

#Predictions on test set (labeled based on altitude)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print("SVM score on test data (altitude-based labels): ", clf.score(test_x, test_y))
test_predict(clf,test_x)

#Predictions on test set (labeled based on pressure)
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
print("SVM score on test data (pressure-based labels): ", clf.score(test_x, test_y))
# print(clf.coef_)
# theta = np.zeros(len(clf.coef_[0]))
# for ind in range(len(clf.coef_[0])):
#     theta[ind] = clf.coef_[0][ind]
# plot(test_x.values,test_y.values,theta,"svcdecisionboundary.png")
svmPressurePreds = test_predict(clf,test_x)

x,y=get_features_labels("ssi_pressure_labels.csv")


maxCV = 0.0
maxScore = 0.0
for cv in np.arange(0.01,1.01,0.01):
    svr = SVC(C=cv)
    svr.fit(x,y)
    test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
    curScore = svr.score(test_x,test_y)
    if curScore > maxScore:
        maxScore = curScore
        maxCV = cv
print("Best C value for SVC is: ", maxCV, "with a score of: ", maxScore)


x,y=get_features_labels("ssi_pressure_labels.csv")

#LinearSVC
lsvc = LinearSVC(C=0.17)
#Cross Validation for SVM
x,y=get_features_labels("ssi_pressure_labels.csv")
scores = cross_val_score(lsvc, x, y, cv = k)
print("LinearSVC Cross Validation Scores for k = ", k, ": ", scores)
lsvc.fit(x,y)
#Prediction accuracy on train set
print("LinearSVC score on train data: ", lsvc.score(x, y))
test_predict(lsvc,x)

#Predictions on test set (labeled based on altitude)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print("LinearSVC score on val data (altitude-based labels): ", lsvc.score(test_x, test_y))
test_predict(lsvc,test_x)

#Predictions on test set (labeled based on pressure)
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
print("LinearSVC score on val data (pressure-based labels): ", lsvc.score(test_x, test_y))
print(lsvc.coef_)
theta = np.zeros(len(lsvc.coef_[0]))
for ind in range(len(lsvc.coef_[0])):
    theta[ind] = lsvc.coef_[0][ind]
plot(test_x.values,test_y.values,theta,"linearsvcdecisionboundary.png")
test_predict(lsvc,test_x)

x,y=get_features_labels("ssi_pressure_labels.csv")

#maxCV = 0.24
maxCV = 0.0
maxScore = 0.0
for cv in np.arange(0.01,1.01,0.01):
    svr = LinearSVC(C=cv)
    svr.fit(x,y)
    test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
    curScore = svr.score(test_x,test_y)
    if curScore > maxScore:
        maxScore = curScore
        maxCV = cv
print("Best C value for LinearSVC is: ", maxCV, "with a score of: ", maxScore)

x,y=get_features_labels("ssi_pressure_labels.csv")
finlsvc = LinearSVC(C=maxCV)
finlsvc.fit(x,y)
test_x, test_y = get_features_labels("ssi_pressure_testfin_badpressure.csv")
print("LinearSVC score on test data (pressure-based labels): ", finlsvc.score(test_x, test_y))
svmPressurePreds = test_predict(finlsvc,test_x)

x,y=get_features_labels("ssi_pressure_labels.csv")
#Random Forest
rf = RandomForestClassifier(n_estimators = 100)
rfscores = cross_val_score(rf, x, y, cv = k)
print("Random Forest Cross Validation Scores for k = ", k, ": ", rfscores)
rf.fit(x,y)

print("Random Forest score on train data: ", rf.score(x, y))
test_predict(rf,x)

x,y=get_features_labels("ssi_pressure_labels.csv")
#Predictions on test set (labeled based on altitude)
test_x,test_y = get_features_labels("ssi_pressure_test1_badaltitude.csv")
print("Random Forest score on test data (altitude-based labels): ", rf.score(test_x, test_y))
test_predict(rf,test_x)

#Predictions on test set (labeled based on pressure)
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")
print("Random Forest score on test data (pressure-based labels): ", rf.score(test_x, test_y))
test_predict(rf,test_x)

x,y=get_features_labels("ssi_pressure_labels.csv")
test_x,test_y = get_features_labels("ssi_pressure_test1_badpressure.csv")

maxNumTrees = 0
maxScore = 0.0
for numtrees in np.arange(100,1001,100):
    currf = RandomForestClassifier(n_estimators = numtrees)
    currf.fit(x,y)
    curScore = currf.score(test_x,test_y)
    if curScore > maxScore:
        maxScore = curScore
        maxNumTrees = numtrees
print("Best n_estimators value for RandomForestClassifier is: ", maxNumTrees, "with a score of: ", maxScore)


x,y=get_features_labels("ssi_pressure_labels.csv")
finrf = RandomForestClassifier(n_estimators = maxNumTrees)
finrf.fit(x,y)
test_x, test_y = get_features_labels("ssi_pressure_testfin_badpressure.csv")
print("RandomForestClassifier score on test data (pressure-based labels): ", finrf.score(test_x, test_y))
rfPressurePreds = test_predict(finrf,test_x)



xaxis = range(len(rfPressurePreds))



plt.figure(figsize=(12, 8))
plt.title("Predictions of Bad Readings")
plt.xlabel("Time")
plt.ylabel("Bad Reading Detected")
for i in range(5,len(svmPressurePreds)):
    if rfPressurePreds[i] == 0:
        rfPressurePreds[i] = 0.2
    if lrPressurePreds[i] == 0:
        lrPressurePreds[i] = 0.4

plt.plot(xaxis, lrPressurePreds,"r.", label = "Logistic Regression")
plt.plot(xaxis, svmPressurePreds,"b.", label = "SVM")
plt.plot(xaxis, rfPressurePreds,"g.", label = "Random Forest")
plt.plot(xaxis, test_y,"c", label = "True Label")
plt.legend()
plt.savefig("predictionplot.png")
