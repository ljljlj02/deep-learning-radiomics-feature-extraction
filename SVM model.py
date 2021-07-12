from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.feature_selection import  RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import precision_score
train = pd.read_csv(r"\train.csv",index_col=0)#stratifiedshufflesplit
test= pd.read_csv(r"\test.csv",index_col=0)#stratifiedshufflesplit

Xtrain = train.iloc[:,1:]
Ytrain = train.iloc[:,0]
# print(Ytrain)
Xtest=test.iloc[:,1:]
Ytest=test.iloc[:,0]
# print(Ytest)

#Cross validation
NUM_RUNS = 1
NUM_JOBS=8
seed=12345 #

classifier = svm.SVC(random_state=seed)
gamma_range = np.logspace(-10, 1, 50)
C_range=np.arange(1,40,1)
# C_range=np.linspace(1,20,40)
parameters_grid = {
    'C':C_range,  # Penalty parameter C of the error term

    'kernel': ['linear','poly','rbf','sigmoid'],  # different type of kernels to be explored
    'gamma': gamma_range,
    'degree':[1, 2, 3, 4]  # Degree of the polynomial kernel function. Other kernels will ignore it
}
# # Arrays to store scores
grid_search_best_scores = np.zeros(NUM_RUNS)  # numpy arrays
final_evaluation_scores = np.zeros(NUM_RUNS)  # numpy arrays
# estimator: estimator object for GridSearchCV
# Loop for each trial
for i in range(NUM_RUNS):
    folds_for_grid_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    folds_for_evaluation = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    tuned_model = GridSearchCV(estimator=classifier
                               , param_grid=parameters_grid
                               # ,C=4.333333333333333
                               , cv=folds_for_grid_search
                               ,scoring='roc_auc'
                               ,n_jobs=NUM_JOBS)
    tuned_model.fit(Xtrain,Ytrain)
    grid_search_best_scores[i] = tuned_model.best_score_
    print()
    print("Best Selected Parameters:")
    print(tuned_model.best_params_)
    print('clf.best_score_:', tuned_model.best_score_)  #
    y_pred = cross_val_predict(tuned_model.best_estimator_,Xtrain,Ytrain, cv=folds_for_evaluation)
    print()
    print("Classification Results")
#     print(classification_report(Ytrain, y_pred))

#modeling
clf = SVC(
    # kernel="sigmoid"
    # , degree=1
    # , gamma=0.15998587196060574
    # ,
    # C=4
    # ,
    # random_state=12345
             ).fit(Xtrain,Ytrain)
result = clf.predict(Xtest)
score = clf.score(Xtest,Ytest)
recall = recall_score(Ytest, result)
auc = roc_auc_score(Ytest,clf.decision_function(Xtest))


