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
train = pd.read_csv(r"\xx.csv",index_col=0)#stratifiedshufflesplit
test= pd.read_csv(r"\xx.csv",index_col=0)#stratifiedshufflesplit

Xtrain = train.iloc[:,1:]#第一列是标签，取出所有特征
Ytrain = train.iloc[:,0]
# print(Ytrain)
Xtest=test.iloc[:,1:]
Ytest=test.iloc[:,0]
# print(Ytest)
#feature selection
#select feature numbers
#Recursive feature elimination
score = []
LR_ = SVC(
    kernel='linear'
    , probability=True
    ,
    random_state=12345
)
for i in range(1, 20, 1):
    X_wrapper = RFE(LR_ , n_features_to_select=i).fit_transform(Xtrain,Ytrain)
    once = cross_val_score(LR_ , X_wrapper,Ytrain, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1),scoring='roc_auc').mean()
    # once = cross_val_score(LR_, X_wrapper, Ytrain, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)).mean()
    score.append(once)
plt.figure(figsize=[10, 5])
plt.plot(range(1, 20, 1), score)
plt.xticks(range(1, 20, 1))
# plt.yticks(range(0.7, 1, 0.02))
plt.xlabel('Number of features')
plt.ylabel('AUC')
# plt.ylabel('Accuracy')
plt.grid(True)
plt.show()#
#
svc = SVC(
    kernel="linear"
    , random_state=1234

          )
Select = RFE(estimator=svc, n_features_to_select=n)#n is your quantity of feature selection
X_embedded = Select.fit_transform(Xtrain,Ytrain)
X2=Xtrain.T
Xtrain=X2[Select.get_support()].T

Xtest=Xtest.T
Xtest=Xtest[Select.get_support()].T
