import pandas as pd

data= pd.read_csv(r"train2.csv",index_col=0)#stratifiedshufflesplit

X = data.iloc[:,1:]
y = data.iloc[:,0]

test=pd.read_csv(r"test2.csv",index_col=0)#stratifiedshufflesplit
X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = preprocessing.MinMaxScaler().fit(X)#
# scaler = preprocessing.StandardScaler().fit(X)#
X_data_transformed = scaler.transform(X)
X_data_transformed=pd.DataFrame(X_data_transformed)
X_data_transformed.columns=X.columns
X=X_data_transformed

scaler = preprocessing.MinMaxScaler().fit(X_test)#####测试集归一化test
# scaler = preprocessing.StandardScaler().fit(X_test)
X_test_transformed = scaler.transform(X_test)
X_test_transformed=pd.DataFrame(X_test_transformed)
X_test_transformed.columns=X_test.columns
X_test=X_test_transformed

import numpy as np
x_cols=[col for col in X.columns if X[col].dtype!='object']
# print('x_cols:',x_cols)
labels=[]
values=[]
for col in x_cols:
    labels.append(col)
    values.append(abs(X[col].corr(y,'spearman')))#np.corrcoef(X[col].values,y.values)[0,1]

corr_df=pd.DataFrame({'col_labels':labels,'corr_values':values})

#
features = corr_df['col_labels']
feature_matrix = X[features]
corr_matrix =feature_matrix.corr(method='spearman')#method='spearman'
remove_features = []
mask = (corr_matrix.iloc[:,:].values>0.95) & (corr_matrix.iloc[:,:].values<1)
for idx_element in range(len(corr_matrix.columns)):
    for idy_element in range(len(corr_matrix.columns)):
        if mask[idx_element,idy_element]:
#             print(idx_element,idy_element)
            if list(corr_df['corr_values'])[idx_element] > list(corr_df['corr_values'])[idy_element]:
                remove_features.append(list(features)[idy_element])
            else:
#                 print(list(features)[idx_element])
                remove_features.append(list(features)[idx_element])
remove_features = set(remove_features)
print(len(remove_features))

print(type(features))
print(type(remove_features))
remain_features = set(features) - remove_features
print(remain_features)
X = X.loc[:,remain_features]
data1 = pd.concat([y,X],axis=1)
X_test = X_test.loc[:,remain_features]
test1 = pd.concat([y_test,X_test],axis=1)

test1.to_csv('test2.csv')
data1.to_csv('train2.csv')




