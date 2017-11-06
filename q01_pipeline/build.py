# %load q01_pipeline/build.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
#from imblearn.over_sampling import SMOTE

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')
label_enc = LabelEncoder()
for column in bank.select_dtypes(include=["object"]).columns.values:
    bank[column] = label_enc.fit_transform(bank[column])

#bank.replace({0:1, 1:0}, inplace=True)
# Write your solution here :
y=bank['y']
X=bank.iloc[:,:-1]
#print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=9)

model = RandomForestClassifier(random_state=9,class_weight='balanced')


def pipeline(X_train,X_test,y_train,y_test,model):

    param_grid = {
              "n_estimators": [10, 50, 120],
              "max_depth": [40, 20, 10],
              "max_leaf_nodes": [5, 10, 2]}

    grid_obj = GridSearchCV(model, param_grid,cv=5)
    grid_obj.fit(X_train,y_train)
    y_pred=grid_obj.predict(X_test)
    acc_score= roc_auc_score(y_test,y_pred)
    return grid_obj, acc_score.item()




# grid_model,  auc_score= pipeline(X_train,X_test,y_train,y_test,model)
# print(auc_score)
# bank_test = pd.read_csv('data/Bank_data_to_test.csv')
# y = bank_test['y']
# X = bank_test.drop(['y'], axis=1)
# prediction = grid_model.predict_proba(X)[:, 1]
# #y = bank_test['y']
# print(type(auc_score))

# auc_score_test = roc_auc_score(y, prediction)
# print(auc_score_test)
