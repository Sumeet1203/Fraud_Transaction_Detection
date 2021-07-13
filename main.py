#Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

#Import dataset
df = pd.read_csv('fraud.csv')

#Remove columns which are not required
df1 = df.drop(columns = ['isFlaggedFraud','type', 'nameOrig', 'nameDest'])

#Assign columns to variables
x = df1.iloc[:,0:6].values
y = df1.iloc[:, 6].values

#Initialize label encoder and onehotencoder
le = LabelEncoder()
enc = OneHotEncoder()

#Create an array t (since payment type is categorical, we need to use onehotencoding) and apply onehotencoder on it
t = df.iloc[:, 1].values
t = le.fit_transform(t)
t= t.reshape(-1,1)

onehotlabels = enc.fit_transform(t).toarray()

#Combine one hot encoded array and rest of database
db = np.concatenate([x,onehotlabels], axis = 1)

#Split dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Scale amount, account balance using MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,10))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#----------------------------------Logistic Regression-----------------------------------
lr_model = LogisticRegression(penalty= 'l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
lr_model.fit(x_train, y_train)

pred_train_lr = lr_model.predict(x_train)
pred_test_lr = lr_model.predict(x_test)

#Evaluation Metrics

#Accuracy
print('\nLogistic Regression')

acc_train = accuracy_score(y_train, pred_train_lr)
print('\nAccuracy (Training):', round(acc_train*100, 2), '%')

acc_test = accuracy_score(y_test, pred_test_lr)
print('Accuracy (Testing):', round(acc_test*100, 2), '%')

#Precision
pre_train = precision_score(y_train, pred_train_lr)
print('\nPrecision (Training):', round(pre_train, 2))

pre_test = precision_score(y_test, pred_test_lr)
print('Precision (Testing):', round(pre_test, 2))

#Recall
recall_train = recall_score(y_train, pred_train_lr)
print('\nRecall score (Training):', round(recall_train, 2))

recall_test = recall_score(y_test, pred_test_lr)
print('Recall score (Testing):', round(recall_test, 2))

#F1 score
f1_train = f1_score(y_train, pred_train_lr)
print('\nF1 score (Training):', round(f1_train, 2))

f1_test = f1_score(y_test, pred_test_lr)
print('F1 score (Testing):', round(f1_test, 2))

#Confusion Matrix
cm_train = confusion_matrix(y_train, pred_train_lr)
print('\nConfusion Matrix (Training):\n', cm_train)

cm_test = confusion_matrix(y_test, pred_test_lr)
print('Confusion Matrix (Testing):\n',cm_test)


#--------------------------------K-Nearest Neighbours-------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree').fit(x_train,y_train)

#Getting predictions from model
pred_train_knn = knn_model.predict(x_train)
pred_test_knn = knn_model.predict(x_test)

#Evaluation Metrics

#Accuracy
print('\nK-Nearest Neighbours(KNN)')

acc_train = accuracy_score(y_train, pred_train_knn)
print('\nAccuracy (Training):', round(acc_train*100, 2), '%')

acc_test = accuracy_score(y_test, pred_test_knn)
print('Accuracy (Testing):', round(acc_test*100, 2), '%')

#Precision
pre_train = precision_score(y_train, pred_train_knn)
print('\nPrecision (Training):', round(pre_train, 2))

pre_test = precision_score(y_test, pred_test_knn)
print('Precision (Testing):', round(pre_test, 2))

#Recall
recall_train = recall_score(y_train, pred_train_knn)
print('\nRecall score (Training):', round(recall_train, 2))

recall_test = recall_score(y_test, pred_test_knn)
print('Recall score (Testing):', round(recall_test, 2))

#F1 score
f1_train = f1_score(y_train, pred_train_knn)
print('\nF1 score (Training):', round(f1_train, 2))

f1_test = f1_score(y_test, pred_test_knn)
print('F1 score (Testing):', round(f1_test, 2))

#Confusion Matrix
cm_train = confusion_matrix(y_train, pred_train_knn)
print('\nConfusion Matrix (Training):\n', cm_train)

cm_test = confusion_matrix(y_test, pred_test_knn)
print('Confusion Matrix (Testing):\n',cm_test)
