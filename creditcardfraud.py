# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:24:05 2020

@author: HP
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')

data.info()

data.isnull().sum()

#1 is fraud 0 is not fraud
data['Class'].value_counts()
data['Class'].value_counts(normalize = True)


class_0, class_1 = data.Class.value_counts()

# Divide by class
d_class_0 = data[data['Class'] == 0]
d_class_1 = data[data['Class'] == 1]

#RANDOM UNDER-SAMPLING

d_class_0_under = d_class_0.sample(class_1)
d_test_under = pd.concat([d_class_0_under, d_class_1], axis=0)

print('Random under-sampling:')
print(d_test_under.Class.value_counts())

d_test_under.Class.value_counts().plot(kind='bar', title='Count (Class)');

#RANDOM OVER-SAMPLING

d_class_1_over = d_class_1.sample(class_0, replace=True)
d_test_over = pd.concat([d_class_0, d_class_1_over], axis=0)

print('Random over-sampling:')
print(d_test_over.Class.value_counts())

d_test_over.Class.value_counts().plot(kind='bar', title='Count (Class)');

d1=data.sample(frac=0.2, replace=True, random_state=1)
d1.columns
fraud = d_test_over.loc[data['Class'] == 1]
not_fraud = d_test_over.loc[data['Class'] == 0]

fraud.describe()
not_fraud.describe()
sns.scatterplot(x="Time", y="Amount", data=fraud)

sns.scatterplot(x="Time", y="Amount", data=not_fraud)

X=d_test_over.drop(['Class'], axis=1)
y=d_test_over["Class"]

#Principle Component Analysis
from sklearn.decomposition import PCA
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
pca = PCA(n_components=2)
X = pca.fit_transform(X)
plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(expected, predicted))

from sklearn import svm
from sklearn.svm import SVC
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1,gamma= 10).fit(X_train, y_train)

expected = y_test
predicted = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(expected, predicted)

from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(222)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

from sklearn.metrics import accuracy_score
accuracy_score(expected,predicted)





































