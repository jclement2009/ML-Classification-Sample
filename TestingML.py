#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:35:13 2017

@author: Joseph
"""
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import sys
import scipy
import sklearn

# Main Imports
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Getting the data from folder.  Will update this to be more flexible in the near future.  For now, we shall use this.
mainrawdata = "/Users/Joseph/Sample-Data-Swiss-Bank-Notes.csv" 

maindata = pd.read_csv(mainrawdata)


print(maindata.shape)

df = pd.DataFrame(maindata)

for column in df.loc[:, df.columns != 'Genuine/Counterfeit']:
    df[column] = df[column].map(lambda x: x.lstrip('Quantity[').rstrip(', "Millimeters"]'))
    
for column in df.loc[:, df.columns != 'Genuine/Counterfeit']:
    df[column] = pd.to_numeric(df[column])

print(df)


#This gives the counts of types recorded in the data set
print(df.groupby('Genuine/Counterfeit').size())

#sns_plot2 = sns.pairplot(data=df, hue="Genuine/Counterfeit", vars =['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal'])
#sns_plot2.savefig("PairplotSwiss.png")

# Split-out validation dataset
transition = df.values
X = transition[:,0:6]
Y = transition[:,6]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

#

# Make predictions on validation dataset
myList = list(range(1,50))
cv_scores = []


# subsetting just the odd ones
neighbors = myList

for k in neighbors:
    knn1 = KNeighborsClassifier(n_neighbors=k)
    scores = model_selection.cross_val_score(knn1, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is " + str(optimal_k))

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=optimal_k)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

 # Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Testing KNN for optimal number of neighbors


#More Stuff  
knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#testarray= np.array([215.7, 130.2, 130.0, 8.7,  10.0, 141.6])
testarray= np.array([0, 0, 0, 0,  0, 0])
print(knn.predict(testarray.reshape(1,-1)))

svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(svc.predict(testarray.reshape(1,-1)))


