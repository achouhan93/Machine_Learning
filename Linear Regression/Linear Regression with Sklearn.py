# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:57:54 2018

@author: Ashish Chouhan
Linear Regression Using Sklearn - Linear Regression
"""
#Simple Linear Regression

#Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('anscombe.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set Reults
y_pred = regressor.predict(X_test)

#Visualising the Training Set Results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Size of House (Training Set)')
plt.xlabel('Size of House')
plt.ylabel('House Price')
plt.show()

#Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Size of House (Test Set)')
plt.xlabel('Size of House')
plt.ylabel('House Price')
plt.show()
