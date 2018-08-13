# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:33:15 2018

@author: Ashish Chouhan
Support Vector Machine Basic Code
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

#Create 40 separable points
X, y = make_blobs(n_samples=100, centers=3, random_state=20)

#fit the model 
clf = svm.SVC(kernel='rbf' , C=1, gamma = 2**-5)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1] , c=y, s=30, cmap=plt.cm.Paired)
#plt.show()
#Plot show clears out the plot so we have to reload the data 

#Plot the decision function
#From pyplot ax is the axis and with the help of get_xlim and get_ylim we get the 
#x and y limits
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#Create grid to evaluate model
#xx and yy will be an array and line space will have value between 0 and 1 and divided into 30 points
#meshgrid with the help of array gathered above will plot a mesh grid on a plot
#Till now all points are in XX and YY format so to be in XY format we used vstack
# Z is the classifier with the 3 values which will tell its support vector or hyperplane

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

#Plot decision boundary and margins
#this will plot the contour with the XX, YY and Z classifier also the line styple are specified
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1],
           alpha=0.5,
           lnestyles=['--','-','--'])

#Plot support vectors
#All support vector points are plotted on each side of the Hyperplane
ax.scatter(clf.support_vectors_[:,0],
           clf.support_vectors_[:,1],s=100,
           linewidth=1, facecolors='none')

#Using to predict unknown data
#newData = [[3,4],[5,6]]
#print(clf.predict(newData))

plt.show()




