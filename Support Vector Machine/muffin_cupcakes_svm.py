# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:37:48 2018

@author: Ashish Chouhan
"""
#Libraries for Analysis

import pandas as pd
import numpy as np
from sklearn import svm

#Libraries for Visulas
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

#Create a function to guess when a  recipe is a muffin 
# or a cupcake using the SVM model  we created

def muffin_or_cupcake(butter, sugar):
    if(model.predict([[butter, sugar]]))==0:
        print('You\'re looking at a muffin receipe!')
    else:
        print('You\'re looking at a Cupcake receipe!')
        
#Read the dataset
recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
recipes.head()

#Plot two ingredients
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s":70})

#Specify inputs for the model
flour_sugar = recipes[['Flour', 'Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', 0 ,1)

#Fit the SVM Model
#SVC - Support Vector Classifier
#SVR - Support Vector Regressor

model = svm.SVC(kernel='linear', decision_function_shape='none')
model.fit(flour_sugar, type_label)

#Visualize Results 
#Get the Separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

#Plot the parallels to the separating hyerplane
#that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[1]
yy_up = a * xx + (b[1] - a * b[0])

#Get the Separating and Support Vectors
sns.lmplot('Flour','Sugar', data=recipes, hue= 'Type',
           palette='Set1', fit_reg=False, scatter_kws={"s":70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:,1],
            s=80, facecolors='none')

#Plot the hyperplane
sns.lmplot('Flour','Sugar', data=recipes, hue= 'Type',
           palette='Set1', fit_reg=False, scatter_kws={"s":70})
plt.plot(xx, yy, linewidth=2, color='black')

#Predict New Case

#Plot the point to visually see where the point  lies
sns.lmplot('Flour','Sugar', data=recipes, hue= 'Type',
           palette='Set1', fit_reg=False, scatter_kws={"s":70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9')

#Predict if 12 parts butter and sugar 
muffin_or_cupcake(40,20)
