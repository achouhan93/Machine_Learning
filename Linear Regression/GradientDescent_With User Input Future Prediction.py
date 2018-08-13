# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:42:31 2018
@author: Ashish Chouhan
Linear Regression with Gradient Descent without Sklearn 
"""

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# y = mx + b
# m is slope, b is y-intercept
class GradientDescent():
    
    def __init__(self,learning_rate,num_iterations, starting_b, starting_m):
        self.lr = learning_rate
        self.num_itr = num_iterations
        self.initial_b = starting_b
        self.initial_m = starting_m
    
    def compute_error_for_line_given_points(self,b, m, points):
        totalError = 0
        
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            totalError += (y - (m * x + b)) ** 2
        
        return totalError / float(len(points))
    
    def plot_training_graph(self,points):
        b = self.initial_b
        m = self.initial_m
        a= []
        
        for i in range(0, len(points)):
            a.append(points[i,0])
            
        data_points = np.array(a).tolist()
        
        for i in range(self.num_itr):
            b, m = self.step_gradient(b, m, np.array(points), self.lr)
            self.plot_lines(m,b,data_points)
        
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            plt.plot(x,y,'bo')
            
        plt.title('House Price vs Size of House (Training Set)')
        plt.xlabel('Size of House (100 Sq ft)')
        plt.ylabel('House Price (1000 Euro)')
        plt.show()
        
        return [b, m]

    def plot_lines(self,m,b,data_points):
        x_values = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
        y_values = [(m*x + b) for x in x_values]
        plt.plot(x_values, y_values, 'r')    
        
    def step_gradient(self,b_current, m_current, points, learningRate):
        b_gradient = 0
        m_gradient = 0
        N = float(len(points))
        
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            b_gradient += (1/N) * (((m_current * x) + b_current) - y)
            m_gradient += (1/N) * x * (((m_current * x) + b_current) - y)
            
        new_b = b_current - (learningRate * b_gradient)
        new_m = m_current - (learningRate * m_gradient)
        return [new_b, new_m]         
    
    def plot_test_graph(self,points,m,b):
        a =[]
        
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            plt.plot(x,y,'bo')
            
        for i in range(0, len(points)):
            a.append(points[i,0])
    
        data_points = np.array(a).tolist()        
        self.plot_lines(m,b,data_points)
        
        plt.title('House Price vs Size of House (Test Set)')
        plt.xlabel('Size of House(100 Sq ft')
        plt.ylabel('House Price (1000 Euro)')
        plt.show()
        
    def usercheck(self,m,b):
        
        print("Algorithm is being trained with the provided data")
        while(True):
            houseSize = input("Kindly enter the Size of House in Sq ft : ")
            housePrice = m * (int(houseSize)/100) + b
            print("Predicted House Price = " + str(housePrice * 1000) + " Euro")
            exitkey = input("To exit press 1 or press Enter to continue -> ")
            if(exitkey == '1'):
                break
        

def run():
    
    #Importing the dataset
    dataset = pd.read_csv('anscombe.csv')
    X = dataset.iloc[:,0].values
    Y = dataset.iloc[:,1].values
    
    #Splitting the dataset into the training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
    
    learning_rate = 0.001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 200
    
    gd_calculation = GradientDescent(learning_rate,num_iterations, initial_b, initial_m)
    
    points = np.vstack((X_train,Y_train)).T
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, gd_calculation.compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gd_calculation.plot_training_graph(points)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, gd_calculation.compute_error_for_line_given_points(b, m, points)))
    
    points = np.vstack((X_test,Y_test)).T
    gd_calculation.plot_test_graph(points, m, b)  
    
    gd_calculation.usercheck(m,b)
    
if __name__ == '__main__':
    run()