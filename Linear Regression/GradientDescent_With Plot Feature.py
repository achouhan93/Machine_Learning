# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:42:31 2018

@author: Ashish Chouhan
"""

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

import numpy as np
from matplotlib import pyplot as plt

# y = mx + b
# m is slope, b is y-intercept

def plot_lines(m,b,data_points):
    x_values = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
    y_values = [(m*x + b) for x in x_values]
    plt.plot(x_values, y_values, 'r')
    
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (1/N) * (((m_current * x) + b_current) - y)
        m_gradient += +(1/N) * x * (((m_current * x) + b_current) - y)
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    a= []
    
    for i in range(0, len(points)):
        a.append(points[i,0])
    
    data_points = np.array(a).tolist()
            
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            plt.plot(x,y,'bo')
            
        plot_lines(m,b,data_points)
        
    return [b, m]

def run():
    points = np.genfromtxt("anscombe.csv", delimiter=",")
        
    learning_rate = 0.001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 50
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()