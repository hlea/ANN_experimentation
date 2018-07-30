# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:20:13 2018

@author: hlea
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score 
import matplotlib.pyplot as plt 
import numpy.random as r
import pdb

def create_data_set():
    '''
    Create training and testing set from UCI Car Evaluation dataset:
        https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    '''
    cars = pd.read_csv(
    'C:\\Users\\hlea\\Documents\\Portfolio Fit\\Audit\\Cars_Sample_DataSet_UCI.txt',
                               sep= ',')
    
    #separate input vars and convert to dummies
    cars_input = cars[['buying', 'maint','persons', 'doors', 'lug_boot', 'safety']]
    cars_input_dum = pd.get_dummies(cars_input)
    X = cars_input_dum.values[:,:]
    
    
    cars['label']=cars['class'].apply(lambda x: make_lable(x))
    Y = cars.values[:,7]
    Y = Y.astype('int')
    
    #create training and testing partitions
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
    return X_train, X_test, y_train, y_test

    #transform label
def make_lable(x):
    '''This is a convenience function that transforms the multi-class label into a binary lable
    Logic: If the car is 'unacceptable', then 0; else if it's labelled as 'acceptable', 'good', or 'very good', then 1'''
    if x== 'unacc':
        return 0.0
    return 1.0

'''''''''''''''''''''''''''''''''''''''''''''
Create NN Functions
'''''''''''''''''''''''''''''''''''''''''''''

#define activation functions

def f(x, func):
    if func == 'sigmoid':
        out =  1 / (1 + np.exp(-x))
    else:
        out = np.tanh(x)
    return out
    
def f_deriv(x, func):
    if func == 'sigmoid': 
        out = f(x, 'sigmoid') * (1 - f(x, 'sigmoid'))
    else:
        out = 1.0 - np.tanh(x)**2
    return out

#initialize weights between layers to random numbers
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b
 
#set delta Weight and delta bias to 0    
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

#feed forward function
def feed_forward(x, W, b, func):
    #pdb.set_trace()  # -- added breakpoint
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise, 
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1], func) # h^(l) = f(z^(l)) 
    return h, z


def calculate_out_layer_delta(y, h_out, z_out, func):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out, func)

def calculate_hidden_delta(delta_plus_1, w_l, z_l, func):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l, func)


def train_nn(nn_structure, func, X, y, iter_num=3000, alpha=0.2):
    W, b = setup_and_init_weights(nn_structure)
    #pdb.set_trace()  # -- added breakpoint
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%100 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b, func)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i], h[l], z[l], func)
                    avg_cost += np.linalg.norm((y[i]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l], func)
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

'''''''''''''''''''''''''''''''''''''''
measure accuracy of model
'''''''''''''''''''''''''''''''''''''''

def predict_y(W, b, X, n_layers, func):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b, func)
        y[i] = np.argmax(h[n_layers])
    return y



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_data_set()
    nn_structure = [21, 10, 1]
    act_function ='sigmoid'
    W, b, avg_cost_func = train_nn(nn_structure, act_function, X_train, y_train, iter_num=1000)
    
    plt.plot(avg_cost_func)
    plt.ylabel('Average Cost')
    plt.xlabel('Iteration number')
    plt.show()

    y_pred = predict_y(W, b, X_test, len(nn_structure), act_function)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
    
    average_precision = average_precision_score(y_test, y_pred)
    
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))


