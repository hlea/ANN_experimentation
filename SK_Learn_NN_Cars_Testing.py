# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:02:43 2018

@author: hlea
"""

'''
This code creates a baseline ANN (MLP) model performance by using the scikit-learn
neural network modual, outlined here: 
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
'''
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, average_precision_score 


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

'''Train and evaluate model'''
def train_score_ann(layers, act_func, solv):
    #import data
    X_train, X_test, y_train, y_test = create_data_set()
    
    #set hyper parameters

    clf = MLPClassifier(layers, activation=act_func, solver=solv,
                            alpha = 0.2, shuffle=True)

        
    #fit model
    clf.fit(X_train, y_train)
    
    #score and evaluate model
    y_pred = clf.predict(X_test)
    average_precision = average_precision_score(y_test, y_pred)
    
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)       
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))

        
if __name__ == "__main__":
  
    train_score_ann((10,),'tanh', 'sgd')
    train_score_ann((10,), 'relu', 'adam')

   


