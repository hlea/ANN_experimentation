# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:04:28 2018

@author: hlea
"""

import csv
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, average_precision_score 
from sklearn import tree


def create_data_set():
    '''
    Create training and testing set originally from UCI Car Evaluation dataset:
        https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    Exact copy of dataset currently stored in following project github directory as Cars_Sample_DataSet_UCI.txt:
        https://github.com/hlea/ANN_experimentation
    '''
    url = 'https://raw.github.com/hlea/ANN_experimentation/master/Cars_Sample_DataSet_UCI.txt'
    cars = pd.read_csv(url)
    
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


def make_lable(x):
    '''This is a convenience function that transforms the multi-class label into a binary lable
    Logic: If the car is 'unacceptable', then 0; else if it's labelled as 'acceptable', 'good', or 'very good', then 1'''
    if x== 'unacc':
        return 0.0
    return 1.0

'''Train and evaluate model'''
def fit_score_tree():
    X_train, X_test, y_train, y_test = create_data_set()
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)
    average_precision = average_precision_score(y_test, y_pred)

    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


if __name__ == "__main__":
    fit_score_tree()



