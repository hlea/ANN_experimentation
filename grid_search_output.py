# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:10:35 2018

@author: hlea
"""

import sys
sys.path.append(r'C:\\Users\\hlea\\Documents\\Career Development\\ANN_Exper_Code\\')

from Cars_NN_Testing import train_nn, create_data_set, predict_y
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, average_precision_score 
import pandas as pd

def grid_search():
    out = []
    X_train, X_test, y_train, y_test = create_data_set()
    
    nn_structures = [[21, 10, 1], [21, 10, 10, 1], [21, 15, 1], [21, 15, 15, 1]] 
    act_functions =['sigmoid', 'tanh']
    num_iterations = [5000, 10000]

    for n in nn_structures:
        for a in act_functions:
            for i in num_iterations:
                if len(n) == 4:
                    ident_model = a+"_"+str(n[1])+"_"+str(n[2])+"_"+str(i)
                else:
                    ident_model = a+"_"+str(n[1])+"_"+str(i)
                W, b, avg_cost_func = train_nn(n, a, X_train, y_train, iter_num=i, alpha=0.1)
                f = plt.figure()
                plt.plot(avg_cost_func)
                plt.ylabel('Average Cost')
                plt.xlabel('Iteration number')
                plt.show()
                file_name = "C:\\Users\\hlea\\Documents\\Career Development\\ANN_test_output\\output_"+ident_model+".pdf"
                f.savefig(file_name)
                
                y_pred = predict_y(W, b, X_test, len(n), a)
                acc_score = accuracy_score(y_test,y_pred)*100
                print("Accuracy is ", acc_score)
                
                average_precision = average_precision_score(y_test, y_pred)
                
                print('Average precision-recall score: {0:0.2f}'.format(
                      average_precision))
                out.append([ident_model, acc_score, average_precision])
    
    df = pd.DataFrame(out, columns = ['Model Iteration', 'Accuracy Score', 'Average Precision-Recall Score'])
    df.to_csv("C:\\Users\\hlea\\Documents\\Career Development\\ANN_test_output\\brute_force_grid_search.csv")

if __name__ == "__main__":
    grid_search()