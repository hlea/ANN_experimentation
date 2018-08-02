# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:52:54 2018

@author: hlea
"""

import sys
sys.path.append(r'C:\\Users\\hlea\\Documents\\Career Development\\ANN_Exper_Code\\')

from Cars_NN_Testing import predict_y, f, f_deriv, setup_and_init_weights, init_tri_values, feed_forward, calculate_hidden_delta, calculate_out_layer_delta
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, average_precision_score 
import pandas as pd
import sys


            
def data_sets_for_shuffle():
    '''
    Create training and testing set originally from UCI Car Evaluation dataset:
        https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    Exact copy of dataset currently stored in following project github directory as Cars_Sample_DataSet_UCI.txt:
        https://github.com/hlea/ANN_experimentation
    Output: Two reproducible dataframes (i.e., one for training, one for testing) that include both inputs and labels 
    '''
    url = 'https://raw.github.com/hlea/ANN_experimentation/master/Cars_Sample_DataSet_UCI.txt'
    cars = pd.read_csv(url)
    
    cars['label']=cars['class'].apply(lambda x: make_label(x))
    
    cars_input = cars[['buying', 'maint','persons', 'doors', 'lug_boot', 'safety', 'label']]
    cars_input_dum = pd.get_dummies(cars_input)
    
    key = pd.Series(range(0,cars_input_dum.shape[0]), name = 'key')
    
    cars_key = cars_input_dum.join(key)
    
    test_df =  cars_key.loc[lambda df: df['key']% 5== 1]
    train_df =  cars_key.loc[lambda df: df['key']% 5 != 1]
    
    return test_df, train_df
    

def make_label(x):
   if x== 'unacc':
       return 0
   return 1

def df_to_array(df, batch_size, data='train'):
    
    input_lst = ['buying_high', 'buying_low', 'buying_med', 'buying_vhigh',
       'maint_high', 'maint_low', 'maint_med', 'maint_vhigh', 'persons_2',
       'persons_4', 'persons_more', 'doors_2', 'doors_3', 'doors_4',
       'doors_5more', 'lug_boot_big', 'lug_boot_med', 'lug_boot_small',
       'safety_high', 'safety_low', 'safety_med']
    
    if data == 'train':
        shuffled = df.sample(n = batch_size)
        y = shuffled['label'].values
        X = shuffled[input_lst].values[:,:]
    else:
        y = df['label'].values
        X = df[input_lst].values[:,:]
    return y, X


def train_nn_shuffle(nn_structure, func, df_train, mode='train', batch_size = 1000, iter_num=3000, alpha=0.2):
    W, b = setup_and_init_weights(nn_structure)
    #pdb.set_trace()  # -- added breakpoint
    cnt = 0
    m = batch_size
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        #resample training sample
        if cnt%100 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
            y, X = df_to_array(df_train, batch_size, mode)
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
        if cnt%100 ==99:
            print(str(cnt))
    return W, b, avg_cost_func



if __name__ == "__main__":

    test_df, train_df = data_sets_for_shuffle()
    nn_structures = [[21, 10, 1], [21, 10, 10, 1], [21, 15, 1], [21, 15, 15, 1]] 
    act_functions =['sigmoid', 'tanh']
    num_iterations = [5000, 10000]
    batch_size = [500]
    out=[]
    for n in nn_structures:
        for a in act_functions:
            for i in num_iterations:
                for b in batch_size:
                    if len(n) == 4:
                        ident_model = a+"_"+str(n[1])+"_"+str(n[2])+"_"+str(i)+"_"+str(b)
                    else:
                        ident_model = a+"_"+str(n[1])+"_"+str(i)+"_"+str(b)
                    W, b, avg_cost_func = train_nn_shuffle(n, a, train_df, mode='train', batch_size = b, iter_num=i, alpha=0.1)
                    
                    f = plt.figure()
                    plt.plot(avg_cost_func)
                    plt.ylabel('Average Cost')
                    plt.xlabel('Iteration number')
                    plt.show()
                    file_name = "C:\\Users\\hlea\\Documents\\Career Development\\ANN_test_output\\output_shuffle_"+ident_model+".pdf"
                    f.savefig(file_name)
                    
                    y_test, X_test = df_to_array(test_df, test_df.shape[0], data='test')
                    y_pred = predict_y(W, b, X_test, len(n), a)
                    acc_score = accuracy_score(y_test,y_pred)*100
                    print("Accuracy is ", acc_score)
                    
                    average_precision = average_precision_score(y_test, y_pred)
                    
                    print('Average precision-recall score: {0:0.2f}'.format(
                          average_precision))
                    out.append([ident_model, acc_score, average_precision])

df = pd.DataFrame(out, columns = ['Iteration', 'Accuracy Score', 'Average Precision'])
df.to_csv("C:\\Users\\hlea\\Documents\\Career Development\\ANN_test_output\\shuffle_brute_force_grid_search.csv")




