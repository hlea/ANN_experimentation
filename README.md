Experimentation with Natively Coded ANN

Author: Halsey Lea

Date: August, 2018


Research Question: Can a simple, low-level neural network outperform a high-level API decision tree model through hyper-parameter tuning?

Objectives: Determine if hyper-parameter testing will result in a natively-coded ANN model that outperforms a Scikit-learn decision tree (i.e., by overall accuracy); hyper-parameters tested include
 - Activation function
 - Number and size of hidden layers 
 - Number of iterations

Technical details:
 - Modeling problem: binary classification, predicting whether a car is “acceptable” for purchase
 - Data: Car Evaluation data set (1728 labeled records, 6 attributes) from UCI Machine Learning Repository*
 - Language: Python 3.6 (Anaconda 4.4 distribution)
 
Algorithms/packages:
 - Scikit-learn: DecisionTreeCalssifier, MLPClassifier 
 - Natively coded neural network, leveraged from Adventures in Machine Learning blog**

*Data source: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
**Source code leveraged from following blog post: http://adventuresinmachinelearning.com/neural-networks-tutorial/ 

**********************************************************************************************************************************

To run the baseline Scikit-learn decision tree model:
 - DT_Cars_Testing.py

To run the grid search for the natively-coded ANN model: 
 - grid_search_output.py

To run the Scikit-learn MLPClassifier model:
 - SK_Learn_NN_Cars_Testing.py

To run a single instance of the natively-coded ANN model for demo purposes:
 - Cars_NN_Testing.py
