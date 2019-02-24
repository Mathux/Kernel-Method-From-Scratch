#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:17:56 2019

@author: evrardgarcelon
"""

import utils
import svm
import kernel_lr
import kernel_knn
import random_search
import voting_classifier
import scipy.stats


# Load the data as embeddings

#x_train_mat, y_train_mat = utils.load_train(mat = True)
#x_train_mat, y_train_mat = utils.process_data(x = x_train_mat, y = y_train_mat)
#x_test_mat = utils.load_test(mat = True)
#x_test_mat = utils.process_data(x = x_test_mat)

# Load the data as DNA sequences

x_train,y_train = utils.load_train(mat = False)
x_test = utils.load_test(mat = False)

x_train,y_train = utils.process_data(x_train,y = y_train)
x_test = utils.process_data(x_test)



## Optimize a classifier for train with a random hyperparamter search 

clfs = [svm.SVM, kernel_lr.KernelLogisticRegression]
grid_base = {'kernel' : ['spectral','mismatch'],
               'k' : scipy.stats.randint(3,8),
               'm' : scipy.stats.randint(1,3),
               'C' : scipy.stats.uniform(loc = 0, scale = 20),
               'la' : scipy.stats.uniform(loc = 0, scale = 20),
               'n_iter' : scipy.stats.randint(500,501)}

clfs_strings = []
n_sampling = 10

for j in range(len(clfs)) :
    clf = clfs[j]
    print('Searching parameters for : ', clf.__name__)
    for i in range(len(x_train)) :
        print('Fitting Dataset : ', i+1)
        parameter_grid = grid_base
        X = x_train[i]
        y = y_train[i]
        random_search_clfs = random_search.RandomHyperParameterTuning(clf, parameter_grid, X, y, n_sampling, n_fold = 3)
        random_search_clfs.fit()    
        parameters,acc = random_search_clfs.best_parameters()
        print('Best parameter found for Dataset {} and classifier {} : '.format(i+1,clf.__name__), parameters,'with average accuracy :',acc)
        clfs_strings.append(clf(**parameters))   
        
        
# Finally before making final predictions, we average predictions of each tra-
# -ined classifiers and create a submission file
    
    
    
predictions = []
for i in range(len(x_train)) :
    X = x_train[i]
    y = y_train[i]
    X_test = x_test[i]
    base_clfs = []
    for j in range(len(clfs)) : 
        clfs_strings[j].fit(X,y)
        base_clfs.append(clfs_strings[j])
    voting_clf = voting_classifier.VotingClassifier(base_clfs, 
                                                    hard_pred = True)
    predictions.append(voting_clf.predict(X_test))

utils.submission(predictions)
