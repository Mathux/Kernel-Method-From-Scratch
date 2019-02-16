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
import pickle

import scipy.stats


''' TO DO :
        
             - 3 sections to train each classifier with a random hyper param
               search and cross-validation with report on mean accuracy on the
               validation set after splitting
             
             - Une section averaging pour average les classifieurs
             
             - Une section predictions pour creer le fichier de predictions
'''

# Load kernel matrices for the spectral kernel

name = 'spectral_kernel_3.pickle'
file = open(name,'rb')
kernel_matrices = pickle.load(file)
file.close()

# Load the data as embeddings

x_train_mat,y_train_mat = utils.load_train(mat = True)
x_test_mat = utils.load_test(mat = True)

# Load the data as DNA sequences

x_train,y_train = utils.load_train(mat = False)
x_test = utils.load_test(mat = False)

#%%

## Optimize a svm classifier for train_mat and train with a random hyperparamter search

# First the embeddings case :

clfs_embeddings = []
for i in range(3) :
    
    parameter_grid = { 'kernel' : ['gaussian','polynomial','linear'],
                       'C' : scipy.stats.uniform(loc = 25, scale = 100),
                       'gamma' : scipy.stats.uniform(loc = 0,scale = 1),
                       'dim' : scipy.stats.randint(1,5),
                       'offset' : scipy.stats.randint(1,2)
                     }
    n_sampling = 10

    clf = svm.SVM
    X = x_train_mat[i].drop('Id',axis = 1).values
    y = y_train_mat[i]['Bound'].values
    y = utils.transform_label(y)
    random_search_svm = random_search.RandomHyperParameterTuning(clf, parameter_grid, X, y, n_sampling)
    random_search_svm.fit()

    parameters,acc = random_search_svm.best_parameters()
    print('Best parameter found for dataset {}:'.format(i+1), parameters)
    print('with average accuracy :',acc)
    clfs_embeddings.append(svm.SVM(**parameters))


#%%
# Then train a SVM with a string adapted kernel

clfs_strings  =[]
for i in range(3) :
    
    parameter_grid = { 'kernel' : ['spectral'],
                       'C' : scipy.stats.uniform(loc = 0, scale = 2),
                     }
    n_sampling = 1

    clf = svm.SVM
    X = x_train[i].drop('Id',axis = 1).values
    y = y_train[i]['Bound'].values
    y = utils.transform_label(y)
    K = kernel_matrices[i]
    random_search_svm = random_search.RandomHyperParameterTuning(clf, parameter_grid, X, y, n_sampling, kernel_matrix = K)
    random_search_svm.fit()

    parameters,acc = random_search_svm.best_parameters()
    print('Best parameter found for dataset {}:'.format(i), parameters)
    print('with average accuracy :',acc)
    clfs_strings.append(svm.SVM(**parameters))
    


#%%
    
## Next we train a Kernel Logistic Regression and a Kernel KNearestNeighbors and optimize them
    
clfs_klr = []
clfs_knn = []
for i in range(3) :
    
    parameter_grid_klr = { 'kernel' : ['gaussian','polynomial','linear'],
                       'la' : scipy.stats.uniform(loc = 25, scale = 100),
                       'gamma' : scipy.stats.uniform(loc = 0,scale = 1),
                       'dim' : scipy.stats.randint(1,5),
                       'offset' : scipy.stats.randint(1,2)
                     }
    parameter_grid_knn = { 'kernel' : ['gaussian','polynomial','linear'],
                       'n_neighbors' : scipy.stats.randint(loc = 5, scale = 15),
                       'gamma' : scipy.stats.uniform(loc = 0,scale = 1),
                       'dim' : scipy.stats.randint(1,5),
                       'offset' : scipy.stats.randint(1,2)
                     }
    n_sampling = 10

    clf_klr = kernel_lr.KernelLogisticRegression
    clf_knn = kernel_knn.KernelKNN
    X = x_train_mat[i].drop('Id',axis = 1).values
    y = y_train_mat[i]['Bound'].values
    y = utils.transform_label(y)
    random_search_klr = random_search.RandomHyperParameterTuning(clf_klr, parameter_grid_klr, X, y, n_sampling)
    random_search_knn = random_search.RandomHyperParameterTuning(clf_knn, parameter_grid_knn, X, y, n_sampling)
    random_search_klr.fit()
    random_search_knn.fit()

    parameters_klr,acc_klr = random_search_klr.best_parameters()
    parameters_knn,acc_knn = random_search_knn.best_parameters()
    print('Best parameter found for dataset {} :'.format(i+1), parameters_klr, 'and classifier : KernelLogisticRegression')
    print('with average accuracy :',acc_klr)
    print('Best parameter found for dataset {} :'.format(i+1), parameters_knn, 'and classifier : KernelKNN')
    print('with average accuracy :',acc_knn)
    clfs_klr.append(kernel_lr.KernelLogisticRegression(**parameters_klr))
    clfs_knn.append(kernel_knn.KernelKNN(**parameters_knn))





