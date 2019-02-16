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

clfs = []
for i in range(3) :
    parameter_grid = { 'kernel' : ['gaussian'],
                       'C' : scipy.stats.uniform(loc = 1, scale = 100),
                       'gamma' : scipy.stats.uniform(loc = 0,scale = 2),
                       'dim' : scipy.stats.randint(1,5),
                       'offset' : scipy.stats.randint(1,2)
                     }
    n_sampling = 20

    clf = svm.SVM
    X = x_train_mat[i].drop('Id',axis = 1).values
    y = y_train_mat[i]['Bound'].values
    random_search_svm = random_search.RandomHyperParameterTuning(clf, parameter_grid, X, y, n_sampling)
    random_search_svm.fit()

    parameters,acc = random_search_svm.best_parameters()
    print('Best parameter found for dataset {}:'.format(i), parameters)
    print('with average accuracy :',acc)




#%%




