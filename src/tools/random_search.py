#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:54:55 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.tools.cross_validation import CrossValidation
from src.tools.utils import Logger


class RandomHyperParameterTuningPerKernel(Logger):
    def __init__(self, dataset, clf, kernel, parameter_grid, n_sampling,
                 kfold=5, verbose=True):

        self.verbose = verbose
        self.clf = clf
        self.kernel = kernel
        self.parameter_grid = parameter_grid
        self.dataset = dataset
        self.n = n_sampling
        self.kfold = kfold
        # self.kernel_parameters = 
        # name = clf.__name__
        self.parameters = kernel.param

    def fit(self):
        self.kernel_parameters = {}
        self.clf_parameters = {}
        self.scores = {}
        for param in self.kernel.param:
            self.kernel_parameters[param] = self.parameter_grid[param].rvs(size=self.n)
        for parameter_name in self.clf_parameters:
            self.parameters[parameter_name] = self.parameter_grid[
                parameter_name].rvs(self.n)

        for j in range(self.n):
            temp_parameters = {'kernel': self.kernel}
            for parameter_name in self.kernel.param:
                temp_parameters[parameter_name] = self.parameters[
                    parameter_name][j]
            for parameter_name in self.clf_parameters:
                temp_parameters[parameter_name] = self.parameters[
                    parameter_name][j]
            temp_clf = self.clf(**temp_parameters)
            CV = CrossValidation(self.dataset, temp_clf, k_fold=self.k_fold)
            temp_report = CV.stats
            self.scores[j] = temp_report


class RandomHyperParameterTuning(Logger):
    def __init__(self,
                 classifier,
                 kernels,
                 dataset,
                 n_sampling,
                 parameter_grid,
                 criteria='accuracy',
                 kfold=5,
                 verbose=True):

        self.verbose = verbose

        self.kernels = kernels
        
        self.clf = classifier
        self.parameter_grid = parameter_grid
        self.dataset = dataset
        self.n = n_sampling
        self.kfold = kfold
        
        if criteria == 'accuracy':
            self.criteria = 'mean_acc'
        elif criteria == 'recall':
            self.criteria = 'mean_recall'
        elif criteria == 'precision':
            self.criteria = 'mean_precision'
        elif criteria == 'f1':
            self.criteria = 'mean_f1'
            
        self.kernels = parameter_grid['kernel']

    def fit(self):
        self.accuracy = {}
        self.parameters = {}
        for kernel in self.kernels:
            temp = RandomHyperParameterTuningPerKernel(
                self.clf,
                kernel,
                self.parameter_grid,
                self.X,
                self.y,
                self.n,
                k_fold=self.n_fold)
            temp.fit()
            scores = np.array([
                temp.scores[j][self.criteria] for j in range(len(temp.scores))
            ])
            argmax_parameters = np.argmax(scores)
            param = []
            for l in list(temp.parameters.values()):
                param.append(l[argmax_parameters])
            self.parameters[kernel] = dict(
                zip(list(temp.parameters.keys()), param))
            self.accuracy[kernel] = scores[argmax_parameters]

    def best_parameters(self):
        argmax_kernel = np.argmax(np.array(list(self.accuracy.values())))
        best_kernel = self.kernels[argmax_kernel]
        best_parameters = self.parameters[best_kernel]
        best_parameters['kernel'] = best_kernel
        return best_parameters, self.accuracy[best_kernel]

if __name__ == '__main__' :
    
    from src.data.seq import SeqData
    from scipy.stats import uniform
    from scipy.stats import randint
    
    data = SeqData(k=0, dataname="train", mat=False, small=False, verbose=True)
    
    parameter_grid = {'kernel' : ['spectral', 'mismatch'],
                      'k' : randint(low = 3, high = 9),
                      'm' : randint(low = 1, high = 3),
                      'C' : uniform(loc = 0.1, scale = 10)
                      }
    