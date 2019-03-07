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
        self.kernel_parameters = list(kernel.defaultParameters.keys())
        self.clf_parameters = list(clf.defaultParameters.keys())
        try : 
            self.kernel_parameters.remove('tol')
        except :
            pass
        try :
            self.clf_parameters.remove('tol')
        except :
            pass
        
    def fit(self):
        self.kernel_parameters_to_test = {}
        self.clf_parameters_to_test = {}
        self.scores = {}
        for param in self.kernel_parameters :
            self.kernel_parameters_to_test[param] = self.parameter_grid[param].rvs(
                size=self.n)
        for parameter_name in self.clf_parameters :
            self.clf_parameters_to_test[parameter_name] = self.parameter_grid[
                parameter_name].rvs(self.n)

        for j in range(self.n):
            temp_clf_parameters = {}
            temp_kernel_parameters = {}
            for parameter_name in self.kernel_parameters:
                temp_kernel_parameters[parameter_name] = self.kernel_parameters_to_test[
                    parameter_name][j]
            for parameter_name in self.clf_parameters:
                temp_clf_parameters[parameter_name] = self.clf_parameters_to_test[
                    parameter_name][j]
            temp_kernel  = self.kernel(self.dataset, parameters = temp_kernel_parameters)
            temp_clf = self.clf(temp_kernel, parameters = temp_clf_parameters)
            CV = CrossValidation(self.dataset, temp_clf, kfolds=self.kfold)
            temp_report = CV.stats
            self.scores[j] = temp_report


class RandomHyperParameterTuning(Logger):
    def __init__(self,
                 classifier,
                 dataset,
                 n_sampling,
                 parameter_grid,
                 criteria='accuracy',
                 kfold=5,
                 verbose=True):

        self.verbose = verbose

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
        self.parameters = {}
        self.criterias = {}
        for kernel in self.kernels:
            temp = RandomHyperParameterTuningPerKernel(
                self.dataset,
                self.clf,
                kernel,
                self.parameter_grid,
                self.n,
                kfold=self.kfold)
            temp.fit()
            scores_for_kernel = []
            for j in range(len(temp.scores)) :
                scores_for_kernel.append(temp.scores[j][self.criteria])
            scores_for_kernel = np.array(scores_for_kernel)
            id_param_to_take = np.argmax(scores_for_kernel)
            temp_dict = {}
            for key, values in temp.clf_parameters_to_test.items() :
                temp_dict[key] = values[id_param_to_take]
            for key, values in temp.kernel_parameters_to_test.items() :
                temp_dict[key] = values[id_param_to_take]
            self.parameters[kernel.__name__] = temp_dict
            self.criterias[kernel.__name__] = scores_for_kernel[id_param_to_take]

    def best_parameters(self):
        argmax_kernel = np.argmax(np.array(list(self.criterias.values())))
        best_kernel = list(self.criterias.keys())[argmax_kernel]
        best_parameters = self.parameters[best_kernel]
        best_parameters['kernel'] = best_kernel
        return best_parameters, self.criterias[best_kernel]


if __name__ == '__main__':

    from src.data.seq import SeqData
    from scipy.stats import uniform
    from scipy.stats import randint
    from src.kernels.mismatch import MismatchKernel
    from src.kernels.spectral import SpectralKernel
    from src.kernels.wd import WDKernel
    from src.kernels.la import LAKernel
    from src.methods.ksvm import KSVM
    from src.methods.klr import KLR

    data = SeqData(k=0, dataname="train", mat=False, small=False, verbose=True)

    parameter_grid = {'kernel': [WDKernel,
                                 SpectralKernel,
                                 MismatchKernel               
                                 ],
                      'k': randint(low=3, high=9),
                      'm': randint(low=1, high=4),
                      'C': uniform(loc=0.1, scale=100),
                      'd': randint(low=3, high=12)
                      }
    rand_svm = RandomHyperParameterTuning(KSVM, data, 20, parameter_grid, kfold= 2)
    rand_svm.fit()
    print(rand_svm.best_parameters())
