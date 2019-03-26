#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:54:55 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.tools.cross_validation import CrossValidation
from src.tools.utils import Logger
from scipy import stats
from copy import deepcopy
from itertools import product

class RandomHyperParameterTuningPerKernel(Logger):
    def __init__(self,
                 dataset,
                 clf,
                 kernel,
                 parameter_grid,
                 kfold=5,
                 verbose=True):

        self.verbose = verbose
        self.clf = clf
        self.kernel = kernel
        self.parameter_grid = parameter_grid
        self.dataset = dataset
        self.kfold = kfold
        self.kernel_parameters = list(kernel.defaultParameters.keys())
        self.clf_parameters = list(clf.defaultParameters.keys())
    
    def get_product(self,lists) :
        return list(product(*lists))
            
    def get_params_to_test(self, kparameters, mparameters, grid):
        temp_params = {}
        parameters = kparameters + mparameters
        for param in parameters:
            if param in list(grid.keys()):
                temp_params[param] = list(grid[param])
        
        temp_params_list = self.get_product(list(temp_params.values()))
        l = []
        for j in range(len(temp_params_list)) :
            temp_dict_kparams = {}
            temp_dict_mparams = {}
            for k,param in enumerate(temp_params.keys()):
                if param in kparameters :
                    temp_dict_kparams[param] = temp_params_list[j][k]
                else :
                    temp_dict_mparams[param] = temp_params_list[j][k]
            l.append((temp_dict_kparams,temp_dict_mparams))
        return l

    def fit(self):
        self.scores = {}
        parameters_to_test = self.get_params_to_test(self.kernel_parameters, 
                                                     self.clf_parameters, 
                                                     self.parameter_grid)
        for j,l in enumerate(parameters_to_test) :
            kernel_params = l[0]
            clf_params = l[1]
            kernel_to_try = self.kernel(
                    self.dataset, parameters=kernel_params)
            temp_clf = self.clf(kernel_to_try, parameters=clf_params)
            print('Testing kernel parameters :', kernel_params)
            print('Testing clf classifiers : ', clf_params)
            CV = CrossValidation(self.dataset, temp_clf, kfolds=self.kfold)
            temp_report = CV.stats
            self.scores[j] = temp_report


class RandomHyperParameterTuning(Logger):
    def __init__(self,
                 classifier,
                 dataset,
                 parameter_grid={},
                 criteria='accuracy',
                 kfold=3,
                 verbose=True):

        self.verbose = verbose

        self.clf = classifier
        self.parameter_grid = parameter_grid
        self.dataset = dataset
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
        self.all_parameters = {}
        self.criterias = {}
        for kernel in self.kernels:
            temp = RandomHyperParameterTuningPerKernel(
                self.dataset,
                self.clf,
                kernel,
                self.parameter_grid,
                kfold=self.kfold)
            temp.fit()
            scores_for_kernel = []
            for j in range(len(temp.scores)):
                scores_for_kernel.append(temp.scores[j][self.criteria])
            scores_for_kernel = np.array(scores_for_kernel)
            id_param_to_take = np.argmax(scores_for_kernel)
            temp_dict = {}
            for key, values in temp.clf_parameters_to_test.items():
                temp_dict[key] = values[id_param_to_take]
            for key, values in temp.kernel_parameters_to_test.items():
                temp_dict[key] = values[id_param_to_take]
            self.parameters[kernel] = temp_dict
            self.all_parameters[kernel] = {**temp.clf_parameters_to_test, 
                               **temp.kernel_parameters_to_test}
            self.criterias[
                kernel] = scores_for_kernel[id_param_to_take]

    def best_parameters(self):
        argmax_kernel = np.argmax(np.array(list(self.criterias.values())))
        best_kernel = list(self.criterias.keys())[argmax_kernel]
        best_parameters = self.parameters[best_kernel]
        best_parameters['kernel'] = best_kernel
        print('Tested parameters : ', self.all_parameters)
        return best_parameters, self.criterias[best_kernel]


if __name__ == '__main__':
    from scipy.stats import uniform, randint
    from src.kernels.mismatch import MismatchKernel
    from src.kernels.spectral import SpectralKernel
    from src.kernels.wd import WDKernel
    from src.kernels.la import LAKernel
    from src.kernels.wildcard import WildcardKernel
    from src.methods.ksvm import KSVM
    from src.methods.klr import KLR
    from src.data.seq import AllSeqData
    from send_sms import send_sms


    alldata = AllSeqData(parameters={"nsmall": 200, "small": False})
    data0 = alldata[1]["train"]

    parameter_grid = {
        'kernel': [SpectralKernel],
        'k': [9],
        'C': [1/2,1, 3/2, 2],
    }
    rand_klr = RandomHyperParameterTuning(
        KSVM, data0, parameter_grid=parameter_grid, kfold=3)
    rand_klr.fit()
    print(rand_klr.best_parameters())
    send_sms("Finished random search")
