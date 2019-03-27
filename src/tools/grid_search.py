#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:54:55 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.tools.cross_validation import CrossValidation
from src.tools.utils import Logger
from itertools import product


class GridHyperParameterTuningPerKernel(Logger):
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

    def get_product(self, lists):
        return list(product(*lists))

    def get_params_to_test(self, kparameters, mparameters, grid):
        temp_params = {}
        parameters = kparameters + mparameters
        for param in parameters:
            if param in list(grid.keys()):
                temp_params[param] = list(grid[param])

        temp_params_list = self.get_product(list(temp_params.values()))
        l = []
        for j in range(len(temp_params_list)):
            temp_dict_kparams = {}
            temp_dict_mparams = {}
            for k, param in enumerate(temp_params.keys()):
                if param in kparameters:
                    temp_dict_kparams[param] = temp_params_list[j][k]
                else:
                    temp_dict_mparams[param] = temp_params_list[j][k]
            l.append((temp_dict_kparams, temp_dict_mparams))
        return l

    def fit(self):
        self.scores = {}
        self.parameters_to_test = self.get_params_to_test(
            self.kernel_parameters, self.clf_parameters, self.parameter_grid)
        for j, l in enumerate(self.parameters_to_test):
            kernel_params = l[0]
            clf_params = l[1]
            kernel_to_try = self.kernel(self.dataset, parameters=kernel_params)
            temp_clf = self.clf(kernel_to_try, parameters=clf_params, verbose=False)
            print('Testing kernel parameters :', kernel_params)
            print('Testing clf classifiers : ', clf_params)
            Logger.indent()
            CV = CrossValidation(self.dataset, temp_clf, kfolds=self.kfold, verbose=True)
            Logger.log(True, CV)
            Logger.log(True, "")
            Logger.dindent()
            temp_report = CV.stats
            self.scores[j] = temp_report


class GridHyperParameterTuning(Logger):
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
            temp = GridHyperParameterTuningPerKernel(
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
            self.all_parameters[kernel] = temp.parameters_to_test
            self.criterias[kernel] = scores_for_kernel[id_param_to_take]
            self.parameters

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
    from src.kernels.spectral import SpectralKernel, SpectralConcatKernel
    from src.kernels.wd import WDKernel
    from src.kernels.la import LAKernel
    from src.kernels.wildcard import WildcardKernel
    from src.methods.ksvm import KSVM
    from src.methods.klr import KLR
    from src.data.seq import AllSeqData
    from send_sms import send_sms

    alldata = AllSeqData(parameters={"small": False, "shuffle": True})
    data0 = alldata[1]["train"]

    parameter_grid = {
        'kernel': [SpectralConcatKernel],
        'kmin': [5 , 6, 7],
        'kmax': [20, 21, 22, 23, 24, 25],
        'C': [1 / 2, 1, 3 / 2, 2, 5, 0.1, 10],
    }

    rand_klr = GridHyperParameterTuningPerKernel(
        data0, KSVM, SpectralConcatKernel, parameter_grid=parameter_grid, kfold=8)
    rand_klr.fit()
    argmax = np.argmax([rand_klr.scores[i]["mean_acc"] for i in range(len(rand_klr.scores))])
    print("Best parameters:")
    print(rand_klr.parameters_to_test[argmax])
    print("Scores:")
    print(rand_klr.scores[argmax])
    
    # print(rand_klr.best_parameters())
    # send_sms("Finished grid search")
