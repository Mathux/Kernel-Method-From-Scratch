
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

class RandomHyperParameterTuningPerKernel(Logger):
    def __init__(self,
                 dataset,
                 clf,
                 kernel,
                 parameter_grid,
                 n_sampling,
                 kfold=5,
                 verbose=True):

        self.verbose = verbose
        self.clf = clf
        self.kernel = kernel
        self.parameter_grid = parameter_grid
        self.dataset = dataset
        self.n = n_sampling
        self.kfold = kfold
        self.kernel_parameters = list(kernel.defaultParameters.keys())
        self.clf_parameters = list(clf.defaultParameters.keys())

    def get_params_to_test(self, parameters, grid, n):
        temp_params = {}
        fix = True
        for param in parameters:
            if param in list(grid.keys()):
                if isinstance(grid[param],
                              stats._distn_infrastructure.rv_frozen):
                    temp_params[param] = grid[param].rvs(size = n)
                else:
                    if isinstance(grid[param], np.ndarray) or isinstance(grid[param], list) :
                        if len(param) == n :
                            temp_params[param] = grid[param]
                        else : 
                            pass
                    else :
                        temp_params[param] = np.array([grid[param]]*n)
                        fix = True
        return temp_params, fix

    def fit(self):
        self.scores = {}
        self.kernel_parameters_to_test, fix_kernel = self.get_params_to_test(
            self.kernel_parameters, self.parameter_grid, self.n)
        self.clf_parameters_to_test, _ = self.get_params_to_test(
            self.clf_parameters, self.parameter_grid, self.n)
        print('fix_kernel : ',fix_kernel)
        if fix_kernel :
            params = {
                key: value[0]
                for key, value in self.kernel_parameters_to_test.items()
            }
            kernel_to_try = self.kernel(
                dataset=self.dataset, parameters=params)
        for j in range(self.n):
            temp_clf_parameters = {}
            for c_param in self.clf_parameters:
                if c_param in list(self.parameter_grid.keys()):
                    temp_clf_parameters[c_param] = self.clf_parameters_to_test[
                        c_param][j]
            if not fix_kernel:
                temp_kernel_parameters = {}
                for k_param in self.kernel_parameters:
                    if k_param in list(self.parameter_grid.keys()):
                        temp_kernel_parameters[
                            k_param] = self.kernel_parameters_to_test[k_param][
                                j]
                kernel_to_try = self.kernel(
                    self.dataset, parameters=temp_kernel_parameters)

            temp_clf = self.clf(kernel_to_try, parameters=temp_clf_parameters)
            CV = CrossValidation(self.dataset, temp_clf, kfolds=self.kfold)
            temp_report = CV.stats
            self.scores[j] = temp_report


class RandomHyperParameterTuning(Logger):
    def __init__(self,
                 classifier,
                 dataset,
                 n_sampling=5,
                 parameter_grid={},
                 criteria='accuracy',
                 kfold=3,
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
        self.all_parameters = {}
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


    alldata = AllSeqData(parameters={"nsmall": 500, "small": False})
    data0 = alldata[0]["train"]

    parameter_grid = {
        'kernel': [SpectralKernel],
        'k': 6,
        'lam': uniform(loc = 1/2, scale = 20),
    }
    rand_klr = RandomHyperParameterTuning(
        KLR, data0, n_sampling=1, parameter_grid=parameter_grid, kfold=7)
    rand_klr.fit()
    print(rand_klr.best_parameters())
    send_sms("Finished random search")