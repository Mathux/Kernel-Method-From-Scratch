#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:00:46 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.tools.utils import Logger
from src.data.dataset import KFold


class CrossValidation(Logger):
    def __init__(self, dataset, estimator=None, kfolds=5, verbose=True):
        self.verbose = verbose

        self.accuracy = np.zeros(kfolds)
        self.recall = np.zeros(kfolds)
        self.precision = np.zeros(kfolds)

        self.dataset = dataset
        self.kfolds = kfolds
        self.folds = KFold(dataset, kfolds, verbose)

        if estimator is not None:
            self.fit(estimator)

    def fit(self, estimator):
        for k in self.vrange(self.kfolds, desc="Fitting folds"):
            train_dataset, val_dataset = self.folds[k]
#            print(val_dataset.labels)
            estimator.fit(train_dataset)

            Score = estimator.score_recall_precision(val_dataset)
            acc, recall, pres = Score.accuracy, Score.recall, Score.precision
            self.accuracy[k] = acc
            self.recall[k] = recall
            self.precision[k] = pres

        self.f1 = 2 * self.precision * self.recall / (
            self.precision + self.recall)

    def constant_kernel(self, kernelClass, params):
        self.kernels = []
        for k in self.vrange(self.kfolds, desc="Prepare kernels"):
            train_dataset, val_dataset = self.folds[k]
            kernel = kernelClass(train_dataset, params)
            _ = kernel.K
            self.kernels.append(kernel)
        return self.kernels
    
    def fit_K(self, estimator, kernels):
        for k in self.vrange(self.kfolds, desc="Fitting folds"):
            _, val_dataset = self.folds[k]
            estimator.fit(K=kernels[k].K)

            Score = estimator.score_recall_precision(val_dataset)
            acc, recall, pres = Score.accuracy, Score.recall, Score.precision
            self.accuracy[k] = acc
            self.recall[k] = recall
            self.precision[k] = pres

        self.f1 = 2 * self.precision * self.recall / (
            self.precision + self.recall)
        
    @property
    def mean_acc(self):
        return np.mean(self.accuracy)

    @property
    def std_acc(self):
        return np.std(self.accuracy)

    @property
    def mean_recall(self):
        return np.mean(self.recall)

    @property
    def std_recall(self):
        return np.std(self.recall)

    @property
    def mean_precision(self):
        return np.mean(self.precision)

    @property
    def std_precision(self):
        return np.std(self.precision)

    @property
    def mean_f1(self):
        return np.mean(self.f1)

    @property
    def std_f1(self):
        return np.std(self.f1)

    SHOW = [
        "mean_acc", "std_acc", "mean_recall", "std_recall", "mean_precision",
        "std_precision", "mean_f1", "std_f1"
    ]

    @property
    def stats(self):
        res = {
            stat: eval("self." + stat, {"self": self})
            for stat in CrossValidation.SHOW
        }
        return res
    
    def __str__(self):
        res = self.stats
        return 'CrossValidation: ' + str(res)

    def __repr__(self):
        return self.__str__()
