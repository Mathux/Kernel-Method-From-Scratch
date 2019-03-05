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
    def __init__(self, dataset, estimator, kfolds=5, verbose=True):
        self.verbose = verbose

        self.accuracy = np.zeros(kfolds)
        self.recall = np.zeros(kfolds)
        self.precision = np.zeros(kfolds)

        self.dataset = dataset
        self.folds = KFold(dataset, kfolds, verbose)

        for k in self.vrange(kfolds, desc="Fitting folds"):
            train_dataset, val_dataset = self.folds[k]
            estimator.fit(train_dataset)

            acc, recall, pres = estimator.score_recall_precision(val_dataset)
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


if __name__ == '__main__':
    # from src.data.seq import SeqData
    # data = SeqData(k=0, dataname="train", small=False, verbose=True)

    # from src.kernels.mismatch import MismatchKernel
    # kernel = MismatchKernel(data, verbose=True)
    
    from src.data.synthetic import GenClassData
    data = GenClassData(500, 2, mode="circle")

    from src.kernels.gaussian import GaussianKernel
    kernel = GaussianKernel(data, verbose=False)

    from src.methods.klr import KLR
    klr = KLR(kernel, verbose=False)

    from src.methods.ksvm import KSVM
    ksvm = KSVM(kernel, verbose=False)

    crossval = CrossValidation(data, estimator=ksvm, kfolds=2, verbose=True)
    print(crossval)
