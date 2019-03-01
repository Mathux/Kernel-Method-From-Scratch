#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:00:46 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
import config


class CrossValidation(object):
    def __init__(self, X, y, estimator, k_fold=5):
        self.accuracy = np.zeros(k_fold)
        self.recall = np.zeros(k_fold)
        self.precision = np.zeros(k_fold)
        self.X, self.y = X, y
        self.splitted_data = self.split_kfold(n_fold=k_fold)
        for k in range(k_fold):
            print('Fitting Fold {}'.format(k + 1), '...')
            data_to_stack = []
            labels_to_stack = []
            for j in range(k_fold):
                if j != k:
                    data_to_stack.append(self.splitted_data[j][0])
                    labels_to_stack.append(self.splitted_data[j][1])
            data = np.vstack(data_to_stack)
            labels = np.hstack(labels_to_stack)
            estimator.fit(data.squeeze(), labels)
            self.accuracy[k], self.recall[k], self.precision[
                k] = estimator.score_recall_precision(
                    self.splitted_data[k][0].squeeze(),
                    self.splitted_data[k][1])
            print('Done')
        self.f1_score = 2 * self.precision * self.recall / (
            self.precision + self.recall)

    def split_kfold(self, n_fold=5, seed=config.SEED, shuffle=False):
        np.random.seed(seed)
        data_shape = self.X.shape
        data = self.X
        if not len(data_shape) > 1:
            data = data.reshape((data_shape[0], 1))
        temp_labels = self.y.reshape((len(self.y), 1))
        data = np.hstack([data, temp_labels])
        if shuffle == True:
            np.random.shuffle(data, axis=0)
        n_samples = data_shape[0]
        indexes = np.linspace(0, n_samples - 1, n_fold + 1, dtype='int')
        splitted = {}
        for j in range(n_fold):
            temp = data[indexes[j]:indexes[j + 1], :]
            train, labels = temp[:, :-1], temp[:, -1]
            splitted[j] = (train, labels.astype('float64').squeeze())
        return list(splitted.values())

    def mean_acc(self):
        return np.mean(self.accuracy)

    def std_acc(self):
        return np.std(self.accuracy)

    def mean_recall_score(self):
        return np.mean(self.recall)

    def std_recall_score(self):
        return np.std(self.recall)

    def mean_precision_score(self):
        return np.mean(self.precision)

    def std_precision_score(self):
        return np.std(self.precision)

    def mean_f1_score(self):
        return np.mean(self.f1_score)

    def std_f1_score(self):
        return np.std(self.f1_score)
