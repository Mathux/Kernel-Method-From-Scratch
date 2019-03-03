#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:50:40 2019

@author: evrardgarcelon, mathispetrovich
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


class Serializable:
    def __init__(self, name, dic):
        self.__name__ = name

    def to_json(self):
        import json
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def __str__(self):
        return str(self.dic)

    def __repr__(self):
        return self.__name__ + "(" + str(self.dic) + ")"

    @property
    def dic(self):
        dic = self.__dict__
        return {i: dic[i] for i in dic if i != "__name__"}


class Parameters(Serializable):
    def __init__(self, param, defaultparam=None):
        super(Parameters, self).__init__("Parameters", self.__dict__)
        if defaultparam is not None:
            if param is not None:
                self.dic_to_param_with_default(param, defaultparam)
            else:
                self.dic_to_param(defaultparam)
        else:
            self.dic_to_param(param)

    def dic_to_param(self, param):
        for (name, value) in param.items():
            self.__dict__[name] = value

    def dic_to_param_with_default(self, param, defaultparam):
        for (name, value) in defaultparam.items():
            self.__dict__[name] = value

        for (name, value) in param.items():
            if name not in defaultparam.keys():
                print(
                    "WARNING: '" + name +
                    "' is not a valid parameter (it is discarded), here are the valid ones: "
                    + str(list(defaultparam.keys())))
            else:
                self.__dict__[name] = value


class Logger:
    @staticmethod
    def log(verbose, *args):
        if verbose:
            print(*args)
    
    def _log(self, *args):
        Logger.log(self.verbose, *args)

    def vrange(self, n, desc="", leave=False):
        if type(n) == int:
            n = (0, n)
        if self.verbose:
            return tqdm(range(*n), desc=desc, leave=leave)
        else:
            return range(*n)

    def viterator(self, it, desc="", leave=False):
        if self.verbose:
            return tqdm(it, desc=desc, leave=leave)
        else:
            return it


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid = np.vectorize(sigmoid)


def quad(K, alpha):
    return np.dot(alpha, np.dot(K, alpha))


def nb_diff(x, y):
    nb_diff = 0
    for char1, char2 in zip(x, y):
        if char1 != char2:
            nb_diff += 1
    return nb_diff


# Tools to give the csv format
def submit(prediction, test_size=1000):
    pred = pd.DataFrame(columns=['Id', 'Bound'])
    for i in range(len(prediction)):
        predictions = 2 * prediction[i] - 1
        temp = pd.DataFrame(data=predictions, columns=['Bound'])
        temp['Id'] = np.linspace(
            i * test_size, (i + 1) * test_size - 1, test_size, dtype='int')
        temp = temp.reindex(columns=['Id', 'Bound'])
        pred = pred.append(temp)
    pred.reset_index(inplace=True)
    pred = pred.drop('index', axis=1)
    pred.to_csv('predictions.csv', index=False)
    return None


if __name__ == "__main__":
    param = {'z': 4, 'r': 5}
    p = Parameters(param)
