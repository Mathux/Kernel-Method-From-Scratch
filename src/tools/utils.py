#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:50:40 2019

@author: evrardgarcelon, mathispetrovich
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time


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

    def items(self):
        return self.dic.items()


class Logger:
    _indent = 0
    _step = 4
    
    @staticmethod
    def log(verbose, *args):
        if verbose:
            print(" "*Logger._indent, end="")
            print(*args)

    @staticmethod
    def indent():
        Logger._indent += Logger._step

    @staticmethod
    def dindent():
        Logger._indent = max(0, Logger._indent - Logger._step)
            
    def _log(self, *args):
        Logger.log(self.verbose, *args)

    def vrange(self, n, desc="", leave=False):
        desc = " "*Logger._indent + desc
        if type(n) == int:
            n = (0, n)
        if self.verbose:
            return tqdm(range(*n), desc=desc, leave=leave)
        else:
            return range(*n)

    def viterator(self, it, desc="", leave=False):
        desc = " "*Logger._indent + desc
        if self.verbose:
            return tqdm(it, desc=desc, leave=leave)
        else:
            return it


class Timer:
    def __init__(self):
        self.start()
        self.stop()

    def start(self):
        self.begin = time.perf_counter()

    def stop(self):
        self.end = time.perf_counter()

    @property
    def value(self):
        return self.end - self.begin

    def __str__(self):
        return "{}s".format(int(self.value*100)/100)

    def __repr__(self):
        return self.__str__()
    

class Score:
    """Score object to evaluate a method

    """
    def __init__(self, pred, dataset):
        n = dataset.data.shape[0]
        labels = dataset.labels
        self.tp = np.sum((pred == 1.) * (labels == 1.))
        self.fn = np.sum((pred == -1.) * (labels == 1.))
        self.fp = np.sum((pred == 1.) * (labels == -1.))
        self.recall = self.tp / (self.fn + self.tp)
        self.precision = self.tp / (self.fp + self.tp)
        self.accuracy = np.sum(labels == pred) / n

    def __str__(self):
        acc = "Accuracy: " + str(self.accuracy)
        pre = "Precision: " + str(self.precision)
        rec = "Recall: " + str(self.recall)
        return ", ".join([acc, pre, rec])
    
    def __repr__(self):
        return self.__str__()
        

def sigmoid(x):
    """Return the sigmoid of x

    Parameters
    ----------
    x : float
        input value

    Returns
    -------
    float
	The sigmoid of the input

    """
    return 1 / (1 + np.exp(-x))


sigmoid = np.vectorize(sigmoid)


def quad(K, alpha):
    """Compute a quadratic form

    Parameters
    ----------
    K : array
        Matrix at the middle
    alpha : array
        Vector

    Returns
    -------
    array
	alpha^T K alpha

    """
    return np.dot(alpha, np.dot(K, alpha))


def nb_diff(x, y):
    nb_diff = 0
    for char1, char2 in zip(x, y):
        if char1 != char2:
            nb_diff += 1
    return nb_diff


def submit(predictions, ids, csvname):
    """Create the csvfile to submit a solution
    
    Parameters
    ----------
    predictions : (labels array) list
        3 lists of predictions
    ids : (int array) list
        3 lists of ids
    csvname : (string)
        Path of the csv file to write
    
    """
    Id = pd.DataFrame(pd.concat([id for id in ids], ignore_index=True))

    Pred = [pd.DataFrame(pred, columns=['Bound']) for pred in predictions]
    Pred = pd.concat(Pred, ignore_index=True)

    table = Id.join(Pred)
    table.to_csv(csvname, index=False)
        

def create_dir(directory):
    """Create a directory if it doesn't exit

    Parameters
    ----------
    directory : string
        Path of the folder

    """
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    param = {'z': 4, 'r': 5}
    p = Parameters(param)

    t = Timer()
