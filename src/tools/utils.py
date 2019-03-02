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
    def __init__(self, param):
        super(Parameters, self).__init__("Parameters", self.__dict__)
        for (name, value) in param.items():
            self.__dict__[name] = value


class Logger:
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def vrange(self, n, desc=""):
        if type(n) == int:
            n = (0, n)
        if self.verbose:
            return tqdm(range(*n), desc=desc)
        else:
            return range(*n)

    def viterator(self, it, desc=""):
        if self.verbose:
            return tqdm(it, desc=desc)
        else:
            return it


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid = np.vectorize(sigmoid)


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
