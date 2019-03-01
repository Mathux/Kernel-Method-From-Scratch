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
    def __init__(self, name):
        self.__name__ = name
        
    def to_json(self):
        import json
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def test(self):
        print(self)

    # @classmethod
    #def __str__(cls):
    #    return str(cls.__dict__)

    #@classmethod
    #def __repr__(cls):
    #    return "(" + str(self.__dict__) + ")"
             
        
class Parameters(Serializable):
    def __init__(self, param):
        # super(Parameters, self).__init__()
        for (name, value) in param.items():
            self.__dict__[name] = value


class Logger:
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def vrange(self, n):
        if self.verbose:
            return tqdm(range(n))
        else:
            return range(n)

    def viterator(self, it):
        if self.verbose:
            return tqdm(it)
        else:
            return it


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid = np.vectorize(sigmoid)


# Tools to give the csv format
def submission(prediction, test_size=1000):
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


def plot_pca(data, labels):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, whiten=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_r = pca.fit(data).transform(data)
    index_0 = (labels == 0)
    index_1 = (labels == 1)
    ax.scatter(
        X_r[index_0][:, 0], X_r[index_0][:, 1], X_r[index_0][:, 2], c='red')
    ax.scatter(
        X_r[index_1][:, 0], X_r[index_1][:, 1], X_r[index_1][:, 2], c='blue')
    plt.show()

    
if __name__ == "__main__":
    param = {'z': 4, 'r': 5}
    p = Parameters(param)
