#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:50:40 2019

@author: evrardgarcelon, mathispetrovich
"""

import pandas as pd
import src.config as conf
import numpy as np
from tqdm import tqdm


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


# Split the train dataset
def split_dataset(data, labels, split_val=0.1, seed=conf.SEED):
    np.random.seed(seed)
    train, val = [], []
    train_labels, val_labels = [], []
    print('Splitting data set...')
    for i in range(len(data)):
        data[i] = data[i].merge(labels[i], on='Id')
        n_samples = data[i].shape[0]
        train_temp = data[i].sample(
            frac=1).iloc[:int((1 - split_val) * n_samples), :].sort_values(
                by=['Id'])
        val_temp = data[i].sample(
            frac=1).iloc[int((1 - split_val) * n_samples):, :].sort_values(
                by=['Id'])
        train_temp.reset_index(inplace=True)
        val_temp.reset_index(inplace=True)
        train_labels_temp = train_temp[['Id', 'Bound']]
        val_labels_temp = val_temp[['Id', 'Bound']]
        data[i] = data[i].drop('Bound', axis=1)
        train_temp, val_temp = train_temp.drop(
            ['Bound', 'index'], axis=1), val_temp.drop(['Bound', 'index'],
                                                       axis=1)
        train.append(train_temp)
        val.append(val_temp)
        train_labels.append(train_labels_temp)
        val_labels.append(val_labels_temp)
    print('Done')
    return train, val, train_labels, val_labels


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


def process_data(x, y=None):
    assert len(x) == 3
    has_labels = y is not None
    if has_labels:
        assert len(y) == 3
        new_y = []

    if isinstance(x[0].loc[0][1], str):
        mat = False
    else:
        mat = True
    new_x = []
    for i in range(len(x)):
        if mat:
            new_x.append(x[i].drop('Id', axis=1).values)
        else:
            new_x.append(x[i]['seq'].values)
        if has_labels:
            new_y.append(transform_label(y[i]['Bound'].values))
    if has_labels:
        return new_x, new_y
    else:
        return new_x


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
