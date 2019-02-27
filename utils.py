#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:50:40 2019

@author: evrardgarcelon
"""

import pandas as pd
from config import *
import numpy as np
import sys
import kernel


# Load train data
def load_train(mat=True):

    print('Loading train set...')
    if mat:
        trainPaths = [
            path_to_train0_mat,
            path_to_train1_mat,
            path_to_train2_mat]
    else:
        trainPaths = [path_to_train0, path_to_train1, path_to_train2]
    trainPathReturns = [
        path_to_train_labels0,
        path_to_train_labels1,
        path_to_train_labels2]
    x_train = []
    y_train = []
    i = 0
    for path_features, path_labels in zip(trainPaths, trainPathReturns):
        y_train.append(pd.read_csv(path_labels))
        if mat:
            temp = pd.read_csv(
                path_features,
                sep=' ',
                dtype='float64',
                header=None)
            temp['Id'] = y_train[i]['Id']
        else:
            temp = pd.read_csv(path_features, sep=',')
        i += 1
        x_train.append(temp)

    print('Done')
    return x_train, y_train


# Load test data
def load_test(mat=True, test_size=1000):

    print('Loading test set...')
    if mat:
        testPaths = [path_to_test0_mat, path_to_test1_mat, path_to_test2_mat]
    else:
        testPaths = [path_to_test0, path_to_test1, path_to_test2]
    x_test = []
    for i, path_features in enumerate(testPaths):
        if mat:
            temp = pd.read_csv(
                path_features,
                sep=' ',
                dtype='float64',
                header=None)
            temp['Id'] = np.linspace(
                i * test_size,
                (i + 1) * test_size - 1,
                test_size,
                dtype='int')

        else:
            temp = pd.read_csv(path_features, sep=',')
        x_test.append(temp)
    print('Done')
    return x_test


# Split the train dataset
def split_dataset(data, labels, split_val=0.1, seed=SEED):

    np.random.seed(seed)
    train, val = [], []
    train_labels, val_labels = [], []
    print('Splitting data set...')
    for i in range(len(data)):
        data[i] = data[i].merge(labels[i], on='Id')
        n_samples = data[i].shape[0]
        train_temp = data[i].sample(frac=1).iloc[:int(
            (1 - split_val) * n_samples), :].sort_values(by=['Id'])
        val_temp = data[i].sample(frac=1).iloc[int(
            (1 - split_val) * n_samples):, :].sort_values(by=['Id'])
        train_temp.reset_index(inplace=True)
        val_temp.reset_index(inplace=True)
        train_labels_temp = train_temp[['Id', 'Bound']]
        val_labels_temp = val_temp[['Id', 'Bound']]
        data[i] = data[i].drop('Bound', axis=1)
        train_temp, val_temp = train_temp.drop(
            ['Bound', 'index'], axis=1), val_temp.drop(['Bound', 'index'], axis=1)
        train.append(train_temp)
        val.append(val_temp)
        train_labels.append(train_labels_temp)
        val_labels.append(val_labels_temp)
    print('Done')
    return train, val, train_labels, val_labels


# Tools to give the csv format
def submission(prediction, test_size=1000):

    pred = pd.DataFrame(columns=['Id', 'Bound'])
    for i in range(len(prediction)):
        predictions = 2 * prediction[i] - 1
        temp = pd.DataFrame(data=predictions, columns=['Bound'])
        temp['Id'] = np.linspace(
            i * test_size,
            (i + 1) * test_size - 1,
            test_size,
            dtype='int')
        temp = temp.reindex(columns=['Id', 'Bound'])
        pred = pred.append(temp)
    pred.reset_index(inplace=True)
    pred = pred.drop('index', axis=1)
    pred.to_csv('predictions.csv', index=False)
    return None


def get_kernel(ker, gamma=1, offset=1, dim=1, k=1, m=1, e = 1, d = 1, beta = 1):
    if ker == 'linear':
        return kernel.Kernel().linear(offset)
    elif ker == 'gaussian':
        return kernel.Kernel().gaussian(gamma)
    elif ker == 'sigmoid':
        return kernel.Kernel().sigmoid(gamma, offset)
    elif ker == 'polynomial':
        return kernel.Kernel().polynomial(dim, offset)
    elif ker == 'laplace':
        return kernel.Kernel().laplace(gamma)
    elif ker == 'spectral':
        return kernel.SpectralKernel(k)
    elif ker == 'mismatch':
        return kernel.MismatchKernel(k, m)
    elif ker == 'WD':
        return kernel.WDKernel(d)
    elif ker == 'LA' :
        return kernel.LAKernel(e,d,beta)
    else:
        raise Exception('Invalid Kernel')


def get_kernel_parameters(kernel):
    if kernel == 'linear':
        return ['offset']
    elif kernel == 'gaussian':
        return ['gamma']
    elif kernel == 'sigmoid':
        return ['gamma', 'offset']
    elif kernel == 'polynomial':
        return ['offset', 'dim']
    elif kernel == 'laplace':
        return ['gamma']
    elif kernel == 'spectral':
        return ['k']
    elif kernel == 'mismatch':
        return ['k', 'm']
    elif kernel == 'WD':
        return ['d']
    elif kernel == 'laplace' :
        return ['gamma']
    else:
        raise Exception('Invalid Kernel')


def transform_label(y):
    if -1 in y:
        return ((y + 1) / 2).astype('float64')
    elif 0 in y:
        return (2 * y - 1).astype('float64')
    else:
        raise Exception('Bad labels')


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


def normalize_kernel(kernel):

    nkernel = np.copy(kernel)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    for i in range(nkernel.shape[0]):
        for j in range(i + 1, nkernel.shape[0]):
            q = np.sqrt(nkernel[i, i] * nkernel[j, j])
            if q > 0:
                nkernel[i, j] /= q
                nkernel[j, i] = nkernel[i, j]
    np.fill_diagonal(nkernel, 1.)

    return nkernel


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
    ax.scatter(X_r[index_0][:, 0], X_r[index_0]
               [:, 1], X_r[index_0][:, 2], c='red')
    ax.scatter(X_r[index_1][:, 0], X_r[index_1]
               [:, 1], X_r[index_1][:, 2], c='blue')
    plt.show()


if __name__ == "__main__":
    x_train, y_train = load_train(mat=False)
    x_test = load_test(mat=False)
    #train,val,train_labels,val_labels = split_dataset(x_train, y_train)
    #train,train_labels = process_data(train,train_labels)
    #val,val_labels = process_data(val,val_labels)
    x, y = process_data(x=x_train, y=y_train)
    x_test = process_data(x=x_test)
    X = x[0]
    y = y[0]
    #import kernel
    #K = kernel.MismatchKernel(k = 6,m = 1).get_kernel_matrix(x[0])
    # plot_pca(K,y[0])
