#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:56:05 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate
from src.methods.ksvm import KSVM
from src.methods.utils import quad


class MultiKernel:
    def __init__(self, kernels, dataset):
        self.kernels = kernels
        self.dataset = dataset
        self.labels = dataset.labels
        self.data = dataset.data
        self.n = self.data.shape[0]
        self.Ks = np.array([k.K for k in kernels])
        self.size = len(self.Ks)

    # Combinaison of kernels
    def Kw(self, weights):
        self.K = np.einsum("ijk,i->jk", self.Ks, weights)
        return self.K

    def __getitem__(self, key):
        assert (key in range(self.size))
        return self.Ks[key]


class SimpleMKL(KMethod, metaclass=KMethodCreate):
    defaultParameters = {
        "C": 10**10,
        "a": 0.2,
        "b": 0.5,
        "weights": None,
        "n_iter_fit": 50,
        "n_iter_linesearch": 100,
        "tol": 10**-10,
        "tol_sv": 10**-4
    }

    def fit(self, dataset, verbose=False):
        self.multiK = MultiKernel(self.kernel, dataset)
        size = self.multiK.size

        _cond = self.param.weights is None
        weights = np.ones(size) / size if _cond else self.param.weights

        labels = dataset.labels

        gap = 1
        m = 0

        K = self.multiK.Kw(weights)

        # for n in self.vrange(self.param.n_iter_fit, desc="Outer loop"):
        for n in range(self.param.n_iter_fit):
            mu = np.argmax(weights)
            J, grad_J, D, _ = self.compute_J_grad_J_D(K, labels, mu, weights)
            J_cross = 0
            D_cross = 1 * D
            d_cross = 1 * weights

            for m in self.vrange(self.param.n_iter_fit, desc="Inner loop"):
                weights = d_cross
                D = 1 * D_cross
                if m > 0:
                    J = J_cross
                print(D)

                # a eclaircir
                temp = 1 * weights / D
                mask = (D >= 0)
                temp[mask] = -1 * np.inf
                nu = np.argmin(-temp)
                gamma_max = -temp[nu]

                d_cross = 1 * weights + gamma_max * D
                D_cross[nu] = 0
                D_cross = self.normalize_D(D_cross, mu)
                K_cross = self.multiK.Kw(d_cross)

                J_cross, _, _, _ = self.compute_J_grad_J_D(
                    K_cross, labels, mu, d_cross)

                if J_cross >= J:
                    break

            gamma, alpha = self.line_search(weights, D, gamma_max, labels)
            self.alpha = alpha
            weights = weights + gamma * D

            K = self.multiK.Kw(weights)
            temp_max = -np.inf
            for i in range(len(self.kernels)):
                temp = np.dot(alpha, np.dot(self.kernels_matrix[i], alpha))
                if temp > temp_max:
                    temp_max = temp
            gap = temp_max - np.dot(alpha, np.dot(K, alpha))
            
            if gap < self.param.tol:
                break
            
        self.sv = np.linspace(
            0, len(self.alpha) - 1, len(self.alpha),
            dtype='int')[np.abs(self.alpha) > self.param.tol_sv]

        self.param.weights = weights
        return weights
    
    def normalize_D(self, D, mu):
        temp_D = 1 * D
        temp_D[mu] = -np.sum(temp_D)
        return temp_D

    def compute_J_grad_J_D(self, K, labels, mu, d, eps=10**-20):
        ksvm = KSVM(self.multiK, verbose=False)
        alpha = ksvm.fit()
        J = -0.5 * quad(K, alpha) + labels.dot(alpha)
        grad_J = -0.5 * np.array([quad(K, alpha) for K in self.multiK.Ks])

        reduced_grad = grad_J - grad_J[mu]

        mask = (d <= eps) * (reduced_grad > 0)
        reduced_grad[mask] = 0
        reduced_grad = -reduced_grad
        reduced_grad[mu] = -np.sum(reduced_grad)
        return J, grad_J, reduced_grad, alpha

    def line_search(self, weights, D, gamma, labels):
        a = self.param.a
        b = self.param.b

        K = self.multiK.Kw(weights)
        f0, grad_J, _, _ = self.compute_J_grad_J_D(K, labels, 0, weights)

        K = self.multiK.Kw(weights + gamma * D)
        f1, _, _, alpha = self.compute_J_grad_J_D(K, labels, 0,
                                                  weights + gamma * D)

        n = 0
        n_iter = self.param.n_iter_linesearch
        while f1 > f0 + a * gamma * np.dot(grad_J, D) and n < n_iter:
            gamma = gamma * b
            n += 1
            K = self.multiK.Kw(weights + gamma * D)
            f1, _, _, alpha = self.compute_J_grad_J_D(K, labels, 0,
                                                      weights + gamma * D)
        return gamma, alpha

    def predict(self, X):
        f = np.zeros(len(X))
        for j, x in enumerate(X):
            proj = 0
            for i in self.sv:
                for m, w in enumerate(self.param.weights):
                    proj += self.alpha[i] * w * utils.get_kernel(
                        self.kernels[m], **self.kernel_parameters)(self.X[i],
                                                                   x)
            f[j] = proj
        return np.sign(f)


if __name__ == '__main__':
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.linear import LinearKernel

    from src.data.synthetic import GenClassData
    data = GenClassData(500, 2, mode="circle")
    kernels = [GaussianKernel(data), LinearKernel(data)]

    smkl = SimpleMKL(kernels)

    weights = smkl.fit(data)


"""
    preds = clf.predict(X)
    import pylab as plt
    plt.figure(1)
    plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='o')
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', marker='^')
    for j in range(len(preds)):
        if preds[j] == 1:
            plt.scatter(X[j][0], X[j][1], color='magenta', marker='+')
        elif preds[j] == -1:
            plt.scatter(X[j][0], X[j][1], color='cyan', marker='x')
        else:
            print('pb')
"""
