#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:56:05 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate
from src.methods.ksvm import KSVM
from src.tools.utils import quad


class SimpleMKL(KMethod, metaclass=KMethodCreate):
    defaultParameters = {
        "C": 10**10,
        "a": 0.2,
        "b": 0.5,
        "weights": None,
        "n_iter_fit": 30,
        "n_iter_linesearch": 100,
        "tol": 10**-6,
        "tol_sv": 10**-6
    }

    def fit(self, verbose=False):
        self.size = size = self.kernel.size

        _cond = self.param.weights is None
        weights = np.ones(size) / size if _cond else self.param.weights

        gap = np.inf
        m = 0

        K = self.kernel.Kw(weights)

        desc = "Fitting the simple MKL"
        for n in self.vrange(self.param.n_iter_fit, desc=desc, leave=True):
            mu = np.argmax(weights)
            J, grad_J, D, _ = self.compute_J_grad_J_D(K, mu, weights)
            J_os = 0
            D_os = 1 * D
            d_os = 1 * weights

            desc = "Inner loop, previous gap (" + str(gap) + ")"
            for m in self.vrange(self.param.n_iter_fit, desc=desc):
                weights = d_os
                D = 1 * D_os

                temp = np.inf * np.ones_like(weights)
                mask = D < 0
                temp[mask] = -weights[mask] / D[mask]

                nu = np.argmin(temp)

                gamma_max = temp[nu]

                d_os = weights + gamma_max * D

                D_os[mu] = D[mu] - D[nu]
                D_os[nu] = 0

                K_os = self.kernel.Kw(d_os)

                J_os, _, _, _ = self.compute_J_grad_J_D(K_os, mu, d_os)

                if J_os >= J:
                    break

            desc = "Line search, previous gap (" + str(gap) + ")"
            gamma, alpha = self.line_search(weights, D, gamma_max, desc=desc)
            self._alpha = alpha
            weights = weights + gamma * D

            K = self.kernel.Kw(weights)
            quad_max = np.max(
                [quad(self.kernel[i], alpha) for i in range(size)])
            gap = quad_max - quad(K, alpha)

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

    def compute_J_grad_J_D(self, K, mu, d, eps=10**-20):
        ksvm = KSVM(self.kernel, verbose=False)
        alpha = ksvm.fit(K=K)
        labels = self.dataset.labels
        J = -0.5 * quad(K, alpha) + labels.dot(alpha)
        grad_J = -0.5 * np.array([quad(K, alpha) for K in self.kernel.Ks])

        reduced_grad = grad_J - grad_J[mu]

        mask = (d <= eps) * (reduced_grad > 0)
        reduced_grad[mask] = 0
        reduced_grad = -reduced_grad
        reduced_grad[mu] = -np.sum(reduced_grad)
        return J, grad_J, reduced_grad, alpha

    def line_search(self, weights, D, gamma, desc="Line search"):
        a = self.param.a
        b = self.param.b

        K = self.kernel.Kw(weights)
        f0, grad_J, _, _ = self.compute_J_grad_J_D(K, 0, weights)

        K = self.kernel.Kw(weights + gamma * D)
        f1, _, _, alpha = self.compute_J_grad_J_D(K, 0, weights + gamma * D)

        n_iter = self.param.n_iter_linesearch
        for n in self.vrange(n_iter, desc):
            gamma = gamma * b
            n += 1
            K = self.kernel.Kw(weights + gamma * D)
            f1, _, _, alpha = self.compute_J_grad_J_D(K, 0,
                                                      weights + gamma * D)
            if f1 <= f0 + a * gamma * np.dot(grad_J, D):
                break
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


data = "seq" if False else "synth"

if __name__ == '__main__':
    if data == "seq":
        from src.kernels.spectral import SpectralKernel
        from src.kernels.wd import WDKernel
        from src.kernels.mismatch import MismatchKernel

        kernels = SpectralKernel, WDKernel, MismatchKernel
            
        from src.data.seq import SeqData
        data = SeqData(small=True, nsmall=200)
    
    elif data == "synth":
        from src.kernels.gaussian import GaussianKernel
        from src.kernels.quad import QuadKernel
        from src.kernels.exponential import ExponentialKernel

        kernels = GaussianKernel, ExponentialKernel, QuadKernel
        
        from src.data.synthetic import GenClassData
        data = GenClassData(500, 2, mode="circle")

    from src.kernels.multikernel import MultiKernel
    multikernel = MultiKernel(data, kernels)
    
    smkl = SimpleMKL(multikernel)
    weights = smkl.fit()

    multikernel.weights = weights

    show = False
    if show:
        from src.methods.kpca import KPCA
        kpca = KPCA(multikernel, verbose=False)
        proj = kpca.project()
        data.show_pca(proj)

    print()
    print("weights:", weights)


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
