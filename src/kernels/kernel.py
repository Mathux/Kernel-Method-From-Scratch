#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:30:29 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.tools.utils import Parameters, Logger
from src.data.trie_dna import Trie


class KernelCreate(type):
    def __init__(cls, clsname, superclasses, attributedict):
        if "toreset" not in cls.__dict__:
            cls.toreset = superclasses[0].toreset

        def init(self, dataset=None, parameters=None, verbose=True):
            super(cls, self).__init__(
                dataset=dataset,
                name=clsname,
                parameters=parameters,
                verbose=verbose,
                cls=cls)

        cls.__init__ = init


class Kernel(Logger):
    def __init__(self,
                 dataset=None,
                 name="Kernel",
                 parameters=None,
                 verbose=True,
                 toreset=["_K", "_KC", "_n", "_m"],
                 cls=None):
        self.verbose = verbose

        self._toreset = toreset

        self.param = Parameters(parameters, cls.defaultParameters)

        if dataset is not None:
            self.dataset = dataset
        else:
            self.reset()

        self.__name__ = name

    def __call__(self, x, y):
        return self.kernel(x, y)

    def reset(self):
        dic = self.__dict__
        for el in self._toreset:
            dic[el] = None

    @property
    def data(self):
        if self._data is None:
            raise ValueError("This kernel didn't have any data")
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            raise ValueError("This kernel didn't have any labels")
        return self._labels

    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError("This kernel didn't have any dataset")
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self._data = value.data
        self._labels = value.labels
        self.reset()
        # Init some values
        self._n = value.data.shape[0]
        self._m = value.data.shape[1] if len(value.data.shape) > 1 else None

    @property
    def n(self):
        if self._n is None:
            raise ValueError("This kernel didn't have any data, so no shape")
        return self._n

    @property
    def m(self):
        if self._m is None:
            raise ValueError("This kernel didn't have any data, so no shape")
        return self._m

    @property
    def K(self):
        # Check if K is computed before
        if self._K is None:
            # Compute the gram matrix
            self._compute_gram()
        return self._K

    def _normalized_kernel(self, K):
        for i in self.vrange(K.shape[0], "Normalise kernel"):
            for j in range(i + 1, K.shape[0]):
                q = np.sqrt(K[i, i] * K[j, j])
                if q > 0:
                    K[i, j] /= q
                    K[j, i] = K[i, j]
        np.fill_diagonal(K, 1.)
        self._K = K + 1

    @property
    def KC(self):
        # Check if KC is computed before
        if self._KC is None:
            # Compute the centered gram matrix
            self._compute_centered_gram()
        return self._KC

    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        for i in self.vrange(self.n, "Gram matrix of " + self.__name__):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = self.kernel(self.data[i], self.data[j])
        self._normalized_kernel(K)

    def _compute_centered_gram(self):
        K = self.K
        self._log("Center the gram matrix..")
        n = self.n
        oneN = (1 / n) * np.ones((n, n))
        self._KC = K - oneN.dot(K) - K.dot(oneN) + oneN.dot(K.dot(oneN))
        self._log("Gram matrix centered!")

    def predict(self, x):
        return np.array([
            self.kernel(xi, x) / np.sqrt(
                self.kernel(xi, xi) * self.kernel(x, x)) + 1
            for xi in self.data
        ])

    def addconfig(self, name, value):
        self.config[name] = value

    def __str__(self):
        name = "Kernel: " + self.__name__
        param = "Parameters: " + str(self.param)
        return name + ", " + param

    def __repr__(self):
        return self.__str__()

    def show_pca(self, pred=None, dim=3):
        from src.methods.kpca import KPCA
        kpca = KPCA(self, parameters={"dim": dim})
        proj = kpca.project()
        self.data.show_pca(proj, pred, dim=dim)


class GenKernel(Kernel):
    def __init__(self,
                 dataset=None,
                 name="GenKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        assert ("toreset" in cls.__dict__)
        toreset = cls.toreset
        super(GenKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            toreset=toreset,
            cls=cls)


class DataKernel(GenKernel):
    toreset = ["_K", "_KC", "_n", "_m"]

    def __init__(self,
                 dataset=None,
                 name="DataKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        super(DataKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            cls=cls)


class StringKernel(GenKernel):
    toreset = ["_K", "_KC", "_n", "_m", "_phis", "_mers", '_mers_wildcard']

    def __init__(self,
                 dataset=None,
                 name="StringKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        super(StringKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            cls=cls)

    def _compute_phis(self):
        phis = []
        for x in self.viterator(self.data, "Phis of " + self.__name__):
            phi = self._compute_phi(x)
            phis.append(phi)
        self._phis = np.array(phis)

    def predict(self, x):
        phix = self._compute_phi(x)
        k_xx = np.dot(phix, phix)
        return np.array([
            np.dot(phix, phi) / np.sqrt(np.dot(phi, phi) * k_xx) + 1
            for phi in self.phis
        ])

    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        phis = self.phis

        for i in self.vrange(self.n, desc="Gram matrix of " + self.__name__):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = np.dot(phis[i], phis[j])

        self._normalized_kernel(K)

    def kernel(self, x, y):
        return np.dot(self._compute_phi(x), self._compute_phi(y))

    @property
    def phis(self):
        # Check if phis are computed before
        if self._phis is None:
            # Compute phis
            self._compute_phis()
        return self._phis

    @property
    def mers(self):
        from itertools import product
        if self._mers is None:
            self._mers = [(''.join(c))
                          for c in product('ACGT', repeat=self.param.k)]
        return self._mers

    @property
    def mers_wildcard(self):
        from itertools import product
        if self._mers_wildcard is None:
            self._mers_wildcard = [
                (''.join(c)) for c in product('ACGT*', repeat=self.param.k)
            ]

        def func(x):
            return np.sum(np.array(list(x)) == '*') <= self.param.m

        vfunc = np.vectorize(func)
        self._mers_wildcard = np.array(self._mers_wildcard)[vfunc(
            self._mers_wildcard)]
        return self._mers_wildcard


class TrieKernel(GenKernel):
    toreset = ["_K", "_KC", "_n"]

    def __init__(self,
                 dataset=None,
                 name="TrieKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        super(TrieKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            cls=cls)

    def unique_kmers(self, x):
        x = list(x)
        ukmers = []
        offset = 0
        seen_kmers = []
        for offset in range(len(x) - self.param.k + 1):
            kmer = x[offset:offset + self.param.k]
            if kmer not in seen_kmers:
                seen_kmers.append(kmer)
            count = 1
            for _offset in range(offset + 1, len(x) - self.param.k + 1):
                if np.all(x[_offset:_offset + self.param.k] == kmer):
                    count += 1
            ukmers.append((''.join(kmer), count))
        return ukmers

    def get_leaf_nodes(self, node):
        leafs = []
        self._collect_leaf_nodes(node, leafs)
        return leafs

    def _collect_leaf_nodes(self, node, leafs):
        if node is not None:
            if len(node.children) == 0:
                leafs.append(node)
            for k, v in node.children.items():
                self._collect_leaf_nodes(v, leafs)

    def k_value(self, x):
        leafs = self.get_leaf_nodes(self.trie)
        self.leaf_kgrams_ = dict((leaf.full_label,
                                  dict((index, (len(kgs),
                                                leaf.full_label.count('*')))
                                       for index, kgs in leaf.kgrams.items()))
                                 for leaf in leafs)
        k_x = np.zeros(len(self.data))
        for kmer, count1 in self.unique_kmers(x, self.param.k):
            if kmer in self.leaf_kgrams_.keys():
                for j in range(len(self.data.data)):
                    if j in self.leaf_kgrams_[kmer].keys():

                        kgrams, nb_wildcard = self.leaf_kgrams_[kmer][j]
                        k_x[j] += self.param.la**nb_wildcard * (
                            count1 * kgrams)

        return k_x

    def predict(self, x):
        t = Trie(la=self.param.la)
        k_xx, _, _ = t.dfs(np.array([x]), self.param.k, self.param.m)
        k_xx = k_xx.squeeze()
        k_v = self.k_value(x, changev=True)
        return np.array([
            k_v[i] / np.sqrt(self.K[i, i] * k_xx) + 1
            for i in range(len(self.K))
        ])

    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        self.trie = Trie(la=self.param.la)
        K, _, _ = (self.trie).dfs(
            self.dataset.data, k=self.param.k, m=self.param.m)
        self._normalized_kernel(K)


def AllStringKernels():
    from src.kernels.spectral import SpectralKernel
    from src.kernels.mismatch import MismatchKernel
    from src.kernels.wd import WDKernel
    from src.kernels.la import LAKernel
    from src.kernels.wildcard import WildcardKernel

    kernels = [
        MismatchKernel, SpectralKernel, WDKernel, LAKernel, WildcardKernel
    ]
    names = [kernel.name for kernel in kernels]
    return kernels, names


def AllDataKernels():
    from src.kernels.exponential import ExponentialKernel
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.laplacian import LaplacianKernel
    from src.kernels.linear import LinearKernel
    from src.kernels.polynomial import PolynomialKernel
    from src.kernels.quad import QuadKernel
    from src.kernels.sigmoid import SigmoidKernel

    kernels = [
        ExponentialKernel, GaussianKernel, LaplacianKernel, LinearKernel,
        PolynomialKernel, QuadKernel, SigmoidKernel
    ]
    names = [kernel.name for kernel in kernels]
    return kernels, names


def AllKernels():
    k1, n1 = AllStringKernels()
    k2, n2 = AllDataKernels()
    return k1 + k2, n1 + n2
