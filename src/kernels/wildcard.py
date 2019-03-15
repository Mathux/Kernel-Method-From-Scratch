#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:04:38 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, KernelCreate
from src.tools.utils import Parameters


def wildcard_match(x, y):
    result = True
    for i in range(len(x)):
        if x[i] != y[i] and y[i] != '*':
            result = False
            break
    return result


class WildcardStringKernel(StringKernel, metaclass=KernelCreate):
    name = "wildcard"
    defaultParameters = {"k": 5, "m": 1, 'la': 1, "trie": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers_wildcard))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers_wildcard):
                phi[j] += self.param.la**(b.count('*')) * wildcard_match(
                    x_kmer, b)
        return phi


class WildcardTrieKernel(TrieKernel, metaclass=KernelCreate):
    name = "wildcard"
    defaultParameters = {"k": 5, 'm': 1, 'la': 1, "trie": True}

    def k_value(self, x):
        leafs = self.get_leaf_nodes(self.trie)
        self.leaf_kgrams_ = dict((leaf.full_label,
                                  dict((index, (len(kgs),
                                                leaf.full_label.count('*')))
                                       for index, kgs in leaf.kgrams.items()))
                                 for leaf in leafs)
        k_x = np.zeros(len(self.data))
        for kmer, count1 in self.unique_kmers(x):
            if kmer in list(self.leaf_kgrams_.keys()):
                for j in range(len(self.data.data)):
                    if j in list(self.leaf_kgrams_[kmer].keys()):
                        kgrams, nb_wildcard = self.leaf_kgrams_[kmer][j]
                        k_x[j] += self.param.la**nb_wildcard * (
                            count1 * kgrams)

        return k_x


class __WildcardKernel:
    def __init__(self):
        self.defaultParameters = {"k": 5, 'm': 1, 'la': 1, "trie": True}
        self.name = "wildcard"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.trie:
            return WildcardTrieKernel(dataset, parameters, verbose)
        else:
            return WildcardStringKernel(dataset, param, verbose)


WildcardKernel = __WildcardKernel()


if __name__ == "__main__":
    dparams = {"small": False, "nsmall": 100}
    kparams = {"k": 6, "m": 0, "trie": True}
    from src.tools.test import EasyTest
    EasyTest(kernel="wildcard", data="seq", dparams=dparams, kparams=kparams)
