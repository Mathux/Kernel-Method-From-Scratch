#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:04:38 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, KernelCreate
from src.tools.utils import Parameters
from src.data.trie_dna import WildcardTrie


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
    Trie = WildcardTrie


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
