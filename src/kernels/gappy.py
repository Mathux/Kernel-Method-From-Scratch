#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:56:15 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.kernels.kernel import GappyTrieKernel, KernelCreate
from src.tools.utils import Parameters
from src.data.trie_dna import GappyTrie




class GappyTrieKernel(GappyTrieKernel, metaclass=KernelCreate):
    name = "gappy"
    defaultParameters = {"g": 5, 'l': 4}
    Trie = GappyTrie


class __GappyKernel:
    def __init__(self):
        self.defaultParameters = {"g": 5, 'l': 4}
        self.name = "gappy"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        return GappyTrieKernel(dataset, parameters, verbose)


GappyKernel = __GappyKernel()


if __name__ == "__main__":
    dparams = {"small": False, "nsmall": 100}
    kparams = {"g": 6, "l": 5}   
    from src.tools.test import EasyTest
    EasyTest(kernels="gappy", data="allseq",methods = 'ksvm', dparams=dparams, kparams=kparams)
