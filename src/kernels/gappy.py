#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:56:15 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.kernels.kernel import GappyTrieKernel, KernelCreate, SparseKernel
from src.tools.utils import Parameters
from src.data.trie_dna import GappyTrie
from itertools import product




class GappyTrieKernel(GappyTrieKernel, metaclass=KernelCreate):
    name = "gappy"
    defaultParameters = {"g": 5, 'l': 4, "sparse" : False}
    Trie = GappyTrie


class __GappyKernel:
    def __init__(self):
        self.defaultParameters = {"g": 5, 'l': 4, "sparse" : False}
        self.name = "gappy"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.sparse :
            return GappySparseKernel(dataset, parameters, verbose)
        else : 
            return GappyTrieKernel(dataset, parameters, verbose)


#class GappySparseKernel(SparseKernel, metaclass=KernelCreate):
#    name = "gappy"
#    defaultParameters = {"g": 6, "l" : 5, "trie": False, "sparse": True}
#    
#    def phi_alpha(self, alpha, l = None) :
#        phi_alpha = {}
#        for i in range(len(alpha) - l  1) :
#            temp_phi_alpha = self.phi_alpha
#            
#        return phi_alpha
#            
#    def _compute_phi(self, x):
#        phi = {}
#        for offset in range(len(x) - self.param.g + 1):
#            xkmer = x[offset:offset + self.param.g]
#            phi[xkmer] = phi.get(xkmer, {}) + self.phi_alpha(xkmer, l = self.param.l)
#        return phi_alpha

GappyKernel = __GappyKernel()


if __name__ == "__main__":
    dparams = {"small": False, "nsmall": 100}
    kparams = {"g": 6, "l": 5, "sparse" : True}   
#    from src.tools.test import EasyTest
#    EasyTest(kernels="gappy", data="allseq",methods = 'ksvm', dparams=dparams, kparams=kparams)
#    dparams = {"small": False, "nsmall": 100}
#    kparams = {"k": 12, "sparse": True}
    # from src.tools.test import EasyTest
    # EasyTest(kernels="spectral", data="seq", dparams=dparams, kparams=kparams)
    from src.tools.test import KernelTest
    parameters = []
    
    g, l  = 6,5 
    #parameters.append({"g": g, "l" : l, "sparse": False})
    parameters.append({"g": g, "l" : l, "sparse": True})
    KernelTest("gappy", parameters)