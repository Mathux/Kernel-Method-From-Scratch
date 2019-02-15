#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:57:37 2019

@author: evrardgarcelon
"""

import numpy as np
from trie_dna import Trie

class Kernel(object) :

    def __init__(self) :
        pass
        
    def linear(self,offset = 1) :
        return lambda x,y : np.dot(x,y) + offset
    
    def gaussian(self,gamma) :
        return lambda x,y : np.exp(-gamma*np.dot(x-y,x-y))
    
    def sigmoid(self,gamma, offset) :
        return lambda x,y : np.tanh(gamma*np.dot(x,y) + offset)
    
    def polynomial(self,dim,offset) :
        return lambda x,y : (offset + np.dot(x,y))**dim
    
    def spectral_kernel(self, k) :
        
        vocab = { 0 : 'A', 1 : 'T', 2 : 'G', 3 : 'C'}
        def mismatch(x,y) :
            trie_x = Trie(x,vocab,k)
            trie_y = Trie(y,vocab,k)
            kernel = 0
            for offset in range(len(y)-k+1) :
                string = y[offset : offset+k]
                is_trie_x, count_x = trie_x.is_leaf(string)
                if is_trie_x :
                    is_trie_y,count_y  = trie_y.is_leaf(string)
                    kernel += count_x*count_y
            return kernel
        return mismatch
    
    def mismatch_kernel(self,k,m) :
        
        vocab = { 0 : 'A', 1 : 'T', 2 : 'G', 3 : 'C'}
        def m_mismatch(x,y) :
            pass
        return m_mismatch
            
            
    
    