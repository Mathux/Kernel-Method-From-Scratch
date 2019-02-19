#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:57:37 2019

@author: evrardgarcelon
"""


import numpy as np
from itertools import product
import utils
import trie_dna
from tqdm import tqdm

def nb_diff(x,y) :
    nb_diff = 0
    for char1,char2 in zip(x,y) :
        if char1 != char2 :
            nb_diff +=1
    return nb_diff

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
    
    def laplace(self,gamma) :
        return lambda x,y  : np.exp(np.linalg.norm(x-y)*gamma)
    


class SpectralKernel(object) :
    
    def __init__(self,k) :
        self.k = k
        self.mers = [(''.join(c)) for c in product('ACGT', repeat=self.k)]
    
    def kernel(self) :
        def compute_kernel(x,y) :
            phi_x = np.zeros(len(self.mers))
            phi_y = np.zeros(len(self.mers))
            for j,b in enumerate(self.mers) : 
                phi_x[j] += 1*(b in x)
                phi_y[j] += 1*(b in y)
            return np.dot(phi_x,phi_y)
        return compute_kernel
    
    def get_kernel_matrix(self,X) :
        n = len(X)
        K = np.zeros((n,n))
        phis = []

        for x in X :
            phi = np.zeros(len(self.mers))
            for j,mer in enumerate(self.mers) :
                phi[j] += 1*(mer in x)
            phis.append(phi)
        for i in range(n) :
            for j in range(i,n) :
                K[i,j] = np.dot(phis[i],phis[j])
                K[j,i] = K[i,j]
        return utils.normalize_kernel(K)
  

class MismatchKernel(object) :
 # TO DO   
    def __init__(self,k,m) :
        self.k = k
        self.m = m
    
    
    def kernel(self) :
        mers = [(''.join(c)) for c in product('ACGT', repeat=self.k)]
        def compute_kernel(x,y) :
            phi_x = np.zeros(len(mers))
            phi_y = np.zeros(len(mers))
            for i in range(len(x) - self.k + 1):
                x_kmer = x[i:i + self.k]
                y_kmer = y[i:i + self.k]
                for  j,b in enumerate(mers) :
                    phi_x[j] += 1*(nb_diff(x_kmer,b)<=self.m)
                    phi_y[j] += 1*(nb_diff(y_kmer,b)<=self.m)
            return np.dot(phi_x,phi_y)
        return compute_kernel
    
#    def get_kernel_matrix(self, X) :
#        K,_,_ = trie_dna.Trie().dfs(X, self.k, self.m)
#        return utils.normalize_kernel(K)
        

class WDKernel() :
    
    def __init__(self,d) :
        self.d = d
        self.beta = 2*np.linspace(1,d,d,dtype = 'int')[::-1]/(d*(d+1))
        
    def kernel(self) :
        
        def compute_kernel(x,y) :
            k_xy = 0
            assert len(x) == len(y)
            l = len(x)
            for k in range(self.d) :
                temp = 0
                for i in range(l-k) :
                    temp += 1*(x[i:i+k] == y[i:i+k])
                k_xy += self.beta[k]*temp
            return k_xy
        
        return compute_kernel
        
    def get_kernel_matrix(self,X) :
        n = len(X)
        K = np.zeros((n,n))
        f = self.kernel()
        for i in tqdm(range(n)) :
            for j in range(i,n) :
                K[i,j] = f(X[i],X[j])
                K[j,i] = K[i,j]
        return utils.normalize_kernel(K)
                    

    

        

            
    
    