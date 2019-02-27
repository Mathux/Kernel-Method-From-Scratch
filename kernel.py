#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:57:37 2019

@author: evrardgarcelon
"""


import numpy as np
from itertools import product
import utils
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigs
import trie_dna

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
            for i in range(len(x) - self.k +1) :
                for j,b in enumerate(self.mers) : 
                    if x[i:i+self.k] == b :
                        phi_x[j] += 1
                    if y[i:i+self.k] == b :
                        phi_y[j] += 1
            return np.dot(phi_x,phi_y)
        return compute_kernel
    
    def get_kernel_matrix(self,X) :
        n = len(X)
        K = np.zeros((n,n))
        phis = []
        for x in X :
            phi = np.zeros(len(self.mers))
            for i in range(len(x) - self.k +1) :
                for j,mer in enumerate(self.mers) :
                    if x[i:i+self.k] == mer :
                        phi[j] += 1
            phis.append(phi)
        for i in range(n) :
            for j in range(i,n) :
                K[i,j] = np.dot(phis[i],phis[j])
                K[j,i] = K[i,j]
        # return utils.normalize_kernel(K)
        return K
  

class MismatchKernel(object) :
    def __init__(self,k,m) :
        self.k = k
        self.m = m
        self.mers = [(''.join(c)) for c in product('ACGT', repeat=self.k)]

    def kernel(self) :
        def compute_kernel(x,y) :
            phi_x = np.zeros(len(self.mers))
            phi_y = np.zeros(len(self.mers))
            for i in range(len(x) - self.k + 1):
                x_kmer = x[i:i + self.k]
                y_kmer = y[i:i + self.k]
                for  j,b in enumerate(self.mers) :
                    phi_x[j] += 1*(nb_diff(x_kmer,b)<=self.m)
                    phi_y[j] += 1*(nb_diff(y_kmer,b)<=self.m)
            return np.dot(phi_x,phi_y)
        return compute_kernel
    
    def get_kernel_matrix(self, X) :
        n = len(X)
        K = np.zeros((n,n))
        phis = []
        for x in X :
            phi = np.zeros(len(self.mers))
            for i in range(len(x) - self.k + 1):
                x_kmer = x[i:i + self.k]
                for  j,b in enumerate(self.mers) :
                    phi[j] += 1*(nb_diff(x_kmer,b)<=self.m)
            phis.append(phi)
        for i in range(n) :
            for j in range(i,n) :
                K[i,j] = np.dot(phis[i],phis[j])
                K[j,i] = K[i,j]
        #return utils.normalize_kernel(K)
        return K
        

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
    
class LAKernel() :
    
    def __init__(self,e=11, d=1, beta=0.5):
        self.S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])
        self.e = e
        self.d = d
        self.beta = beta

    def affine_align(self,x, y) : 
        e = self.e 
        d = self.d
        beta = self.beta
        S = self.S
        x, y = format(x)-1, format(y)-1
        n_x, n_y = len(x), len(y)
        M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))]*5
        for i in range(1, n_x):
            for j in range(1, n_y):
                M[i, j] = np.exp(beta * S[x[i], y[j]]) * (1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1])
                X[i, j] = np.exp(beta * d) * M[i-1, j] + np.exp(beta * e) * X[i-1, j]
                Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
                X2[i, j] = M[i-1, j] + X2[i-1, j]
                Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
        return (1/beta) * np.log(1 + X2[n_x, n_y] + Y2[n_x, n_y] + M[n_x, n_y])


    def Smith_Waterman(self,x, y, e=11, d=1, beta=0.5):
        S = self.S
        x, y = format(x) - 1, format(y) - 1
        n_x, n_y = len(x), len(y)
        M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
        for i in range(1, n_x):
            for j in range(1, n_y):
                M[i, j] = np.exp(beta * S[x[i], y[j]]) * max(1, X[i - 1, j - 1], Y[i - 1, j - 1], M[i - 1, j - 1])
                X[i, j] = max(np.exp(beta * d) * M[i - 1, j], np.exp(beta * e) * X[i - 1, j])
                Y[i, j] = max(np.exp(beta * d) * M[i, j - 1], np.exp(beta * d) * X[i, j - 1], np.exp(beta * e) * Y[i, j - 1])
                X2[i, j] = max(M[i - 1, j], X2[i - 1, j])
                Y2[i, j] = max(M[i, j - 1], X2[i, j - 1], Y2[i, j - 1])
        return (1/beta) * np.log(max(1, X2[n_x, n_y], Y2[n_x, n_y], M[n_x, n_y]))


    def get_LA_K(self,X, e=11, d=1, beta=0.5, smith=0, eig=1):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
            for j, y in tqdm(enumerate(X.loc[:, 'seq']), total=n-i):
                if j >= i:
                    K[i, j] = self.Smith_Waterman(x, y, e, d, beta) if smith else self.affine_align(x, y)
                    K[j, i] = K[i, j]
        K1 = deepcopy(K)
        if eig == 1:
            vp = np.min(np.real(eigs(K1)[0]))
            s = vp if vp < 0 else 0
            np.fill_diagonal(K1, np.diag(K1) - s * np.ones(n))
        else:
            for i in tqdm(range(K1.shape[0]), desc='Empirical kernel'):
                for j in range(i, n):
                    K1[i, j] = np.dot(K[i], K[j])
                    K1[j, i] = K1[i, j]
        return K

 
def format(x):
    """
    Transform string 'AGCT' to list [1, 3, 2, 4]
    :param x: string, DNA sequence
    :return: np.array, array of ints with 'A':1, 'C':2, 'G':3, 'T':4
    """
    return np.array(list(x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')), dtype=np.int64)
                   
if __name__ == '__main__' :
    
    X,y = utils.load_train(mat = False)
    X,y = utils.process_data(X,y)
    X,y = X[0],y[0]
    import time
    debut = time.clock()
    kernel = MismatchKernel(6,2).get_kernel_matrix(X)
    fin = time.clock()
    print('Temps calcul Mismatch Kernel  (3,1) brute force : ',fin - debut, 'secondes')
    debut = time.clock()
    t = trie_dna.Trie()
    kernel,_,_ = t.dfs(X,6,2)
    fin = time.clock()
    print('Temps calcul Mismatch Kernel  (3,1) Trie : ',fin - debut, 'secondes')

    
    

        

            
    
    