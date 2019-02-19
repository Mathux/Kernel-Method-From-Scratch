#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:57:27 2019

@author: evrardgarcelon
"""

from cvxopt import matrix, solvers
import numpy as np
import utils
from kernel import *
from config import *

solvers.options['show_progress'] = False

class SVM(object) :
    
    def __init__(self, kernel = 'linear', C = 1.0, gamma = 1, dim = 0, offset = 1, k = 3, m = 1, d = 1) :
        
        self.C = C
        self.kernel_type = kernel
        self.kernel = utils.get_kernel(kernel, gamma = gamma, dim = dim, offset = offset, k= k, m = m, d = d)
        
    def fit(self,X, y, tol = 10**-5) :

        self.X = X
        self.n_samples = (self.X).shape[0]
        self.K = self.gram_matrix(X)
        temp = np.diag(y)
        Q = matrix(self.K)
        p = -1*matrix(y)
        h = matrix(np.hstack([self.C*np.ones(self.n_samples),np.zeros(self.n_samples)]))
        G = matrix(np.vstack([temp,-temp]))
        sol=solvers.qp(Q, p, G, h)
        self.alpha = np.ravel(sol['x'])
        self.support_vectors = np.linspace(0,self.n_samples-1,self.n_samples,dtype = 'int')[(np.abs(self.alpha)>tol)]
        self.alpha = self.alpha
    
    def predict(self,X) :
        n_samples = X.shape[0]
        projection = np.zeros(n_samples)
        if  not self.kernel_type in string_kernels:
            for j in range(n_samples) : 
                for i in self.support_vectors :
                    projection[j] += self.alpha[i]*self.kernel(self.X[i],X[j])
        else : 
            big_X = np.hstack([self.X,X])
            temp_K = self.gram_matrix(big_X)
            temp_K = temp_K[self.n_samples:,:self.n_samples].T
            for j in range(n_samples) :
                for i in self.support_vectors :
                    projection[j] = self.alpha[i]*temp_K[i,j]
        return np.sign(projection).astype('float64')
            
    def score(self,X,y) :
        predictions = self.predict(X)
        return np.sum(y == predictions)/X.shape[0]
    
    def recall_and_precision(self,X,y) :
        predictions = self.predict(X)
        tp = np.sum((predictions == 1.)*(y == 1.))
        fn = np.sum((predictions == -1.)*(y == 1.))
        fp = np.sum((predictions == 1.)*(y == -1.))
        return tp/(fn+tp),tp/(fp+tp)
    
    def gram_matrix(self,X) :
        
        if self.kernel_type == 'linear': 
            return self.kernel(X,X.T)
        elif not self.kernel_type in string_kernels :
            n_samples = X.shape[0]
            K = np.zeros((n_samples,n_samples))
            for i in range(n_samples) :
                K[i,i] = self.kernel(X[i],X[i])
                for j in range(i+1,n_samples) :
                    K[i,j] = self.kernel(X[i],X[j])
                    K[j,i] = K[i,j]
            K = utils.normalize_kernel(K)
        else :
            K = self.kernel.get_kernel_matrix(X)
            return K
        
    
        
        
        