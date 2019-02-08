#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:57:27 2019

@author: evrardgarcelon
"""

from kernel import Kernel
from cvxopt import matrix, solvers
import numpy as np

#### TODO: verifier opti, biais, tester function

class SVM(object) :
    
    def __init__(self, kernel = 'linear', C = 1.0, gamma = 0, dim = 0, offset = 0) :
        
        self.C = C
        self.kernel_type = kernel
        
        if kernel == 'linear' : 
            self.kernel = Kernel().linear()
        elif kernel == 'gaussian' :
            self.kernel = Kernel().gaussian(gamma)
        elif kernel == 'sigmoid' :
            self.kernel = Kernel().sigmoid(gamma,offset)
        elif kernel == 'polynomial' :
            self.kernel = Kernel().polynomial(dim,offset)
        else :
            raise Exception('Invalid Kernel')
        
    def fit(self,X, y, tol = 10**-5) :
        
        self.X = X
        y  = self.transform_label(y).astype('float64')
        n_samples,n_features = self.X.shape
        self.K = self.gram_matrix(X)
        temp = np.diag(y)
        Q = matrix(self.K)
        p = -1*matrix(y)
        h = matrix(np.hstack([self.C*np.ones(n_samples),np.zeros(n_samples)]))
        G = matrix(np.vstack([temp,-temp]))
        sol=solvers.qp(Q, p, G, h)
        self.alpha = np.ravel(sol['x'])
        self.support_vectors = (self.alpha>tol)
        self.alpha = self.alpha*self.support_vectors
    
    def predict(self,X) :
        
        n_samples,n_features = X.shape
        indexes = np.linspace(0,n_samples-1,n_samples,dtype = 'int')[self.support_vectors]
        projection = 0
        for i in indexes :
            projection += self.alpha[i]*self.kernel(self.X[i],X)
        return np.sign(projection)
            
    def score(self,X,y) :
        predictions = self.predict(X)
        return np.sum(y == predictions)/X.shape[0]
    
    def gram_matrix(self,X) :
        
        if self.kernel_type == 'linear' : 
            return self.kernel(X,X.T)
        else :
            n_samples = X.shape[0]
            K = np.zeros((n_samples,n_samples))
            for i in range(n_samples) :
                K[i,i] = self.kernel(X[i],X[i])
                for j in range(i+1,n_samples) :
                    K[i,j] = self.kernel(X[i],X[j])
                    K[j,i] = K[i,j]
            return K
                    
    
    def transform_label(self,y) :
        return 2*y - 1
    
        
        
        