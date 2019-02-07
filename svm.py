#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:57:27 2019

@author: evrardgarcelon
"""

from kernel import Kernel
from cvxopt import matrix, solvers
import numpy as np

class SVM(object) :
    
    def __init__(self, kernel = 'linear', C = 1.0, gamma = 0, dim = 0, offset = 0) :
        
        self.C = C
        
        if kernel == 'linear' : 
            self.kernel = Kernel.linear()
        elif kernel == 'gaussian' :
            self.kernel = Kernel.gaussian(gamma)
        elif kernel == 'sigmoid' :
            self.kernel = Kernel.sigmoid(gamma,offset)
        elif kernel == 'polynomial' :
            self.kernel = Kernel.polynomial(dim,offset)
        else :
            raise Exception('Invalid Kernel')
        
    def fit(self,X, y, tol = 10**-5) :
        self.X = X
        n_samples,n_features = self.X.shape
        self.K = self.gram_matrix(X)
        temp = np.diag(y)
        Q = self.K
        p = self.transform_label(y)
        h = np.hstack([self.C*np.ones(n_samples),np.zeros(n_samples)])
        G = np.vstack([temp,-temp])
        A = np.zeros((1,1))
        b = np.zeros(1)
        sol=solvers.qp(Q, p, G, h, A, b)
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
        pass
    
    def gram_matrix(self,X) :
        
        return self.kernel(X,X.T)
    
    def transform_label(self,y) :
        
        return 2*y - 1
    
        
        
        