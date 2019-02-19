#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:59:32 2019

@author: evrardgarcelon
"""

import numpy as np
import utils
from config import *

class KernelLogisticRegression(object) :
    
    def __init__(self,kernel = 'linear', 
                 la = 1, 
                 n_iter = 10**3, 
                 gamma = 1, 
                 dim = 1, 
                 offset = 1,
                 k = 1,
                 m = 1,
                 verbose = False) :
        
        self.la = la
        self.n_iter = n_iter
        self.kernel = utils.get_kernel(kernel, 
                                       gamma = gamma, 
                                       dim = dim, 
                                       offset = offset,
                                       k = k,
                                       m = m)
        self.kernel_type = kernel
        self.verbose = verbose
    
    def fit(self,X,y,tol = 10**-5,eps = 10**-5 ):
        
        self.X = X
        self.n_samples = self.X.shape[0]
        self.alpha = np.zeros(self.n_samples)
        old_alpha = self.alpha +1
        self.K = self.gram_matrix(self.X)
        t = 0
        if self.verbose :
            print('Fitting LogisticRegression...')
        while t < self.n_iter and np.linalg.norm(self.alpha - old_alpha) > tol :
            if self.verbose :
                utils.progressBar(t,self.n_iter)
            m = np.dot(self.K,self.alpha)
            W = self.sigmoid(y*m)*self.sigmoid(-y*m)
            z = m + y/np.maximum(self.sigmoid(y*m),0)
            W = np.diag(W)
            old_alpha = self.alpha
            self.alpha = self.WKRR(W,z,self.la,self.n_samples)
            t +=1
        if self.verbose :
            print('\n Done')
        if t == self.n_iter :
            print('Attention Convergence non atteinte')
        
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

    def WKRR(self,W,z,la,n) :
        alpha = np.linalg.solve(n*la*np.eye(n) + np.dot(W,self.K),np.dot(W,z))
        return  alpha
    
    def sigmoid(self,x) :
        return 1/(1+np.exp(-x))
        
    def predict(self,X) :
        n_samples = X.shape[0]
        projection = np.zeros(n_samples)
        if  not self.kernel_type in string_kernels :
            for j in range(n_samples) : 
                for i in range(self.n_samples) :
                    projection[j] += self.alpha[i]*self.kernel(self.X[i],X[j])
        else : 
            big_X = np.hstack([self.X,X])
            temp_K = self.gram_matrix(big_X)
            temp_K = temp_K[self.n_samples:,:self.n_samples].T
            for j in range(n_samples) :
                for i in range(self.n_samples) :
                    projection[j] = self.alpha[i]*temp_K[i,j]
        proba = self.sigmoid(projection)
        return 2.*(proba >= 1/2) - 1.
            
    def score(self,X,y) :
        predictions = self.predict(X)
        return np.sum(y == predictions)/X.shape[0]
    
    def recall_and_precision(self,X,y) :
        predictions = self.predict(X)
        tp = np.sum((predictions == 1.)*(y == 1.))
        fn = np.sum((predictions == -1.)*(y == 1.))
        fp = np.sum((predictions == 1.)*(y == -1.))
        return tp/(fn+tp),tp/(fp+tp)
