#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:59:32 2019

@author: evrardgarcelon
"""

import numpy as np
import utils
from scipy.linalg import sqrtm
import pylab as plt

class KernelLogisticRegression(object) :
    
    def __init__(self,kernel = 'linear', la = 10**2, n_iter = 10**3, gamma = 1, dim = 1, offset = 1, scale = False) :
        
        self.la = la
        self.n_iter = n_iter
        self.kernel = utils.get_kernel(kernel, gamma = gamma, dim = dim, offset = offset)
        self.scale = scale
        self.kernel_type = kernel
    
    def fit(self,X,y,tol = 10**-5,eps = 10**-5 ):
        
        if self.scale :
            self.X = utils.scale(X)
        else :
            self.X = X
        y  = self.transform_label(y)
        self.n_samples,self.n_features = self.X.shape
        self.alpha = np.zeros(self.n_samples)
        old_alpha = self.alpha +1
        self.K = self.gram_matrix(self.X)
        t = 0
        error = []
        print('Fitting LogisticRegression...')
        while t < self.n_iter and np.linalg.norm(self.alpha - old_alpha) > tol :
            #utils.progressBar(t,self.n_iter)
            m = np.dot(self.K,self.alpha)
            W = self.sigmoid(y*m)*self.sigmoid(-y*m)
            z = m + y/(self.sigmoid(-y*m) + eps)
            W = np.diag(W)
            old_alpha = self.alpha
            self.alpha = self.WKRR(W,z,self.la,self.n_samples)
            error.append(np.linalg.norm(self.alpha - old_alpha))
            t +=1
        print('Done')
        error = np.array(error)
        plt.figure(5)
        plt.plot(error,marker = '+')
        if t == self.n_iter :
            print('Attention Convergence non atteinte')
        
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

    def WKRR(self,W,z,la,n) :
        sqrt_W = sqrtm(W)
        inv_reg_W = np.linalg.inv(la*np.eye(n) + np.dot(sqrt_W,np.dot(self.K,sqrt_W)))
        return np.dot(sqrt_W,np.dot(inv_reg_W,np.dot(sqrt_W,z)))
    
    def sigmoid(self,x) :
        return 1/(1+np.exp(-x))
    
    def transform_label(self,y) :
        return 2*y-1
    
    def predict(self,X) :
        n_samples,n_features = X.shape
        projection = np.zeros(n_samples)
        for j in range(n_samples) : 
            for i in range(self.n_samples) :
                projection[j] += self.alpha[i]*self.kernel(self.X[i],X[j])
        proba = self.sigmoid(projection)
        return 2*(proba >= 1/2) - 1
            
    def score(self,X,y) :
        predictions = self.predict(X)
        return np.sum(y == predictions)/X.shape[0]
    
    def recall_and_precision(self,X,y) :
        y = self.transform_label(y)
        predictions = self.predict(X).astype('int')
        tp = np.sum((predictions == 1)*(y == 1))
        fn = np.sum((predictions == -1)*(y == 1))
        fp = np.sum((predictions == 1)*(y == -1))
        return tp/(fn+tp),tp/(fp+tp)
    
            
        