#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:56:13 2019

@author: evrardgarcelon
"""
import numpy as np
import utils
import heapq as hq
from config import *
class KernelKNN(object) :
    
    def __init__(self, n_neighbors = 5, kernel = 'linear', gamma = 1, dim = 1, offset = 0, k = 3, m = 1) :
        
        self.n_neighbors = n_neighbors
        self.kernel = utils.get_kernel(kernel, gamma = 1, dim = 1, offset = 0, k = m, m = m)
        self.kernel_type = kernel
    
    def fit(self,X,y) :
        
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
    
    def predict(self,X) :
        
        n_test = X.shape[0]
        predictions = np.zeros(n_test)
        if self.kernel_type in string_kernels:
            big_X = np.hstack([self.X,X])
            temp_K = self.kernel.get_kernel_matrix(big_X)
        else : 
            temp_K = None
        for j in range(n_test) :
            predictions[j] = self.majority_vote(X[j], j, temp_K)
        return predictions.astype('float64')
    
    def majority_vote(self, x, j, K) :
        
        nearest_neighbors = self.compute_nearest_neighbors(x, j, K)
        majority = 0
        for j in nearest_neighbors :
            if self.y[j] > 0 :
                majority +=1
            else :
                majority += -1
        if majority > 0 :
            return 1.
        elif majority < 0 :
            return -1.
        else :
            return np.random.choice(np.array([-1.,1.]))
            
    def compute_nearest_neighbors(self,x, j, K) :
        distance = []
        nearest_neighbors = []
  
        for i in range(self.n_samples) :
            if self.kernel_type in string_kernels :
                d = K[i,i] + K[self.n_samples + j, self.n_samples + j] - 2*K[i,self.n_samples + j]
            else :
                d = self.kernel(self.X[i],self.X[i]) + self.kernel(x,x) - 2*self.kernel(self.X[i],x)
            hq.heappush(distance,(d,i))
        for j in range(self.n_neighbors) :
            nearest_neighbors.append(hq.heappop(distance)[1])
        return nearest_neighbors
    
    def score(self,X,y) :
        predictions = self.predict(X)
        return np.sum(y == predictions)/X.shape[0]
    
    def recall_and_precision(self,X,y) :
        predictions = self.predict(X)
        tp = np.sum((predictions == 1.)*(y == 1.))
        fn = np.sum((predictions == -1.)*(y == 1.))
        fp = np.sum((predictions == 1.)*(y == -1.))
        return tp/(fn+tp),tp/(fp+tp)
            
        
        
    
            