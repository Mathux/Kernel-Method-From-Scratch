#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:56:13 2019

@author: evrardgarcelon
"""
import numpy as np
from utils import *
import heapq as hq

class KernelKNN(object) :
    
    def __init__(self, n_neighbors = 5, kernel = 'linear', gamma = 1, dim = 1, offset = 0, scale = False) :
        
        self.n_neighbors = n_neighbors
        self.kernel = get_kernel(kernel, gamma = 1, dim = 1, offset = 0)
        self.scale = scale
        self.kernel_type = kernel
    
    def fit(self,X,y) :
        
        self.X = X
        self.y = self.transform_label(y)
        self.n_samples,self.n_features = X.shape
    
    def predict(self,X) :
        
        n_test,_ = X.shape
        predictions = np.zeros(n_test)
        for j in range(n_test) :
            predictions[j] = self.majority_vote(X[j])
        return predictions
    
    def majority_vote(self,x) :
        
        nearest_neighbors = self.compute_nearest_neighbors(x)
        majority = 0
        for j in nearest_neighbors :
            if self.y[j] > 0 :
                majority +=1
            else :
                majority += -1
            
        if majority > 0 :
            return 1
        elif majority < 0 :
            return 0
        else :
            print('lol')
            return np.random.randint(0,2)
            
    def compute_nearest_neighbors(self,x) :
        
        distance = []
        nearest_neighbors = []
        for i in range(self.n_samples) :
            d = self.kernel(self.X[i],self.X[i]) + self.kernel(x,x) - 2*self.kernel(self.X[i],x)
            hq.heappush(distance,(d,i))
        for j in range(self.n_neighbors) :
            nearest_neighbors.append(hq.heappop(distance)[1])
        return nearest_neighbors
    
    def transform_label(self,y) :
        return 2*y - 1
    
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
            
        
        
    
            