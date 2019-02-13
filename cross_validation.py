#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:00:46 2019

@author: evrardgarcelon
"""

import numpy as np
import utils


class CrossValidation(object):
    
    def __init__(self,X,y,estimator,n_fold = 5):
        self.accuracy = np.zeros(n_fold)
        self.recall = np.zeros(n_fold)
        self.precision = np.zeros(n_fold)
        self.splitted_data = utils.split_kfold(X,y,n_fold = 5,shuffle = False)
        for k in range(n_fold) :
            data_to_stack = []
            labels_to_stack = []
            for j in range(n_fold) :
                if j != k :
                    data_to_stack.append(self.splitted_data[j][0])
                    labels_to_stack.append(self.splitted_data[j][1])
            data = np.vstack(data_to_stack)
            labels = np.hstack(labels_to_stack)
            estimator.fit(data,labels)
            self.accuracy[k] = estimator.score(self.splitted_data[k][0],self.splitted_data[k][1])
            self.recall[k],self.precision[k] = estimator.recall_and_precision(self.splitted_data[k][0],self.splitted_data[k][1])
        
        self.f1_score = 2*self.precision*self.recall/(self.precision+self.recall)
            
    
    def mean_acc(self):
        return np.mean(self.accuracy)
    
    def std_acc(self):
        return np.std(self.accuracy)
    
    def mean_recall_score(self) :
        return np.mean(self.recall)
    
    def std_recall_score(self):
        return np.std(self.recall)
    
    def mean_precision_score(self) :
        return np.mean(self.precision)
    
    def std_precision_score(self):
        return np.std(self.precision)
    
    def mean_f1_score(self) :
        return np.mean(self.f1_score)
    
    def std_f1_score(self) :
        return np.std(self.f1_score)
            