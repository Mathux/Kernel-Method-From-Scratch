#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:43:18 2019

@author: evrardgarcelon
"""

#! /usr/bin/env python
# -*- coding: utf-8; mode: python -*-



print("\nLaunching script, importing modules ...")
from time import time
start_time_0 = time()  # Just to count the total running time at the end

import numpy as np


def gen_lin_separable_data():
    
    A = np.array([[2,1],[1,2]])
    mean1 = np.array([0, 2]) 
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * (-1)
    return np.dot(X1,A) + np.array([[1,1]]), y1, np.dot(X2,A) + np.array([[1,1]]), y2


def gen_non_lin_separable_data():
    """ Generate non-linearly separable training data in the 2-d case. """
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * (-1)
    return X1, y1, X2, y2



def gen_lin_separable_overlap_data():
    """ Generate training data in the 2-d case. """
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * (-1)
    return X1, y1, X2, y2

def gen_circular_data() :
    rayon1,rayon2 = 1,2 
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(np.zeros(2), cov, 100)
    X1 = rayon1*X1/np.linalg.norm(X1,axis = 1).reshape((100,1))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(np.zeros(2), cov, 100)
    X2 = rayon2*X2/np.linalg.norm(X2,axis = 1).reshape((100,1))
    y2 = np.ones(len(X2)) * (-1)
    return X1, y1, X2, y2
