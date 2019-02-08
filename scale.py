#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:16:34 2019

@author: evrardgarcelon
"""

import numpy as np

def scale(X) :
    
    mu = np.mean(X,axis = 1)
    sigma = np.std(X,axis = 1)
    
    if sigma > 0 :
        return (X-mu)/sigma
    else :
        return X-mu