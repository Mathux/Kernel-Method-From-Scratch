#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:57:37 2019

@author: evrardgarcelon
"""

import numpy as np


class Kernel(object) :

    def __init__(self) :
        pass
        
    def linear(self,offset = 1) :
        return lambda x,y : np.dot(x,y) + offset
    
    def gaussian(self,gamma) :
        return lambda x,y : np.exp(-gamma*np.dot(x-y,x-y))
    
    def sigmoid(self,gamma, offset) :
        return lambda x,y : np.tanh(gamma*np.dot(x,y) + offset)
    
    def polynomial(self,dim,offset) :
        return lambda x,y : (offset + np.dot(x,y))**dim
    
    
    