#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:17:56 2019

@author: evrardgarcelon
"""

import utils
import svm
import kernel_lr
import kernel_knn
import random_search

import scipy.stats


''' TO DO :
        
             - 3 sections to train each classifier with a random hyper param
               search and cross-validation with report on mean accuracy on the
               validation set after splitting
             
             - Une section averaging pour average les classifieurs
             
             - Une section predictions pour creer le fichier de predictions
'''
