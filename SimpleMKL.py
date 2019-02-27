#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:56:05 2019

@author: evrardgarcelon
"""

import numpy as np
import svm
import kernel
from config import *
import utils

class SimpleMKL() :
    
    def __init__(self, kernels = [], weights = None, C = 1, gamma = 1, dim = 1, k = 1, m = 1, e = 11, beta = 1, d = 1, tol = 10**-5, n_iter = 100) :
        
        self.kernels = kernels
        if weights == None :
            self.weights = np.ones(len(kernels))/len(kernels)
        else :
            self.weights = np.array(weights)
        self.kernel_parameters = {'gamma' : gamma, 
                                  'dim' : dim, 
                                  'k' : k, 
                                  'm' : m,
                                  'beta' : beta,
                                  'e' : e,
                                  'd' : d}
        self.C = C
        self.n_iter = n_iter
        self.tol = tol
        print(self.weights)
    def fit(self,X,y) :
        
        self.kernels_matrix = self.compute_matrices(X)
        gap = 1
        n = 0
        m = 0
        K = self.compute_K(self.weights, self.kernels_matrix)
        while gap > self.tol and n < self.n_iter :
            n+=1
            mu = np.argmax(self.weights)
            J, grad_J, D, _ = self.compute_J_grad_J_D(K, y, mu)
            print('J = ', J, 'grad_J = ', grad_J, 'D = ', D)
            J_cross = 0
            D_cross = D
            d_cross = self.weights
            while J_cross < J and m < self.n_iter :
                m +=1
                self.weights = d_cross
                D = D_cross
                temp = 1*self.weights/D
                mask = (D>=0)
                temp[mask] =  - np.inf
                nu = np.argmin(-temp)
                gamma_max = -self.weights[nu]/D[nu]
                d_cross = self.weights + gamma_max*D
                D_cross[mu] = D[mu] - D[nu]
                D_cross[nu] = 0
                K_cross = self.compute_K(d_cross, self.kernels_matrix)
                J_cross,_,_,_ = self.compute_J_grad_J_D(K_cross,y,0)
            gamma,alpha = self.line_search(self.weights,D,gamma_max,y)
            self.weights = self.weights + gamma*D
            self.weights = self.weights/np.sum(self.weights)
            K = self.compute_K(self.weights, self.kernels_matrix)
            temp_max = -np.inf
            for i in range(len(self.kernels)) :
                temp = np.dot(alpha,np.dot(self.kernels_matrix[i],alpha))
                if temp > temp_max :
                    temp_max = temp
            gap = temp_max - np.dot(alpha,np.dot(K,alpha))
            print('gap = ', gap, 'nb iterations = ', n)
                
    def compute_matrices(self, X) :
        temp = []
        for j,ker in enumerate(self.kernels) :
            n = len(X)
            temp_kernel = utils.get_kernel(ker, **self.kernel_parameters)
            if not kernel in string_kernels :
                M = np.zeros((n,n))
                for i in range(n) :
                    for j in range(i,n) :
                        M[i,j] = temp_kernel(X[i],X[j])
                        M[j,i] = M[i,j]
            else :
                M = temp_kernel.get_kernel_matrix(X)
            temp.append(M)
        return temp
    
    def compute_J_grad_J_D(self, K, y, mu, eps = 10**-20) :
        clf = svm.SVM(kernel_matrix = K, **self.kernel_parameters)
        clf.fit_kernel(y)
        alpha = clf.alpha
        # alpha or y*alpha ? 
        J = -1/2*np.dot(alpha,np.dot(K,alpha)) + np.sum(np.dot(np.diag(y),alpha))
        n_kernels = len(self.kernels)
        grad_J = np.zeros(n_kernels)
        for m in range(n_kernels) :
            grad_J[m] = -1/2*np.dot(alpha,np.dot(self.kernels_matrix[m],alpha))
        D = np.zeros(n_kernels)
        for m in range(n_kernels) :
            if np.abs(self.weights[m]) <= eps and (grad_J[m] - grad_J[mu] > 0) :
                D[m] = 0
            elif m != mu and self.weights[m] > 0 :
                D[m] = grad_J[mu] - grad_J[m]
            elif m == mu :
                temp = grad_J - grad_J[mu]
                mask = (self.weights > 0)*(np.linspace(0,n_kernels-1,n_kernels,dtype = 'int') != mu)
                D[m] = np.sum(temp[mask])
        return J,grad_J,D,alpha
    
    def compute_K(self,d,matrices) :
        K = 0*matrices[0]
        for dd,M in zip(d,matrices) :
            K += dd*M
        return K
            
    def line_search(self,d, D, gamma_max, y, a = 0.2, b = 0.9, n_iter = 10) :
        n = 0
        gamma = gamma_max
        K = self.compute_K(d,self.kernels_matrix)
        f0,grad_J,_,_ = self.compute_J_grad_J_D(K, y, 0)
        m = np.dot(grad_J,D)
        K = self.compute_K(d + gamma*D,self.kernels_matrix)
        f1,_,_,alpha = self.compute_J_grad_J_D(K, y, 0)
        #print('f0 = ',f0, 'f1 = ', f1, 'n = ', n, 'm = ', m)
        while f1 > f0 + a*gamma*m and n < n_iter :
            gamma = gamma*b
            n +=1
            K = self.compute_K(d + gamma*D,self.kernels_matrix)
            f1,_,_,alpha = self.compute_J_grad_J_D(K, y, 0)  
        return gamma,alpha
    
if __name__ == '__main__' :
    kernels = ['gaussian','linear']
    clf = SimpleMKL(kernels)
    clf.fit(X,y)
                