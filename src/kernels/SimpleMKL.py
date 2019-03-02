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
    
    def __init__(self, kernels = [], weights = None, C = 1, gamma = 1, dim = 1, k = 1, m = 1, e = 11, beta = 1, d = 1, tol = 10**-10, n_iter = 50) :
        
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
        
    def fit(self,X,y, tol_sv = 10**-4) :
        self.X = X
        self.y = y
        self.kernels_matrix = self.compute_matrices(X)
        gap = 1
        n = 0
        m = 0
        K = self.compute_K(self.weights, self.kernels_matrix)
        while gap > self.tol and n < self.n_iter :
            n+=1
            mu = np.argmax(self.weights)
            J, grad_J, D, _ = self.compute_J_grad_J_D(K, y, mu,self.weights)
            J_cross = 0
            D_cross = 1*D
            d_cross = 1*self.weights
            m = 0
            print('J = ', J)
            while J_cross < J and m < self.n_iter :
                self.weights = d_cross
                D = 1*D_cross
                if m > 0 :
                    J = J_cross
                temp = 1*self.weights/D
                mask = (D>=0)
                temp[mask] =  -1*np.inf
                nu = np.argmin(-temp)
                gamma_max = -temp[nu]
                #print('D = ', D, 'gamma = ',gamma_max)
                d_cross = 1*self.weights + gamma_max*D
                D_cross[nu] = 0
                D_cross = self.normalize_D(D_cross,mu)
                K_cross = self.compute_K(d_cross, self.kernels_matrix)
                #print('D_cross = ',D_cross, 'D = ', D, 'nu = ',nu, 'mu = ',mu)
                
                J_cross,_,_,_ = self.compute_J_grad_J_D(K_cross,y,mu,d_cross)
                #print('J_cross = ', J_cross)
                #print('J_cross - J = ', J_cross - J, 'J = ', J, 'm = ', m)
                m +=1
            gamma,alpha = self.line_search(self.weights,D,gamma_max,y)
            self.alpha = alpha
            self.weights = self.weights + gamma*D
            #print('d = ', np.sum(self.weights))
            #self.weights = self.weights/np.sum(self.weights)
            K = self.compute_K(self.weights, self.kernels_matrix)
            temp_max = -np.inf
            for i in range(len(self.kernels)) :
                temp = np.dot(alpha,np.dot(self.kernels_matrix[i],alpha))
                if temp > temp_max :
                    temp_max = temp
            gap = temp_max - np.dot(alpha,np.dot(K,alpha))
        self.sv = np.linspace(0,len(self.alpha) - 1,len(self.alpha),dtype = 'int')[np.abs(self.alpha) > tol_sv]
        
            #print('gap = ', gap, 'n = ', n)
                
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
    
    def normalize_D(self,D,mu) :
        temp_D = 1*D
        temp_D[mu] = -np.sum(temp_D)
        return temp_D
    
    def compute_J_grad_J_D(self, K, y, mu, d,eps = 10**-20) :
        clf = svm.SVM(kernel_matrix = K, **self.kernel_parameters)
        clf.fit_kernel(y)
        alpha = clf.alpha 
        J = -1/2*np.dot(alpha,np.dot(K,alpha)) + np.sum(np.dot(np.diag(y),alpha))
        n_kernels = len(self.kernels)
        grad_J = np.zeros(n_kernels)
        for m in range(n_kernels) :
            grad_J[m] = -1/2*np.dot(alpha,np.dot(self.kernels_matrix[m],alpha))
        reduced_grad = grad_J - grad_J[mu]
        mask = (d <= eps)*(reduced_grad>0)
        reduced_grad[mask] = 0
        reduced_grad = -reduced_grad
        reduced_grad[mu] = -np.sum(reduced_grad)
        return J,grad_J,reduced_grad,alpha
    
    def compute_K(self,d,matrices) :
        K = 0*matrices[0]
        for dd,M in zip(d,matrices) :
            K += dd*M
        return K
            
    def line_search(self,d, D, gamma_max, y, a = 0.2, b = 0.5, n_iter = 100) :
        n = 0
        gamma = gamma_max
        K = self.compute_K(d,self.kernels_matrix)
        f0,grad_J,_,_ = self.compute_J_grad_J_D(K, y, 0,d)
        m = np.dot(grad_J,D)
        K = self.compute_K(d + gamma*D,self.kernels_matrix)
        f1,_,_,alpha = self.compute_J_grad_J_D(K, y, 0,d+gamma*D)
        #print('f0 = ',f0, 'f1 = ', f1, 'n = ', n, 'm = ', m)
        while f1 > f0 + a*gamma*m and n < n_iter :
            gamma = gamma*b
            n +=1
            K = self.compute_K(d + gamma*D,self.kernels_matrix)
            f1,_,_,alpha = self.compute_J_grad_J_D(K, y, 0,d + gamma*D)  
        #print(n)
        return gamma,alpha
    
    def predict(self,X) :
        
        f = np.zeros(len(X))
        for j,x in enumerate(X) :
            proj = 0
            for i in self.sv :
                for m,w in enumerate(self.weights) :
                    proj += self.alpha[i]*w*utils.get_kernel(self.kernels[m], **self.kernel_parameters)(self.X[i],x)
            f[j] = proj
        return np.sign(f)
                    
if __name__ == '__main__' :
    kernels = ['gaussian','linear']
    clf = SimpleMKL(kernels, n_iter = 10)
    clf.fit(X,y)
    preds = clf.predict(X)
    import pylab as plt
    plt.figure(1)
    plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='o')
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', marker='^')
    for j in range(len(preds)) :
        if preds[j] == 1 :
            plt.scatter(X[j][0],X[j][1],color = 'magenta',marker = '+')
        elif preds[j] == -1 :
            plt.scatter(X[j][0],X[j][1],color = 'cyan',marker = 'x')
        else :
            print('pb')
                    