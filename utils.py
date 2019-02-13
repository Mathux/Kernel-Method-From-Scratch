#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:50:40 2019

@author: evrardgarcelon
"""

import pandas as pd
from config import *
import numpy as np
import sys
from kernel import Kernel
import svm
import kernel_lr
import kernel_knn 


# Load train data
def load_train(mat=True):
    
    print('Loading train set...')
    if mat :
        trainPaths = [path_to_train0_mat, path_to_train1_mat, path_to_train2_mat]
    else :
        trainPaths = [path_to_train0, path_to_train1, path_to_train2]
    trainPathReturns = [path_to_train_labels0, path_to_train_labels1, path_to_train_labels2]
    x_train = []
    y_train = []
    i = 0
    for path_features,path_labels in zip(trainPaths,trainPathReturns) : 
        y_train.append(pd.read_csv(path_labels))
        if mat :
            temp = pd.read_csv(path_features,sep = ' ', dtype = 'float64', header = None)
            temp['Id'] = y_train[i]['Id']
        else : 
            temp = pd.read_csv(path_features,sep = ',')
        i +=1
        x_train.append(temp)
        
    print('Done')
    return x_train, y_train


# Load test data
def load_test(mat=True, test_size = 1000):
    
    print('Loading test set...')
    if mat :
        testPaths = [path_to_test0_mat, path_to_test1_mat, path_to_test2_mat]
    else :
        testPaths = [path_to_test0, path_to_test1, path_to_test2]
    x_test = []
    for i,path_features in enumerate(testPaths) : 
        if mat :
            temp = pd.read_csv(path_features,sep = ' ',dtype = 'float64', header = None)
            temp['Id'] = np.linspace(i*test_size, (i+1)*test_size - 1, test_size, dtype = 'int')
        
        else : 
            temp = pd.read_csv(path_features,sep = ',')
        x_test.append(temp)
    print('Done')
    return x_test



# Split the train dataset 
def split_dataset(data, labels, split_val=0.1, seed=SEED):
    
    np.random.seed(seed)
    train,val = [],[]
    train_labels, val_labels = [],[]
    print('Splitting data set...')
    for i in range(len(data)) :
        data[i] = data[i].merge(labels[i], on = 'Id')
        n_samples = data[i].shape[0]
        train_temp = data[i].sample(frac = 1).iloc[:int((1 - split_val)*n_samples),:].sort_values(by = ['Id'])
        val_temp = data[i].sample(frac = 1).iloc[int((1 - split_val)*n_samples):,:].sort_values(by = ['Id'])
        train_temp.reset_index(inplace = True)
        val_temp.reset_index(inplace = True)
        train_labels_temp = train_temp[['Id','Bound']]
        val_labels_temp = val_temp[['Id','Bound']]
        data[i] = data[i].drop('Bound',axis = 1)
        train_temp,val_temp = train_temp.drop(['Bound','index'],axis = 1),val_temp.drop(['Bound','index'],axis = 1)
        train.append(train_temp)
        val.append(val_temp)
        train_labels.append(train_labels_temp)
        val_labels.append(val_labels_temp)
    print('Done')
    return train, val, train_labels, val_labels

def split_kfold(data, labels, n_fold, seed=SEED, shuffle = False):
    np.random.seed(seed)
    
    data = np.hstack([data,labels.reshape((len(labels),1))])
    if shuffle == True :
       np.random.shuffle(data,axis = 0)
    n_samples = data.shape[0]
    indexes = np.linspace(0,n_samples-1,n_fold+1,dtype = 'int')
    splitted = {}
    for j in range(n_fold) :
        temp = data[indexes[j] : indexes[j+1],:]
        train,labels = temp[:,:-1],temp[:,-1]
        splitted[j] = (train,labels)
    return list(splitted.values())        
    


# Tools to give the csv format
def submission(prediction,test_size = 1000) :  
    
    pred = pd.DataFrame(columns = ['Id','Bound'])
    for i in range(len(prediction)) :
        predictions = prediction[i]
        temp = pd.DataFrame(data = predictions,columns = ['Bound'])
        temp['Id'] = np.linspace(i*test_size, (i+1)*test_size - 1, test_size, dtype = 'int')
        temp = temp.reindex(columns=['Id','Bound'])
        pred = pred.append(temp)
    pred.reset_index(inplace = True)
    pred = pred.drop('index',axis = 1)
    pred.to_csv('predictions.csv',index = False)
    return None

def get_kernel(kernel, gamma = 1, offset = 1, dim = 1) :
    if kernel == 'linear' : 
        return Kernel().linear(offset)
    elif kernel == 'gaussian' :
        return Kernel().gaussian(gamma)
    elif kernel == 'sigmoid' :
        return Kernel().sigmoid(gamma,offset)
    elif kernel == 'polynomial' :
        return Kernel().polynomial(dim,offset)
    else :
        raise Exception('Invalid Kernel')
    
def get_kernel_parameters(kernel) :
    if kernel == 'linear' : 
        return ['offset']
    elif kernel == 'gaussian' :
        return ['gamma',]
    elif kernel == 'sigmoid' :
        return ['gamma','offset']
    elif kernel == 'polynomial' :
        return ['offset','dim']
    else :
        raise Exception('Invalid Kernel')

# Give some cool bar when we are waiting
def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def scale(X) :
    
    mu = np.mean(X,axis = 1)
    sigma = np.std(X,axis = 1)
    
    return (X-mu.reshape((mu.shape[0],1)))/sigma.reshape((sigma.shape[0],1))

if __name__ == "__main__":
    x_train,y_train = load_train(mat = True)
    x_test = load_test(mat = True)
    train,val,train_labels,val_labels = split_dataset(x_train, y_train)
    
