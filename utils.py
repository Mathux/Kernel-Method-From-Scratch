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
    for path_features,path_labels in zip(trainPaths,trainPathReturns) : 
        y_train.append(pd.read_csv(path_labels))
        temp = pd.read_csv(path_features,sep = ' ',dtype = 'float64', header = None)
        temp['Id'] = y_train[0]['Id']
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
        temp = pd.read_csv(path_features,sep = ' ',dtype = 'float64', header = None)
        temp['Id'] = np.linspace(i*test_size, (i+1)*test_size - 1, test_size, dtype = 'int')
        x_test.append(temp)
    print('Done')
    return x_test



# Split the train dataset into training and validation (keep different date)
def split_dataset(data, labels, split_val=0.1, seed=SEED):
    np.random.seed(seed)
    
    data = data.merge(labels, on = 'Id')

    dates = data["date"].unique().copy()
    n_dates = len(dates)
    all_index = np.arange(n_dates)
    np.random.shuffle(all_index)

    train_index = all_index[int(split_val*n_dates):]
    val_index = all_index[0:int(split_val*n_dates)]

    train = data[data["date"].isin(dates[train_index])]
    val = data[data["date"].isin(dates[val_index])]
    
    train_labels = train[['ID','end_of_day_return']]
    val_labels = val[['ID','end_of_day_return']]
    
    train = train.drop('end_of_day_return',axis = 1)
    val = val.drop('end_of_day_return',axis = 1)
    
    return train, val, train_labels, val_labels


# Tools to give the csv format
def submission(prediction, ID = None) :  
    
    if isinstance(prediction,pd.core.frame.DataFrame) :
        prediction.to_csv('predictions.csv', index=False)
    else : 
        pred = pd.DataFrame()
        pred['ID'] = ID
        pred['end_of_day_return'] = prediction
        pred.to_csv('predictions.csv', index=False)
        


# Give some cool bar when we are waiting
def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


if __name__ == "__main__":
    x_train,y_train = load_train()
    x_test = load_test()
    
