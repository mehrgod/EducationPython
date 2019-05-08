# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:07:24 2019

@author: mirza
"""

import numpy as np
from sklearn.cluster import KMeans

def test_sum_array():
    x = np.array([[1,  2,   3],
              [ 4,   5,  6],
              [ 7,   8,  9]])
    
    y = np.array([[1,  2,   3]])
    
    z = np.zeros(3)
    
    print y + z

def test_stack():
    X = np.zeros(shape =(1,2))
    X [0,0] = 1
    X [0,1] = 2
    Y = np.zeros(shape =(1,2))
    Y [0,0] = 3
    Y [0,1] = 4
    Z = np.hstack((X,Y))
    print Z

def test_csv():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/k10/c2d8/"
    with open(path + 'W1.csv') as file:
        array2d = [[float(digit) for digit in line.split(',')] for line in file]
        
    X = np.array(array2d)
    print X.shape
    print X
        
def test_sum():
    x = np.array([[1,  2,   3],
              [ 4,   5,  6],
              [ 7,   8,  9]])
    y = np.array([[3,  2,   1],
              [ 6,   5,  4],
              [ 9,   8,  7]])
    
    z = (x + y) / 2
    print z
    
    w = np.hstack((x, y))
    print w
    
    V = np.hstack(( (x + y) / 2 , np.hstack((x, y)) ))
    print V
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(V)
    
    print(kmeans.labels_)
    for l in kmeans.labels_:
        print l

def test_divide():
    X = np.array([
        [50, 30, 0, 10, 40, 20],
        [40, 0, 0, 10, 30, 50],
        [10, 10, 0, 50, 10, 0],
        [10, 0, 0, 40, 20, 20],
        [0, 10, 50, 40, 30, 40],
    ])
    
    a = X[:, 2:]
    b = X[:, :2]
    
    print a.shape
    print a
    print b.shape
    print b
    

def test_sum_one():
    X = np.array([
        [50, 30, 0, 10],
        [40, 0, 0, 10],
        [10, 10, 0, 50],
        [10, 0, 0, 40],
        [0, 10, 50, 40],
    ])
    
    print X / X.max(axis = 0)
    print X / X.sum(axis = 0)
    return X / X.sum(axis = 0)

def copy():
    x = np.array([[1000,  10,   0.5],
              [ 765,   5,  0.35],
              [ 800,   7,  0.09]])
    X = np.array([
        [50.0, 30.0, 0.0, 10.0],
        [40.0, 0.0, 0.0, 10.0],
        [10.0, 10.0, 0.0, 50.0],
        [10.0, 0.0, 0.0, 40.0],
        [0.0, 10.0, 50.0, 40.0],
    ])
    x = X
    x_normed = x / x.max(axis=0)
    print (x_normed)
    x_normed1 = x / x.sum(axis=0)
    print (x_normed1)

def multiplying():

    X = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
            ])

    X = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    A = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
    ])
    
    print A.shape
    
    B = np.array([
        [5, 3, 0, 1 ,2],
        [4, 0, 0, 1 ,3],
        [1, 1, 0, 5 ,1],
        [1, 0, 0, 4, 4],
    ])    
    
    wd = X[:,:2]
    print wd
    
    print X[:,2:]    
    
    hc = X[:2,:]
    print hc
    
if __name__ == "__main__":
    #test_sum()
    #test_divide()
    #test_csv()
    #test_stack()
    test_sum_array()